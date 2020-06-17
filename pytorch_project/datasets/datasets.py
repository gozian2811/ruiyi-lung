from collections import OrderedDict
from skimage.io import imread
from torchvision import transforms as T   
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import sys
import glob
import copy
import math
import random
import torch
sys.path.append("/home/fyl/programs/lung_project")
from toolbox import BasicTools as bt
from toolbox import MITools as mt
from toolbox import LIDCTools as lt
from toolbox import SPHTools as st
from toolbox import Data_Augmentation as da

class Basic_Dataset(object):
	def __init__(self, filelist, data_size=64, mode=None, file_retrieval=False):
		if type(filelist)==str:
			self.filelist = bt.filelist_load(filelist)
		elif type(filelist)==list:
			self.filelist = filelist
		else:
			print('invalid filelist type:{}' .format(type(filelist)))
			exit()
		if 'lidc' in filelist[0]:
			self.dataset_name = 'LIDC'
		elif 'sph' in filelist[0]:
			self.dataset_name = 'SPH'
		elif 'luna' in filelist[0]:
			self.dataset_name = 'LUNA'
		elif 'tianchild' in filelist[0]:
			self.dataset_name = 'TLD'
		self.data_size = data_size
		self.mode = mode
		self.file_retrieval = file_retrieval

	def data_preprocess(self, input_image, normalization=None, num_channels=0):
		if type(input_image)!=np.ndarray:
			input_image = np.stack(input_image, axis=0)
		if normalization=='CT':
			image = mt.medical_normalization(input_image, input_copy=False)
		elif normalization=='CT_chest':
			image = mt.medical_normalization(input_image, max_bound=1000, min_bound=-1000, pixel_mean=0.25, crop=False, input_copy=False)
		elif normalization=='CT_lung':
			image = mt.medical_normalization(input_image, max_bound=400, min_bound=-1000, pixel_mean=0.25,  crop=True, input_copy=False)
		elif normalization == 'image':
			image_max = (np.zeros(1, dtype=input_image.dtype)-1)[0]
			#image_max = 65535
			image = input_image / float(image_max)
		else:
			image = input_image
		if num_channels>0:
			image = np.expand_dims(image, axis=0)
		if num_channels>1:
			image = image.repeat(num_channels, axis=0)
		image = torch.from_numpy(image).float()
		return image

	def patch_concatenate(self):
		patchs = []
		for i in range(len(self)):
			volume = self[i][0]
			patchs.append(volume[:,int(volume.shape[1]/2)])
		concpatchs = torch.stack(patchs, dim=0)
		return concpatchs

	def filenames(self):
		filenames = []
		for filepath in self.filelist:
			filename = os.path.splitext(os.path.basename(filepath))[0]
			filenames.append(filename)
		return filenames

class Augmentation_Parameters_Rough(object):
	def __init__(self, translation_num=0, translation_range=(-3,3), rotation_augment=False, flip_augment=False, noise_range=(0, 0)):
		self.translation_num = translation_num
		self.translation_range = translation_range
		self.rotation_augment = rotation_augment
		self.flip_augment = flip_augment
		self.noise_range = noise_range
	
	def num_augmentation(self):
		if self.rotation_augment:
			rotation_num = da.rotation_augment_number
		else:
			rotation_num = 0
		if self.flip_augment:
			flip_num = da.flip_augment_number
		else:
			flip_num = 0
		return (1 + self.translation_num) * (1 + rotation_num + flip_num)
		
class Augmentation_Parameters(object):
	def __init__(self, translation_num=0, translation_range=(-3,3), rotation_num=0, flip_num=0, shear_num=0,  noise_range=(0, 0)):
		self.translation_num = translation_num
		self.translation_range = translation_range
		self.rotation_num = rotation_num
		self.flip_num = flip_num
		self.shear_num = shear_num
		self.noise_range = noise_range
	
	def num_augmentation(self):
		return max(self.translation_num, 1) * max(self.rotation_num, 1) * max(self.flip_num, 1) * max(self.shear_num, 1)
		
	def data_augment(self, image_overbound, aug_idx):
		image_augmented = da.extract_augment_random(aug_idx, image_overbound, self.data_size, self.translation_num, self.translation_range, self.rotation_num, self.flip_num, self.shear_num)
		image_noised = da.add_noise(image_augmented, self.noise_range)
		#image_noised = image_augmented + np.random.uniform(*self.noise_range, image_augmented.shape)
		return image_noised.astype(image_overbound.dtype)

class LIDC_Pathology_Dataset(Dataset, Basic_Dataset, Augmentation_Parameters):
	def __init__(self, filelist, data_size, mode='binclass', label_collection=None, file_retrieval=False, **augargs):
		#the parameter 'mode' lies in {'diameter', 'malignancy', 'binclass', 'bincregression'}
		super(LIDC_Pathology_Dataset, self).__init__(filelist, data_size, mode, file_retrieval)
		super(Basic_Dataset, self).__init__(**augargs)
		self.label_collection = label_collection
		if label_collection is not None:
			if self.mode in ('binclass', 'bincregression'):
				self.label_item = 'malignancy'
			else:
				self.label_item = self.mode
			self.statistics = self.label_statistic()

	def label_statistic(self):
		if self.label_collection is None:
			return None
		else:
			statistics = {}
			collection = pd.read_csv(self.label_collection)
			statistics['max'] = collection[self.label_item].max()
			magnitude = 10 ** int(math.log(statistics['max']) / math.log(10))
			statistics['max'] = int(math.ceil(statistics['max'] / float(magnitude)) * magnitude)
			statistics['min'] = collection[self.label_item].min()
			return statistics

	def __len__(self):
		num_aug = self.num_augmentation()
		return len(self.filelist) * num_aug
		#return 6
		
	def __getitem__(self, idx):
		num_aug = self.num_augmentation()
		data_idx = int(idx / num_aug)
		aug_idx = idx % num_aug

		#label_dict = {'mg':0,'bn':1}
		data_path = self.filelist[data_idx]
		volume_overbound = np.load(data_path)
		#volume_augmented = da.extract_augment_random(aug_idx, volume_overbound, self.data_size, self.translation_num, self.translation_range, self.rotation_num, self.flip_num)
		#volume = volume_augmented + np.random.random_sample(volume_augmented.shape) * random.choice(self.noise_range)
		volume = self.data_augment(volume_overbound, aug_idx)
		volume = self.data_preprocess(volume, 'CT_lung', num_channels=1)
		'''
		volume = mt.medical_normalization(volume_augmented, input_copy=False)
		volume = volume + np.random.random_sample(volume.shape) * random.choice(self.noise_range)
		volume = volume.reshape(1, volume.shape[0], volume.shape[1], volume.shape[2])
		volume = torch.from_numpy(volume).float()
		'''
		if self.label_collection is None:
			label = lt.sample_malignancy_label(data_path, malignant_positive=False, mode=self.mode)
		else:
			label = lt.sample_statistic_label(data_path, self.label_collection, statistic_item=self.label_item)
			if self.label_item=='malignancy':
				label = lt.malignancy_label(label, malignant_positive=False, mode=self.mode)
			if self.mode==self.label_item:
				#normalize the label value to the range [-1, 1]
				label = (label - self.statistics['max'] / 2).astype(np.float32)
		output = {'data': volume, 'label': label}
		if self.file_retrieval:
			output['path'] = data_path
			#return {'data': volume, 'label': label, 'path': data_path}
		#else: return {'data': volume, 'label': label}
		return output
		
class LIDC_Pathology_RandomPairs_Dataset(LIDC_Pathology_Dataset):
	def __init__(self, filelist, data_size, mode='binclass', file_retrieval=False, translation_num=0, translation_range=(0, 0), rotation_num=0, flip_num=0, noise_range=(0, 0)):
		super(LIDC_Pathology_RandomPairs_Dataset, self).__init__(filelist, data_size, mode, file_retrieval, translation_num, translation_range, rotation_num, flip_num, noise_range)
		num_aug = self.num_augmentation()
		self.idxlist1 = random.sample([i for i in range(len(self.filelist)*num_aug)], len(self.filelist)*num_aug)
		self.idxlist2 = random.sample([i for i in range(len(self.filelist)*num_aug)], len(self.filelist)*num_aug)
		
	def __getitem__(self, idx):
		idx1 = self.idxlist1[idx]
		idx2 = self.idxlist2[idx]
		data1 = super(LIDC_Pathology_RandomPairs_Dataset, self).__getitem__(idx1)
		data2 = super(LIDC_Pathology_RandomPairs_Dataset, self).__getitem__(idx2)
		return data1, data2
		
	def shuffle(self):
		random.shuffle(self.idxlist1)
		random.shuffle(self.idxlist2)
		'''
		choicelist = copy.copy(self.idxlist1)
		self.idxlist2 = []
		for i in range(len(self.idxlist1)):
			fileidx1 = int(self.idxlist1[i]/num_aug)
			for r in range(len(self.idxlist1)-len(self.idxlist2)):
				idx2 = random.choice(choicelist)
				fileidx2 = int(idx2/num_aug)
				if fileidx1!=fileidx2 or self.idxlist1[i]==idx2: break
			if 'idx2' not in dir(): idx2 = random.choice(choicelist)
			choicelist.remove(idx2)
			self.idxlist2.append(idx2)
		'''

class LIDC_Pathology_Relation_Dataset(LIDC_Pathology_Dataset):
	def __init__(self, filelist, data_size, mode='binclass', file_retrieval=False, translation_num=0, translation_range=(0, 0), rotation_num=0, flip_num=0, noise_range=(0, 0)):
		super(LIDC_Pathology_Relation_Dataset, self).__init__(filelist, data_size, mode, file_retrieval, translation_num, translation_range, rotation_num, flip_num, noise_range)
		num_aug = self.num_augmentation()
		#self.idxlist1 = random.sample([i for i in range(len(self.filelist)*num_aug)], len(self.filelist)*num_aug)
		#self.idxlist2 = random.sample([i for i in range(len(self.filelist)*num_aug)], len(self.filelist)*num_aug)
		self.idxlist = [(i, j) for i in range(len(self.filelist)*num_aug) for j in range(len(self.filelist)*num_aug)]

	def __len__(self):
		return len(self.idxlist)
		
	def __getitem__(self, idx):
		#idx1 = self.idxlist1[idx]
		#idx2 = self.idxlist2[idx]
		idx1, idx2 = self.idxlist[idx]
		data1 = super(LIDC_Pathology_Relation_Dataset, self).__getitem__(idx1)
		data2 = super(LIDC_Pathology_Relation_Dataset, self).__getitem__(idx2)
		return data1, data2
		
	#def shuffle(self):
		#random.shuffle(self.idxlist1)
		#random.shuffle(self.idxlist2)
		#random.shuffle(self.idxlist)
		'''
		choicelist = copy.copy(self.idxlist1)
		self.idxlist2 = []
		for i in range(len(self.idxlist1)):
			fileidx1 = int(self.idxlist1[i]/num_aug)
			for r in range(len(self.idxlist1)-len(self.idxlist2)):
				idx2 = random.choice(choicelist)
				fileidx2 = int(idx2/num_aug)
				if fileidx1!=fileidx2 or self.idxlist1[i]==idx2: break
			if 'idx2' not in dir(): idx2 = random.choice(choicelist)
			choicelist.remove(idx2)
			self.idxlist2.append(idx2)
		'''


class SPH_Pathology_Dataset(Dataset, Basic_Dataset, Augmentation_Parameters):
	def __init__(self, filelist, data_size, mode=None, file_retrieval=False, **augargs):
		super(SPH_Pathology_Dataset, self).__init__(filelist, data_size, mode, file_retrieval)
		super(Basic_Dataset, self).__init__(**augargs)
	
	def __len__(self):
		num_aug = self.num_augmentation()
		return len(self.filelist) * num_aug

	def __getitem__(self, idx):
		num_aug = self.num_augmentation()
		data_idx = int(idx / num_aug)
		aug_idx = idx % num_aug

		data_path = self.filelist[data_idx]
		volume_overbound = np.load(data_path)
		#volume_augmented = da.extract_augment_random(aug_idx, volume_overbound, self.data_size, self.translation_num, self.translation_range, self.rotation_num, self.flip_num)
		#volume = volume_augmented + np.random.random_sample(volume_augmented.shape) * random.choice(self.noise_range)
		volume = self.data_augment(volume_overbound, aug_idx)
		volume = self.data_preprocess(volume, 'CT_lung', num_channels=1)

		output = {'data': volume}
		if self.mode is not None:
			label = st.sample_pathology_label(data_path, mode=self.mode)
			output['label'] = label
		if self.file_retrieval:
			output['path'] = data_path
			#return {'data': volume, 'label': label, 'path': data_path}
		#else: return {'data': volume, 'label': label}
		return output

class Pathology_Slicewise_Dataset(Dataset, Basic_Dataset, Augmentation_Parameters):
	def __init__(self, filelist, data_size, mode='binclass', patch_mode='ensemble', num_channels=3, file_retrieval=False, **augargs):
		super(Pathology_Slicewise_Dataset, self).__init__(filelist, data_size, mode, file_retrieval)
		super(Basic_Dataset, self).__init__(**augargs)
		self.num_channels = num_channels
		self.patch_mode = patch_mode
		self.patchdict = {'oa': [], 'hs': [], 'hvv': []}
		for file in self.filelist:
			filename = os.path.splitext(os.path.basename(file))[0]
			patchcat = filename.split('_')[-2]
			self.patchdict[patchcat].append(file)
		
	def __len__(self):
		num_aug = self.num_augmentation()
		return len(self.patchdict['oa']) * num_aug
		
	def __getitem__(self, idx):
		num_aug = self.num_augmentation()
		data_idx = int(idx / num_aug)
		aug_idx = idx % num_aug
		
		if self.patch_mode=='ensemble':
			image = []
			oapath = self.patchdict['oa'][data_idx]
			data_path = os.path.splitext(os.path.basename(oapath))[0].replace('_oa', '')
			for pkey in self.patchdict.keys():
				dpath = oapath.replace('_oa', '_'+pkey)
				image_overbound = imread(dpath, as_gray=True)
				'''
				image_noised = image_overbound + np.random.random_sample(image_overbound.shape) * random.choice(self.noise_range)
				#image_reshaped = np.moveaxis(image_noised, 2, 0)
				#image_tensor = torch.from_numpy(image_reshaped).float()
				image_preprocessed = self.data_preprocess(image_noised, normalization=False)
				image.append(image_preprocessed)
				'''
				image.append(image_overbound)
				#data_path.append(dpath)
			label_path = oapath
		else:
			data_path = self.patchdict[self.patch_mode][data_idx]
			image = imread(data_path, as_gray=True)
			label_path = data_path
			'''
			image_noised = image_overbound + np.random.random_sample(image_overbound.shape) * random.choice(self.noise_range)
			#image_reshaped = np.moveaxis(image_noised, 2, 0)
			#image = torch.from_numpy(image_reshaped).float()
			image = self.data_preprocess(image_noised, normalization=False)
			'''

		if isinstance(image, np.ndarray):
			#image_augmented = da.extract_augment_random(aug_idx, image, self.data_size, self.translation_num, self.translation_range, self.rotation_num, self.flip_num, self.shear_num)
			#image_noised = image_augmented + np.random.random_sample(image_augmented.shape) * random.choice(self.noise_range)
			image_noised = self.data_augment(image, aug_idx)
			image = self.data_preprocess(image_noised, normalization='image', num_channels=self.num_channels)
		else:
			#image_augmented = []
			#for nc in range(self.num_channels):
			#	image_augmented.append(da.extract_augment_random(aug_idx, image, self.data_size, self.translation_num, self.translation_range, self.rotation_num, self.flip_num))
			#image_augmented = np.stack(image_augmented, axis=1)
			image_augmented = da.extract_augment_random(aug_idx, image, self.data_size, self.translation_num, self.translation_range, self.rotation_num, self.flip_num, self.shear_num)
			for i in range(len(image_augmented)):
				#image_noised = image_augmented[i] + np.random.uniform(*self.noise_range, image_augmented[i].shape)
				#image_noised[image_noised<0] = 0
				#image_noised = np.minimum(image_noised, np.zeros_like(image_augmented)-1)
				#image_noised = image_noised.astype(image_augmented[i].dtype)
				image_noised = da.add_noise(image_augmented[i], self.noise_range)
				image[i] = self.data_preprocess(image_noised, normalization='image', num_channels=self.num_channels)

		if 'lidc' in label_path:
			label = lt.sample_malignancy_label(label_path, malignant_positive=False, mode=self.mode)
		elif 'sph' in label_path:
			label = st.sample_pathology_label(label_path, mode=self.mode)
			
		#if self.file_retrieval:
		#	return image, label, data_path
		#else: return image, label
		output = {'data': image, 'label': label}
		if self.file_retrieval:
			output['path'] = data_path
		return output

class Pathology_Multiview_Dataset(Dataset, Basic_Dataset, Augmentation_Parameters):
	def __init__(self, filelist, data_size, mode='binclass', num_channels=3, file_retrieval=False, **augargs):
		super(Pathology_Multiview_Dataset, self).__init__(filelist, data_size, mode, file_retrieval)
		super(Basic_Dataset, self).__init__(**augargs)
		self.num_channels = num_channels
		self.filedict = OrderedDict()
		for file in self.filelist:
			filename = os.path.splitext(os.path.basename(file))[0]
			dataname = '_'.join(filename.split('_')[:-1])
			if dataname in self.filedict.keys():
				self.filedict[dataname].append(filename)
			else:
				self.filedict[dataname] = [filename]
		
	def __len__(self):
		num_aug = self.num_augmentation()
		return len(self.filedict) * num_aug

	def __getitem__(self, idx):
		num_aug = self.num_augmentation()
		data_idx = int(idx / num_aug)
		aug_idx = idx % num_aug

		image_views = []
		data_name = self.filedict.keys()[data_idx]
		data_files = self.filedict[data_name]
		for data_file in data_files:
			image_overbound = imread(data_file, as_gray=True)
			image_augmented = self.data_augment(image_overbound, aug_idx)
			image_views.append(image_augmented)

		if 'lidc' in data_files[0]:
			label = lt.sample_malignancy_label(data_name, malignant_positive=False, mode=self.mode)
		elif 'sph' in data_files[0]:
			label = st.sample_pathology_label(data_name, mode=self.mode)
			
		output = {'data': image_views, 'label': label}
		if self.file_retrieval:
			output['path'] = data_name
		return output

class Detection_Dataset(Dataset, Basic_Dataset, Augmentation_Parameters):
	def __init__(self, filelist, data_size, mode=None, file_retrieval=False, **augargs):
		super(Detection_Dataset, self).__init__(filelist, data_size, mode, file_retrieval)
		super(Basic_Dataset, self).__init__(**augargs)
		num_augment = self.num_augmentation()

		#if type(filelist)==str:
		#	filelist = bt.filelist_load(filelist)
		label_dict = {'noduleclass':1, 'stripeclass':5, 'arterioclass':31, 'lymphnodecalclass':32}
		self.norm_mode_dict = {"noduleclass":'CT_lung', "stripeclass":'CT_lung', "arterioclass":'CT_chest', "lymphnodecalclass":'CT_chest'}
		self.positive_list = []
		self.negative_list = []
		self.datamap_list = []
		for file in self.filelist:
			filename = os.path.basename(file)
			filenamenoext = os.path.splitext(filename)[0]
			fileinfo = filenamenoext.split('_')
			classlabel = fileinfo[2]
			annotationlabel = fileinfo[-1]
			if annotationlabel=='annotation':
				if mode is None or mode not in label_dict.keys() or classlabel==str(label_dict[mode]):
					self.positive_list.append(file)
					self.datamap_list.extend([(1, len(self.positive_list)-1, i) for i in range(num_augment)])
				else:
					self.negative_list.append(file)
					self.datamap_list.append((0, len(self.negative_list)-1))
			elif annotationlabel=='nonannotation':
				self.negative_list.append(file)
				self.datamap_list.append((0, len(self.negative_list)-1))
			else: print("unaccepted name of file {}" .format(file))
		#random.shuffle(self.datamap_list)
		
	def __len__(self):
		return len(self.datamap_list)
		
	def __getitem__(self, idx):
		datamap = self.datamap_list[idx]
		if datamap[0]:
			label, data_idx, aug_idx = datamap
			data_path = self.positive_list[data_idx]
			volume_overbound = np.load(data_path)
			volume_fitted = da.extract_augment_random(aug_idx, volume_overbound, self.data_size, self.translation_num, self.translation_range, self.rotation_num, self.flip_num)
		else:
			label, data_idx = datamap
			data_path = self.negative_list[data_idx]
			volume_fitted = np.load(data_path)
			if np.linalg.norm(np.int_(volume_fitted.shape)-self.data_size)!=0:
				volume_fitted = mt.local_crop(volume_fitted, np.rint(np.array(volume_fitted.shape)/2).astype(int), self.data_size)	#the sample should be cropped to fit the data size

		#volume = volume_fitted + np.random.random_sample(volume_fitted.shape) * random.choice(self.noise_range)
		volume = da.add_noise(volume_fitted, self.noise_range)
		if self.mode not in self.norm_mode_dict.keys():
			norm_mode = "CT_lung"
		else:
			norm_mode = self.norm_mode_dict[self.mode]
		volume = self.data_preprocess(volume, norm_mode, num_channels=1)
		'''
		volume_normalized = mt.medical_normalization(volume_fitted, input_copy=False)
		volume_noisy = volume_normalized + np.random.random_sample(volume_normalized.shape) * random.choice(self.noise_range)
		volume_reshaped = volume_noisy.reshape(1, volume_noisy.shape[0], volume_noisy.shape[1], volume_noisy.shape[2])
		volume = torch.from_numpy(volume_reshaped).float()
		'''
		output = {'data': volume, 'label': label}
		if self.file_retrieval:
			output['path'] = data_path
			#return {'data': volume, 'label': label, 'path': data_path}
		#else: return {'data': volume, 'label': label}
		return output

class Detection_DirectSample_Dataset(Dataset, Basic_Dataset, Augmentation_Parameters):
	def __init__(self, filelist, data_size, mode=None, file_retrieval=False, **augargs):
		super(Detection_Dataset, self).__init__(filelist, data_size, mode, file_retrieval)
		super(Basic_Dataset, self).__init__(**augargs)
		num_augment = self.num_augmentation()

