#-*-coding:utf-8-*-

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import SimpleITK as sitk
import os
import cv2
import time
import glob
import random
import matplotlib.pyplot as plt
from collections import OrderedDict
from skimage import measure, feature
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from . import BasicTools as bt
from . import MITools as mt
from . import Data_Augmentation as da

ios_stat = []

def dcm_count(file_list):
	dcm_count = 0
	for file_name in file_list:
		if file_name.split('.')[1] == 'dcm':
			dcm_count += 1
	return dcm_count

def find_scan(path):
	path_list = []   #paths that contain .dcm files
	#dcm_counts = []   #num of .dcm files

	subdir_list = os.listdir(path)
	for subdir in subdir_list:
		subdir = path + '/' + subdir
		subsubdir_list = os.listdir(subdir)
		for subsubdir in subsubdir_list:
			subsubdir = subdir + '/' + subsubdir
			file_list = os.listdir(subsubdir)
			count = dcm_count(file_list)
			if count>=50:
				path_list.append(subsubdir)
				#dcm_counts.append(count)
	return path_list
	#return path_list[dcm_counts.index(max(dcm_counts))]

def retrieve_scan(data_path):
	if data_path[-1] != '/':
		data_path += '/'
	scan_list = []
	patient_list = os.listdir(data_path)
	for patient in patient_list:
		patient_path = data_path + patient
		if os.path.isdir(patient_path):
			scan_paths = find_scan(patient_path)
			scan_list.extend(scan_paths)
	return scan_list
	
def filelist_remove_uncertain(filelist):
	'''
	while 1:
		uc_exist = False
		for file in filelist:
			if sample_malignancy_label(file)<0:
				filelist.remove(file)
				uc_exist = True
				break
		if not uc_exist:
			break
	'''
	for f in range(len(filelist), 0, -1):
		if sample_malignancy_label(filelist[f-1])<0:
			filelist.pop(f-1)
	return filelist

def filelist_training(sample_path, subpaths=["npy"], remove_uncertain=True, shuffle=False, cross_fold=5, val_fold=0, test_fold=5):
	print("{} fold dataset division" .format(cross_fold))
	files = []
	for subpath in subpaths:
		sdir = os.path.join(sample_path, subpath, "*.npy")
		files.extend(glob.glob(sdir))

	if remove_uncertain:
		files = filelist_remove_uncertain(files)
	if shuffle:
		random.shuffle(files)
	filelists, folddict = bt.foldlist(files, cross_fold, {'val':val_fold, 'test':test_fold})
	
	return filelists, folddict

def sample_statistic_label(data_path, label_statistic, statistic_item):
	data_info = os.path.splitext(os.path.basename(data_path))[0].split('_')
	patient_id = data_info[0]
	nodule_id = data_info[1]
	statistics = pd.read_csv(label_statistic)
	data_indices = np.where(np.logical_and(statistics.patient_id==patient_id, statistics.nodule_id.astype(type(nodule_id))==nodule_id))[0]
	if len(data_indices)>1:
		print("more than one corresponding data in statistics")
	label = statistics[statistic_item].astype(np.float32)[data_indices[0]]
	return label

def sample_malignancy_label(filepath, malignant_positive=False, mode='binclass'):
	#the parameter 'mode' lies in {'binclass', 'bincregression'}.
	#if the mode is 'bincregression', the parameter 'malignant_positive' is of no use.
	label_dict = {'mg':int(malignant_positive), 'bn':1-int(malignant_positive), 'uc':-1}
	filename = os.path.basename(filepath)
	label = filename.split('_')[2]
	if label in label_dict.keys():
		return label_dict[label]
	else:
		return malignancy_label(label, malignant_positive, mode)
	'''
	elif mode in ('sinclass', 'binclass'):
		malcode = int(float(label)+0.5)
		if malcode>3: return int(malignant_positive)
		elif malcode<3: return 1-int(malignant_positive)
		else: return -1
	elif mode=='bincregression':
		return float(label)
	else:
		print('label mode incorrect in sample_malignancy_label(.)')
		exit()
	'''

def malignancy_label(label, malignant_positive=False, mode='binclass'):
	if mode in ('sinclass', 'binclass'):
		malcode = int(float(label)+0.5)
		if malcode>3: return int(malignant_positive)
		elif malcode<3: return 1-int(malignant_positive)
		else: return -1
	else:
		return float(label)
	
def category_statistics(filelist):
	num_malignant = 0
	num_benign = 0
	num_uncertain = 0
	for file in filelist:
		if sample_malignancy_label(file)==0:
			num_malignant += 1
		elif sample_malignancy_label(file)==1:
			num_benign += 1
		else:
			num_uncertain += 1
	return num_malignant, num_benign, num_uncertain

def parseXML(scan_path):
	'''
	parse xml file

	args:
	xml file path

	output:
	nodule list
	[{nodule_id, roi:[{z, sop_uid, xy:[[x1,y1],[x2,y2],...]}]}]
	'''
	file_list = os.listdir(scan_path)
	for file in file_list:
		if file.split('.')[1] == 'xml':
			xml_file = file
			break
	prefix = "{http://www.nih.gov}"
	tree = ET.parse(scan_path +'/' + xml_file)
	root = tree.getroot()
	readingSession_list = root.findall(prefix + "readingSession")

	nodules = []
	nonnodules = []
	for session in readingSession_list:
		#print(session)
		unblinded_list = session.findall(prefix + "unblindedReadNodule")
		#print(unblinded_list)
		for unblinded in unblinded_list:
			nodule_id = unblinded.find(prefix + "noduleID").text
			edgeMap_num = len(unblinded.findall(prefix+"roi/"+prefix+"edgeMap"))
			if edgeMap_num > 1 and unblinded.find(prefix+"characteristics/") is not None:
			# it's segmentation label
				nodule_info = {}
				nodule_info['nodule_id'] = nodule_id
				nodule_info['characteristics'] = OrderedDict()
				nodule_info['characteristics']['subtlety'] = unblinded.find(prefix+"characteristics/"+prefix+"subtlety").text
				nodule_info['characteristics']['internalStructure'] = unblinded.find(prefix+"characteristics/"+prefix+"internalStructure").text
				nodule_info['characteristics']['calcification'] = unblinded.find(prefix+"characteristics/"+prefix+"calcification").text
				nodule_info['characteristics']['sphericity'] = unblinded.find(prefix+"characteristics/"+prefix+"sphericity").text
				nodule_info['characteristics']['margin'] = unblinded.find(prefix+"characteristics/"+prefix+"margin").text
				nodule_info['characteristics']['lobulation'] = unblinded.find(prefix+"characteristics/"+prefix+"lobulation").text
				nodule_info['characteristics']['spiculation'] = unblinded.find(prefix+"characteristics/"+prefix+"spiculation").text
				nodule_info['characteristics']['texture'] = unblinded.find(prefix+"characteristics/"+prefix+"texture").text
				nodule_info['characteristics']['malignancy'] = unblinded.find(prefix+"characteristics/"+prefix+"malignancy").text
				nodule_info['roi'] = []
				roi_list = unblinded.findall(prefix + "roi")
				for roi in roi_list:
					roi_info = {}
					roi_info['z'] = float(roi.find(prefix + "imageZposition").text)
					roi_info['sop_uid'] = roi.find(prefix + "imageSOP_UID").text
					roi_info['xy'] = []
					edgeMap_list = roi.findall(prefix + "edgeMap")
					for edgeMap in edgeMap_list:
						x = float(edgeMap.find(prefix + "xCoord").text)
						y = float(edgeMap.find(prefix + "yCoord").text)
						xy = [x, y]
						roi_info['xy'].append(xy)
					nodule_info['roi'].append(roi_info)
				nodules.append(nodule_info)
		nonnodule_list = session.findall(prefix + "nonNodule")
		for nonnodule in nonnodule_list:
			nonnodule_info = {}
			nonnodule_info['nonnodule_id'] = nonnodule.find(prefix + "nonNoduleID").text
			locus_info = {}
			locus_info['z'] = float(nonnodule.find(prefix + "imageZposition").text)
			locus_info['sop_uid'] = nonnodule.find(prefix + "imageSOP_UID").text
			locus = nonnodule.find(prefix + "locus")
			x = float(locus.find(prefix + "xCoord").text)
			y = float(locus.find(prefix + "yCoord").text)
			locus_info['xy'] = [x, y]
			nonnodule_info['locus'] = locus_info
			nonnodules.append(nonnodule_info)
	return nodules, nonnodules

def coord_trans(nodules, scan_path, target_spacing=None):
	'''
	transform z coord from world to voxel

	args:
	nodule: dict of nodule info, {nodule_id, roi:[{z, sop_uid, xy:[[x,y]]}]}
	scan_path

	output:
	nodules_boundary_coords:
	[{nodule_id, boundary_coords:[[x1,y1,z1],[x2,y2,z2],...]}]

	'''
	#reader = sitk.ImageSeriesReader()
	#dicom_names = reader.GetGDCMSeriesFileNames(scan_path)
	#reader.SetFileNames(dicom_names)
	#image = reader.Execute()
	#image_array = sitk.GetArrayFromImage(image)    #z, y, x
	image_array, image_info = mt.read_dicom_scan(scan_path)
	zpositions = image_info['zpositions']
	origin = image_info['origin']
	spacing = image_info['spacing']
	#origin = image.GetOrigin()[::-1]         #z, y, x
	#spacing = image.GetSpacing()[::-1]       #z, y, x
	if target_spacing is not None:
		image_array, spacing = mt.resample(image_array, spacing, target_spacing)
	spacing = np.array(spacing)
	deps = image_array.shape[0]
	cols = image_array.shape[1]
	rows = image_array.shape[2]

	transed_nodule_list = []
	nodule_num = 0
	for nodule in nodules:
		'''
		subtlety = int(nodule['characteristics']['subtlety'])
		internalStructure = int(nodule['characteristics']['internalStructure'])
		calcification = int(nodule['characteristics']['calcification'])
		sphericity = int(nodule['characteristics']['sphericity'])
		margin = int(nodule['characteristics']['margin'])
		lobulation = int(nodule['characteristics']['lobulation'])
		spiculation = int(nodule['characteristics']['spiculation'])
		texture = int(nodule['characteristics']['texture'])
		malignancy = int(nodule['characteristics']['malignancy'])
		'''
		#label_image = np.zeros((deps, cols, rows), dtype=int)  # the array saved binary label array
		boundary_coords = []
		for roi in nodule['roi']:
			roi['z'] = zpositions.index(roi['z'])	# trans z from world to voxel
			#roi['z'] = np.rint((roi['z'] - origin[0])/spacing[0])
			for xy in roi['xy']:
				boundary_coords.append([int(roi['z']), int(xy[1]), int(xy[0])])
		#		label_image[int(roi['z']), int(xy[1]), int(xy[0])] = 1  #boundary points in label image = 1
		#index = np.where(label_image==1)   # find boundary coords
		#for i in range(index[0].shape[0]):
		#	boundary_coords.append([index[0][i], index[1][i], index[2][i]])
		nodule_num += 1
		transed_nodule = {}
		transed_nodule['nodule_id'] = nodule_num
		#transed_nodule['nodule_id'] = nodule['nodule_id']
		'''
		transed_nodule['subtlety'] = subtlety
		transed_nodule['internalStructure'] = internalStructure
		transed_nodule['calcification'] = calcification
		transed_nodule['sphericity'] = sphericity
		transed_nodule['margin'] = margin
		transed_nodule['lobulation'] = lobulation
		transed_nodule['spiculation'] = spiculation
		transed_nodule['texture'] = texture
		transed_nodule['malignancy'] = malignancy
		'''
		transed_nodule['characteristics'] = OrderedDict()
		for charkey in nodule['characteristics'].keys():
			transed_nodule['characteristics'][charkey] = nodule['characteristics'][charkey]
		transed_nodule['boundary_coords'] = boundary_coords	#z, y, x
		transed_nodule_list.append(transed_nodule)

	return transed_nodule_list, image_array, spacing

def bounding_box(transed_nodule_dict):
	boundary_arr = np.array(transed_nodule_dict['boundary_coords'])
	col_max = boundary_arr.max(axis=0)
	col_min = boundary_arr.min(axis=0)
	return col_max, col_min

def duplicate_nodules(transed_nodule_list):
	#fuse annotations of the same nodule to one
	for i in range(len(transed_nodule_list)-1):
		for j in range(i+1, len(transed_nodule_list)):
			if transed_nodule_list[i]['nodule_id'] != transed_nodule_list[j]['nodule_id']:
				i_col_max, i_col_min = bounding_box(transed_nodule_list[i])
				j_col_max, j_col_min = bounding_box(transed_nodule_list[j])
				z_low = max(i_col_min[0], j_col_min[0])
				z_high = min(i_col_max[0], j_col_max[0])
				y_low = max(i_col_min[1], j_col_min[1])
				y_high = min(i_col_max[1], j_col_max[1])
				x_low = max(i_col_min[2], j_col_min[2])
				x_high = min(i_col_max[2], j_col_max[2])
				if z_low > z_high or y_low > y_high or x_low > x_high:
					#iou = 0
					ios = 0
				else:
					inter_area = (z_high - z_low + 1) * (y_high - y_low + 1) * (x_high - x_low + 1)
					i_bbox_area = (i_col_max[0] - i_col_min[0] + 1) * (i_col_max[1] - i_col_min[1] + 1) * (i_col_max[2] - i_col_min[2] + 1)
					j_bbox_area = (j_col_max[0] - j_col_min[0] + 1) * (j_col_max[1] - j_col_min[1] + 1) * (j_col_max[2] - j_col_min[2] + 1)
					#iou = inter_area / (i_bbox_area + j_bbox_area - inter_area)
					ios = inter_area / float(min(i_bbox_area, j_bbox_area))
				#if iou >= 0.4:
				if ios>=0.5:
					transed_nodule_list[j]['nodule_id'] = transed_nodule_list[i]['nodule_id']
				elif ios>0:
					print('nodule_overlap:{} {}' .format(transed_nodule_list[i]['nodule_id'], transed_nodule_list[j]['nodule_id']))
	return transed_nodule_list
					
def fill_hole(transed_nodule_list, deps, cols, rows):
	filled_nodule_list = []

	for nodule_dict in transed_nodule_list:
		filled_nodule = {}
		filled_nodule['nodule_id'] = nodule_dict['nodule_id']
		#filled_nodule['malignancy'] = nodule_dict['malignancy']
		filled_nodule['characteristics'] = nodule_dict['characteristics']
		filled_nodule['coords'] = []
		label_image = np.zeros((deps, cols, rows),  dtype=int)
		label_slices = np.zeros(deps, dtype=bool)
		for coord in nodule_dict['boundary_coords']:
			label_image[coord[0], coord[1], coord[2]] = 1
			label_slices[coord[0]] = True
		sliceinds = label_slices.nonzero()[0]
		for i in sliceinds:
			label_image[i,:,:] += fill_nodule(label_image[i,:,:])             # fill segmentation mask
		#for i in range(deps):
		#	if label_image[i].max()>0:
		#		label_image[i,:,:] += fill_nodule(label_image[i,:,:])             # fill segmentation mask
			#label_image[i,:,:] = binary_erosion(label_image[i,:,:]).astype(label_image[i,:,:].dtype)
		index = np.where(label_image>0)
		for i in range(index[0].shape[0]):
			filled_nodule['coords'].append([index[0][i], index[1][i], index[2][i]])
		filled_nodule['mask'] = label_image
		filled_nodule_list.append(filled_nodule)

	return filled_nodule_list

def fill_nodule(nodule_z):
	h, w = nodule_z.shape
	canvas = np.zeros((h + 2, w + 2), np.uint8)
	canvas[1:h + 1, 1:w + 1] = nodule_z.copy()
	mask = np.zeros((h + 4, w + 4), np.uint8)
	cv2.floodFill(canvas, mask, (0, 0), 1)
	canvas = canvas[1:h + 1, 1:w + 1].astype(np.bool)

	return (~canvas | nodule_z.astype(np.uint8))

def overall_mask(filled_nodule_list):
	omask = np.zeros(filled_nodule_list[0]['mask'].shape, dtype=bool)
	for nodule in filled_nodule_list:
		omask = np.logical_or(omask, nodule['mask'])
	return omask

def fuse_nodules(filled_nodule_list, overlap_threshold=0.3):
	adjmatrix = np.zeros((len(filled_nodule_list), len(filled_nodule_list)), dtype=float)
	for i in range(len(filled_nodule_list)):
		for j in range(i, len(filled_nodule_list)):
			intersection = np.logical_and(filled_nodule_list[i]['mask'], filled_nodule_list[j]['mask'])
			volis = np.count_nonzero(intersection)
			if volis>0:
				voli = np.count_nonzero(filled_nodule_list[i]['mask'])
				volj = np.count_nonzero(filled_nodule_list[j]['mask'])
				ios = volis / float(min(voli, volj))
				adjmatrix[i][j] = ios
				adjmatrix[j][i] = ios

	converge = False
	while not converge:
		converge = True
		for i in range(len(filled_nodule_list)-1):
			for j in range(i+1, len(filled_nodule_list)):
				if filled_nodule_list[i]['nodule_id'] != filled_nodule_list[j]['nodule_id']:
					#voli = np.count_nonzero(filled_nodule_list[i]['mask'])
					#volj = np.count_nonzero(filled_nodule_list[j]['mask'])
					#intersection = np.logical_and(filled_nodule_list[i]['mask'], filled_nodule_list[j]['mask'])
					#volis = np.count_nonzero(intersection)
					#ios = volis / float(min(voli, volj))
					ios = adjmatrix[i][j]
					if ios>0:
						ios_stat.append(ios)
						if ios>=overlap_threshold:
							min_id = min(filled_nodule_list[i]['nodule_id'], filled_nodule_list[j]['nodule_id'])
							filled_nodule_list[i]['nodule_id'] = min_id
							filled_nodule_list[j]['nodule_id'] = min_id
							converge = False
						else:
							print('nodule_overlap:{} {} ios:{}' .format(filled_nodule_list[i]['nodule_id'], filled_nodule_list[j]['nodule_id'], ios))
	np.save('ios_statistics.npy', np.array(ios_stat))
	return filled_nodule_list

def calc_union_freq(filled_nodule_list):
	union_nodule_list = []
	nodule_freqs = {}
	for i in range(len(filled_nodule_list)):
		if filled_nodule_list[i]['nodule_id'] not in nodule_freqs.keys():
			# the nodule has not been calculate union yet
			nodule_freqs[filled_nodule_list[i]['nodule_id']] = 1
		else:
			nodule_freqs[filled_nodule_list[i]['nodule_id']] += 1
	for nodule_id in nodule_freqs.keys():
		umask = np.zeros(filled_nodule_list[0]['mask'].shape, dtype=bool)
		for i in range(len(filled_nodule_list)):
			if filled_nodule_list[i]['nodule_id'] == nodule_id:
				umask = np.logical_or(umask, filled_nodule_list[i]['mask'])
		if nodule_freqs[nodule_id]>4:
			#eliminate small annotations to fit the number of rates under 4
			sizes = np.zeros(len(filled_nodule_list), dtype=int)
			for i in range(len(filled_nodule_list)):
				if filled_nodule_list[i]['nodule_id'] == nodule_id:
					sizes[i] = np.count_nonzero(filled_nodule_list[i]['mask'])
			tsize = np.count_nonzero(umask)
			for i in range(len(filled_nodule_list)-1, -1, -1):
				if sizes[i]>0 and sizes[i]<tsize*0.5:
					#little parts separated from a whole lobulation
					filled_nodule_list.pop(i)
					sizes = np.delete(sizes, i)
			while np.count_nonzero(sizes)>4:
				minsize = sizes[sizes>0].min()
				minind = np.where(sizes==minsize)[0]
				for ind in minind:
					sizes = np.delete(sizes, ind)
					filled_nodule_list.pop(ind)
		union = {}
		union['nodule_id'] = nodule_id
		union['mask'] = umask
		union['masks'] = []
		union['coords'] = []
		union['freq'] = []
		union['radiologist'] = 0
		#union['malignancy'] = 0
		union['characteristics'] = OrderedDict()
		for charkey in filled_nodule_list[0]['characteristics'].keys():
			union['characteristics'][charkey] = 0
		maxmal = 1
		minmal = 5
		for j in range(len(filled_nodule_list)):
			if filled_nodule_list[j]['nodule_id'] == union['nodule_id']:
				# they are the same nodule
				union['masks'].append(filled_nodule_list[j]['mask'])
				#union['malignancy'] += filled_nodule_list[j]['malignancy']
				for charkey in filled_nodule_list[j]['characteristics'].keys():
					union['characteristics'][charkey] += int(filled_nodule_list[j]['characteristics'][charkey])
				for coord in filled_nodule_list[j]['coords']:
					if coord not in union['coords']:
						union['coords'].append(coord)
						union['freq'].append(1)
					else:
						union['freq'][union['coords'].index(coord)] += 1
			malignancy = int(filled_nodule_list[j]['characteristics']['malignancy'])
			if maxmal<malignancy:
				maxmal = malignancy
			if minmal>malignancy:
				minmal = malignancy

		if maxmal-minmal>3:
			print("malignancy divergence: {}" .format(union["nodule_id"]))
		union['radiologist'] = max(union['freq'])
		for charkey in union['characteristics'].keys():
			union['characteristics'][charkey] /= float(nodule_freqs[nodule_id])
		union['freq'] = [i/float(nodule_freqs[nodule_id]) for i in union['freq']]

		union_nodule_list.append(union)

	return union_nodule_list
	
def nodule_centers(union_nodule_list, crop_shape=None):
	nodule_center_list = []
	for i in range(len(union_nodule_list)):
		center = {}
		center['nodule_id'] = union_nodule_list[i]['nodule_id']
		center['mask'] = mt.mask_outlier_eliminate(union_nodule_list[i]['mask'])
		#center['malignancy'] = union_nodule_list[i]['malignancy']
		center['characteristics'] = union_nodule_list[i]['characteristics']
		coords = np.array(union_nodule_list[i]['coords'], dtype=float)
		if 'freq' in union_nodule_list[i].keys():
			freqs = np.array(union_nodule_list[i]['freq'], dtype=float)
			center['coord'] = [int((coords[:,0]*freqs).sum()/freqs.sum()+0.5), int((coords[:,1]*freqs).sum()/freqs.sum()+0.5), int((coords[:,2]*freqs).sum()/freqs.sum()+0.5)]
		else:
			center['coord'] = [int(coords[:,0].sum()/len(coords)+0.5), int(coords[:,1].sum()/len(coords)+0.5), int(coords[:,2].sum()/len(coords)+0.5)]
		if crop_shape is not None:
			center['masks'] = []
			crop_shape_half = crop_shape / 2
			for mask in union_nodule_list[i]['masks']:
				mask_padded = np.pad(mask, ((crop_shape_half[0], crop_shape_half[0]), (crop_shape_half[1], crop_shape_half[1]), (crop_shape_half[2], crop_shape_half[2])), 'minimum')
				mask_cropped = mask_padded[center['coord'][0]:center['coord'][0]+crop_shape[0], center['coord'][1]:center['coord'][1]+crop_shape[1], center['coord'][2]:center['coord'][2]+crop_shape[2]]
				center['masks'].append(mask_cropped)
		nodule_center_list.append(center)
	return nodule_center_list

def plot_3d(image, lidc_id, nodule_id, pic_s_path, threshold=0.5):
	p = image.transpose(2,1,0)

	verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)
	fig = plt.figure(figsize=(10,10))
	ax = fig.add_subplot(111, projection='3d')

	mesh = Poly3DCollection(verts[faces], alpha=0.1)
	face_color = [0.5, 0.5, 1]
	mesh.set_facecolor(face_color)
	ax.add_collection3d(mesh)
	ax.set_xlim(0, p.shape[0])
	ax.set_ylim(0, p.shape[1])
	ax.set_zlim(0, p.shape[2])
	plt.savefig(pic_s_path+lidc_id+"_label_"+str(nodule_id)+".png")

def generate_cube(union_nodule_list, image_array, sample_s_path, label_s_path, pic_s_path, lidc_id):
	deps = image_array.shape[0]
	cols = image_array.shape[1]
	rows = image_array.shape[2]

	for nodule in union_nodule_list:
		nodule_id = nodule['nodule_id']
		roi = np.array(nodule['coords'])
		#print(roi.shape)
		score = np.array(nodule['freq'])
		#print(score.shape)

		col_max = roi.max(axis=0)    # zyx
		col_min = roi.min(axis=0)
		cube_d = col_max[0] - col_min[0] + 1
		cube_h = col_max[1] - col_min[1] + 1
		cube_w = col_max[2] - col_min[2] + 1
		label_array = np.zeros([cube_d, cube_h, cube_w])
		#print(label_array.shape)
		pic_array = np.zeros([cube_d, cube_h, cube_w])
		for i in range(roi.shape[0]):
			label_array[int(roi[i,0]-col_min[0]), int(roi[i,1]-col_min[1]), int(roi[i,2]-col_min[2])] = score[i]
			pic_array[int(roi[i,0]-col_min[0]), int(roi[i,1]-col_min[1]), int(roi[i,2]-col_min[2])] = 1
		nodule_array = image_array[col_min[0]:(col_max[0]+1), col_min[1]:(col_max[1]+1), col_min[2]:(col_max[2]+1)]
		assert nodule_array.shape == label_array.shape
		if nodule_array.shape[0] > 2 and min(nodule_array.shape[1], nodule_array.shape[2]) >= 10 and nodule['radiologist'] == 4:
			plot_3d(pic_array, lidc_id, nodule_id, pic_s_path)
			np.save(sample_s_path + lidc_id + "_sample_" + str(nodule_id) + ".npy", nodule_array)
			np.save(label_s_path + lidc_id + "_label_" + str(nodule_id) + ".npy", label_array)
			print("Saved %s's no.%d nodule" % (lidc_id, nodule_id))

def net_batch(filelist, start, end, volume_shape, scale_augment=False, translation_augment=False, rotation_augment=False, flip_augmen=False):
	if start<0:
		print("batch start corrected")
		start = 0
	
	out_batch = []
	out_label = []
	for f in range(start, end):
		file = filelist[f]
		malignancy = sample_malignancy_label(file)
		volume_overbound = np.load(file)
		data_aug = da.rotation_flip_augment(da.scale_translation_augment(volume_overbound, volume_shape), rotation_augment, flip_augment)

if __name__ == '__main__':
	'''
	scan_path = "DOI/LIDC-IDRI-0003/1.3.6.1.4.1.14519.5.2.1.6279.6001.101370605276577556143013894866/1.3.6.1.4.1.14519.5.2.1.6279.6001.170706757615202213033480003264"
	lidc_path = "DOI/"
	out_path = lidc_path + "seg_3d_samples/"
	data_path = lidc_path + "data/"
	label_path = out_path + "label/"
	sample_path = out_path + "sample/"
	pic_path = out_path + "picture/"
	'''
	data_path = "DOI"
	patient_list = os.listdir(data_path)
	for patient in patient_list:
		patient_path = data_path + '/' + patient
		if os.path.isdir(patient_path):
			scan_path = find_scan(patient_path)
			nodules, nonnodules = parseXML(scan_path)
			#print(nodules)
			transed_nodule_list, image_array, deps, cols, rows = coord_trans(nodules, scan_path)
			#print(transed_nodule_list)
			transed_nodule_list = duplicate_nodules(transed_nodule_list)
			#print(transed_nodule_list)
			filled_nodule_list = fill_hole(transed_nodule_list, deps, cols, rows)
			union_nodule_list = calc_union_freq(filled_nodule_list)
			print("%s %d nodules" %(scan_path, len(union_nodule_list)))
			nodule_center_list = nodule_centers(union_nodule_list)
			print(nodule_center_list)
