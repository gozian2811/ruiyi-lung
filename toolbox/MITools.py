import os
import sys
sys.path.append('/home/fyl/programs/lung_project/toolbox')
import shutil
import math
import random
import copy
import array
import scipy.ndimage
import pydicom as pdc
import pandas as pd
import numpy as np
import SimpleITK as sitk
from skimage import measure
from basicconfig import config

def sample_statistics(filelist, printing=False):
	labels = {}
	for sample_file in filelist:
		filename = os.path.basename(sample_file)
		label = filename.split('_')[2]
		if label not in labels.keys():
			labels[label] = 1
		else:
			labels[label] += 1
	print("sample_statistics:")
	if printing:
		for key in labels.keys():
			print(key + ':' + str(labels[key]))
	return labels

def get_mhd_uid(filepath):
	filename = os.path.basename(filepath)
	fileparts = os.path.splitext(filename)
	return fileparts[0]

def get_volume_informations(filepath):
	filename = os.path.basename(filepath)
	fileparts = os.path.splitext(filename)
	fileinfos = fileparts[0].split("_")
	return fileinfos

def get_luna_coordinates(patient_uid, coordfile):
	coordinfo = pd.read_csv(coordfile)
	coordlines = (coordinfo["seriesuid"].values==patient_uid).nonzero()[0]
	coordlist = []
	for coordline in coordlines:
		coordX = coordinfo["coordX"].values[coordline]
		coordY = coordinfo["coordY"].values[coordline]
		coordZ = coordinfo["coordZ"].values[coordline]
		coordlist.append([coordX, coordY, coordZ])
	return coordlist
	
def get_luna_annotations(patient_uid, annofile):
	annotations = pd.read_csv(annofile)
	annolines = (annotations["seriesuid"].values.astype(type(patient_uid))==patient_uid).nonzero()[0]
	annolist = []
	for annoline in annolines:
		coordX = annotations["coordX"].values[annoline]
		coordY = annotations["coordY"].values[annoline]
		coordZ = annotations["coordZ"].values[annoline]
		diameter_mm = annotations["diameter_mm"].values[annoline]
		annolist.append([coordX, coordY, coordZ, diameter_mm])
	return annolist

def get_challenge_annotations(patient_uid, annofile, label=None):
	annotations = pd.read_csv(annofile)
	filtercondition = annotations["seriesuid"].values.astype(type(patient_uid))==patient_uid
	if label is not None and 'label' in annotations.keys():
		filtercondition = np.logical_and(filtercondition, annotations["label"].values.astype(type(label))==label)
	annolines = filtercondition.nonzero()[0]
	annolist = []
	for annoline in annolines:
		coordX = annotations["coordX"].values[annoline]
		coordY = annotations["coordY"].values[annoline]
		coordZ = annotations["coordZ"].values[annoline]
		annotation = [coordX, coordY, coordZ]
		if "diameter_mm" in annotations.keys():
			annotation.append(annotations["diameter_mm"].values[annoline])
		elif "diameterX" in annotations.keys() and "diameterY" in annotations.keys() and "diameterZ" in annotations.keys():
			diameter = np.array([annotations["diameterX"].values[annoline], annotations["diameterY"].values[annoline], annotations["diameterZ"].values[annoline]])
			annotation.append(diameter)
		annolist.append(annotation)
	return annolist
	
'''
def get_sph_annotations(patient_uid, annofile):
	annolist = []
	annotations = pd.read_excel(annofile)
	silist = annotations.serie_id.tolist()
	if silist.count(patient_uid)==0:
		return annolist
	serie_index = silist.index(patient_uid)
	annotation_columns = ["anno1", "anno2", "anno3", "anno4", "anno5", "anno6", "anno7", "anno8", "anno9"]
	for annocol in annotation_columns:
		annostr = annotations.get(annocol)[serie_index]
		if type(annostr)==unicode:
			#annotation = np.array(annostr.split(u'\uff08')[0].split(' '), dtype=int)
			#patient_nodules.append([serie_index, annotation]) #the index order is [x,y,z]
			if annostr.find(u'*')>=0:
				continue
			coordbegin = -1
			coordend = -1
			for ci in range(len(annostr)):
				if coordbegin<0:
					if annostr[ci]>=u'0' and annostr[ci]<=u'9':
						coordbegin = ci
				elif (annostr[ci]<u'0' or annostr[ci]>u'9') and annostr[ci]!=u' ':
					coordend = ci
					break
			if coordbegin>=0:
				if coordend<0:
					coordend = len(annostr)
				coordstr = annostr[coordbegin:coordend]
				annotation = np.array(coordstr.split(u' '), dtype=int)
				annolist.append(annotation)  # the index order is [x,y,z]
		if type(annostr)==str:
			if annostr.find('*')>=0:
				continue
			coordbegin = -1
			coordend = -1
			for ci in range(len(annostr)):
				if coordbegin<0:
					if annostr[ci]>='0' and annostr[ci]<='9':
						coordbegin = ci
				elif (annostr[ci]<'0' or annostr[ci]>'9') and annostr[ci]!=' ':
					coordend = ci
					break
			if coordbegin>=0:
				if coordend<0:
					coordend = len(annostr)
				coordstr = annostr[coordbegin:coordend]
				annotation = np.array(coordstr.split(' '), dtype=int)
				annolist.append(annotation)  # the index order is [x,y,z]
	return annolist
'''

def get_luna_annotation_informations(filepath, annofile):
	fileinfos = get_volume_informations(filepath)
	patient_uid = fileinfos[0]
	annoline = int(fileinfos[1])

	annotations = pd.read_csv(annofile)
	anno_uid = annotations["seriesuid"].values[annoline]
	if patient_uid != anno_uid:
		return "", -1
	nodule_diameter = float(annotations["diameter_mm"].values[annoline])
	return anno_uid, nodule_diameter
	
def correct_luna_annotations(annofile, filelist, outputfile):
	corrannos = []
	for filename in filelist:
		full_image_info = sitk.ReadImage(patient)
		origin = np.array(full_image_info.GetOrigin())
		spacing = np.array(full_image_info.GetSpacing())
		uid = get_serie_uid(filepath)
		annolist = get_annotations(uid, annofile)
		for anno in annolist:
			for ci in range(3):
				if anno[ci]-origin[ci]<0:
					print("{} {} corrected" .format(uid, anno))
					anno[ci] = origin[ci] + abs(anno[ci]-origin[ci])
			corrannos.append([uid, anno[0], anno[1], anno[2], anno[3]])
	output_frame = pd.DataFrame(data=corrannos, columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm'])
	output_frame.to_csv(outputfile, index=False)

def get_dicom_spacing(case_path, mode='imageposition'):
	slices = os.listdir(case_path)
	for slice in slices:
		if slice[-3:]=='xml':
			slices.remove(slice)
			break
	if mode=='slicethickness':
		randslices = np.random.choice(slices, 2)
		info0 = pdc.read_file(case_path+'/'+randslices[0])
		info1 = pdc.read_file(case_path+'/'+randslices[0])
		thickness0 = info0.SliceThickness
		thickness1 = info1.SliceThickness
		pixelspacing0 = info0.PixelSpacing
		pixelspacing1 = info1.PixelSpacing
		spacing0 = [float(thickness0), float(pixelspacing0[1]), float(pixelspacing0[0])]
		spacing1 = [float(thickness1), float(pixelspacing1[1]), float(pixelspacing1[0])]
		if spacing0[0]!=spacing1[0] or spacing0[1]!=spacing1[1] or spacing0[2]!=spacing1[2]:
			return None
		else:
			return spacing0		#z, y, x
	elif mode=='imageposition':
		randsind = random.randint(0, len(slices)-1)
		rinfo = pdc.read_file(case_path+'/'+slices[randsind])
		rinstnum = int(rinfo.InstanceNumber)
		rpixelspacing = rinfo.PixelSpacing
		rimgpos = rinfo.ImagePositionPatient
		rslpos = float(rimgpos[2])
		for slice in slices:
			sinfo = pdc.read_file(case_path+'/'+slice)
			sinstnum = int(sinfo.InstanceNumber)
			if abs(rinstnum-sinstnum)==1:
				simgpos = sinfo.ImagePositionPatient
				spos = float(simgpos[2])
				slicethickness = abs(rslpos-spos)
				break
		if 'slicethickness' in dir():
			spacing = [slicethickness, float(rpixelspacing[1]), float(rpixelspacing[0])]
			return spacing
		else:
			return None
	else:
		return None

def read_dicom_scan(case_path):
	slices = os.listdir(case_path)
	zpositions = []
	infos = []
	for slice in slices:
		if slice[-3:]!='xml':
			info = pdc.read_file(case_path+'/'+slice)
			zpositions.append(float(info.ImagePositionPatient[2]))
			infos.append(info)
	if len(infos)==0:
		return None, None

	volume = np.zeros((len(infos), infos[0].pixel_array.shape[0], infos[0].pixel_array.shape[1]), dtype=int)
	zparray = np.float_(zpositions)
	indices = zparray.argsort()

	volinfo = {}
	volinfo['uid'] = infos[0].StudyInstanceUID
	oriposition = infos[indices[0]].ImagePositionPatient
	origin = [float(oriposition[2]), float(oriposition[1]), float(oriposition[0])]
	volinfo['origin'] = origin
	#info = infos[0]
	#pixelspacing = info.PixelSpacing
	#slicethickness = zpositions[indices[1]] - zpositions[indices[0]]
	#spacing = [slicethickness, float(pixelspacing[1]), float(pixelspacing[0])]
	#volinfo['spacing'] = spacing

	volinfo['zpositions'] = []
	spacingstat = ([], [])
	for i in range(len(indices)):
		info = infos[indices[i]]
		volume[i] = info.pixel_array * float(info.RescaleSlope) + int(info.RescaleIntercept)
		volinfo['zpositions'].append(zpositions[indices[i]])
		spacingstat[0].append(zpositions[indices[i]] - zpositions[indices[i-1]])
		spacingstat[1].append(info.PixelSpacing)
	ststat, stcounts = np.unique(spacingstat[0], return_counts=True)
	spstat, spcounts = np.unique(spacingstat[1], return_counts=True, axis=0)
	slicethickness = ststat[stcounts.argmax()]
	spacing = spstat[spcounts.argmax()].astype(float)
	volinfo['spacing'] = (slicethickness, spacing[1], spacing[0])

	return volume, volinfo

def sph_arrange(case_path, output_path='.'):
	for s in os.listdir(case_path):
		filename = case_path + '/' + s
		info = pdc.read_file(filename)
		patient_id = info.PatientID
		outpath = output_path + '/' + patient_id
		if not os.access(outpath, os.F_OK): os.makedirs(outpath)
		shutil.copyfile(filename, outpath+'/'+s)

def read_sph_id(case_path):
	files = os.listdir(case_path)
	randfile = case_path + '/' + files[random.randint(0, len(case_path))]
	randinfo = pdc.read_file(randfile)
	return randinfo.PatientID

def read_sph_scan(case_path):
	reader = sitk.ImageSeriesReader()
	dicom_infos = []
	serie_count = {}
	max_serie = -1
	max_serie_count = 0
	mess = False
	for s in os.listdir(case_path):
		filename = case_path + '/' + s
		info = pdc.read_file(filename)
		if 'patient_id' not in dir():
			patient_id = info.PatientID
		elif patient_id != info.PatientID:
			mess = True
		siuid = info.SeriesInstanceUID
		instancenumber = info.InstanceNumber
		dicom_infos.append([filename, siuid, instancenumber])
		if siuid in serie_count:
			serie_count[siuid] += 1
		else:
			serie_count[siuid] = 1
		if max_serie_count < serie_count[siuid]:
			max_serie_count = serie_count[siuid]
			max_serie = siuid
	if mess: print(case_path + ' slices mess.')
	dicom_series = []
	instance_numbers = []
	for filename, siuid, instancenumber in dicom_infos:
		if siuid == max_serie:
			if len(dicom_series)==0:
				dicom_series.append(filename)
				instance_numbers.append(instancenumber)
			else:
				for ds in range(len(dicom_series)):
					if instancenumber<instance_numbers[ds]:
						dicom_series.insert(ds, filename)
						instance_numbers.insert(ds, instancenumber)
						break
				if ds>=len(dicom_series)-1:
					dicom_series.append(filename)
					instance_numbers.append(instancenumber)

	reader.SetFileNames(dicom_series)
	full_image_info = reader.Execute()
	full_scan = sitk.GetArrayFromImage(full_image_info)
	return full_scan, full_image_info, info.PatientID

def write_mhd_file(mhdfile, data, dsize):
	def write_meta_header(filename, meta_dict):
		header = ''
		# do not use tags = meta_dict.keys() because the order of tags matters
		tags = ['ObjectType', 'NDims', 'BinaryData',
			'BinaryDataByteOrderMSB', 'CompressedData', 'CompressedDataSize',
			'TransformMatrix', 'Offset', 'CenterOfRotation',
			'AnatomicalOrientation',
			'ElementSpacing',
			'DimSize',
			'ElementType',
			'ElementDataFile',
			'Comment', 'SeriesDescription', 'AcquisitionDate', 'AcquisitionTime', 'StudyDate', 'StudyTime']
		for tag in tags:
			if tag in meta_dict.keys():
				header += '%s = %s\n' % (tag, meta_dict[tag])
		f = open(filename, 'w')
		f.write(header)
		f.close()

	def dump_raw_data(filename, data):
		""" Write the data into a raw format file. Big endian is always used. """
		# Begin 3D fix
		data = data.reshape([data.shape[0], data.shape[1] * data.shape[2]])
		# End 3D fix
		rawfile = open(filename, 'wb')
		a = array.array('f')
		for o in data:
			a.fromlist(list(o))
		# if is_little_endian():
		#    a.byteswap()
		a.tofile(rawfile)
		rawfile.close()
	assert (mhdfile[-4:] == '.mhd')
	meta_dict = {}
	meta_dict['ObjectType'] = 'Image'
	meta_dict['BinaryData'] = 'True'
	meta_dict['BinaryDataByteOrderMSB'] = 'False'
	meta_dict['ElementType'] = 'MET_FLOAT'
	meta_dict['NDims'] = str(len(dsize))
	meta_dict['DimSize'] = ' '.join([str(i) for i in dsize])
	meta_dict['ElementDataFile'] = os.path.split(mhdfile)[1].replace('.mhd', '.raw')
	write_meta_header(mhdfile, meta_dict)
	pwd = os.path.split(mhdfile)[0]
	if pwd:
		data_file = pwd + '/' + meta_dict['ElementDataFile']
	else:
		data_file = meta_dict['ElementDataFile']
	dump_raw_data(data_file, data)

def medical_normalization(x, max_bound = None, min_bound = None, pixel_mean = None, crop = None, input_copy = True):
	if max_bound is None:
		max_bound = config['MAX_BOUND']
	if min_bound is None:
		min_bound = config['MIN_BOUND']
	if pixel_mean is None:
		pixel_mean = config['PIXEL_MEAN']
	if crop is None:
		crop = config['NORM_CROP']
	if input_copy:
		x = copy.copy(x)
	x = (x - min_bound) / float(max_bound - min_bound)
	if crop:
		x[x>1] = 1
		x[x<0] = 0
	x = x - pixel_mean
	return x
	
def coord_conversion(coord, origin, old_spacing, old_shape, new_shape, dir_array=True):
	#noticing that the first scan and the last scan between volumes before and after resample are morphologically equal
	if dir_array:
		ccoord = (coord - origin) / old_spacing * (np.array(new_shape) - 1) / (np.array(old_shape) - 1)
	else:
		ccoord = coord * (np.array(old_shape) - 1) / (np.array(new_shape) - 1) * old_spacing + origin
	return ccoord

def coord_overflow(coord, shape, topopen=False):
	for i in range(len(coord)):
		bottom_overflow = coord[i]<0
		if topopen:
			top_overflow = coord[i]>shape[i]
		else:
			top_overflow = coord[i]>=shape[i]
		if bottom_overflow or top_overflow:
			return True
		#if coord[i]<0 or coord[i]>=shape[i]:
		#	return True
	return False

def resample(image, old_spacing, new_spacing=np.array([1, 1, 1])):
	resize_factor = old_spacing / new_spacing.astype(float)
	new_real_shape = image.shape * resize_factor
	new_shape = np.round(new_real_shape)
	real_resize_factor = new_shape / image.shape
	new_spacing = old_spacing / real_resize_factor
	if image.shape[0]<10000:
        	image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
	else:
		num_batch = int(math.ceil(image.shape[0]/1000.0))
		for b in range(num_batch):
			image_batch = image[b*1000:min((b+1)*1000,image.shape[0]), :, :]
			image_batch = scipy.ndimage.interpolation.zoom(image_batch, real_resize_factor, mode='nearest')
			if 'new_image' in dir():
				new_image = np.append(new_image, image_batch, axis=0)
			else:
				new_image = image_batch
		image = new_image

	return image, new_spacing

def crop(volume, crop_shape):
	if volume.shape[0]<crop_shape[0] or volume.shape[1]<crop_shape[1] or volume.shape[2]<crop_shape[2]:
		return np.zeros(crop_shape)
	else:
		volshape = np.float_(volume.shape)
		cropshape = np.float_(crop_shape)
		volcenter = volshape / 2.0
		crophalf = cropshape / 2.0
		cdbt = np.int_(volcenter-crophalf+0.5)
		cdtp = np.int_(volcenter+crophalf+0.5)
		volume_cropped = volume[cdbt[0]:cdtp[0], cdbt[1]:cdtp[1], cdbt[2]:cdtp[2]]
		return volume_cropped

def local_crop(full_scan, nodule_coordinate, box_shape, padding=config['MIN_BOUND']):
	#if not isinstance(box_shape, np.ndarray):
	if isinstance(box_shape, int):
		box_shape = np.repeat(box_shape, full_scan.ndim)
	box_prehalf = np.int_(np.array(box_shape)/2)
	box_afthalf = np.array(box_shape) - box_prehalf
	v_center = np.array(nodule_coordinate)
	zyx_1 = v_center - box_prehalf
	zyx_2 = v_center + box_afthalf
	if zyx_1.min()<0 or (np.array(full_scan.shape)-zyx_2).min()<0:
		full_scan = np.pad(full_scan, ((box_prehalf[0], box_afthalf[0]), (box_prehalf[1], box_afthalf[1]), (box_prehalf[2], box_afthalf[2])), 'constant', constant_values = ((padding, padding), (padding, padding), (padding, padding)))
		zyx_1 = v_center
		zyx_2 = v_center + np.array(box_shape)
	vol_crop = full_scan[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]  # ---截取立方体
	return vol_crop

def mask_box(mask, square=False, overbound=True):
	mask_coords = mask.nonzero()
	bbox = np.array(((mask_coords[0].min(), mask_coords[0].max()+1), (mask_coords[1].min(), mask_coords[1].max()+1), (mask_coords[2].min(), mask_coords[2].max()+1)), dtype=int)
	center = np.int_((bbox[:,1] + bbox[:,0]) / 2.0)
	shape = bbox[:,1] - bbox[:,0]
	if square:
		shape[1:] = shape[1:].max()
	if overbound:
		shape[1:] *= 2
	return center, shape
	
def mask_crop(nodule_scan, mask, square=False, overbound=True):
	center, shape = mask_box(mask, square, overbound)
	return local_crop(nodule_scan, center, shape), local_crop(mask, center, shape, padding=0)

def mask_outlier_eliminate(mask, outlier_size=1):
	mask_connection = measure.label(mask, connectivity=2, background=0)
	num_regions = mask_connection.max()
	if num_regions>1:
		for i in range(num_regions):
			region_label = i + 1
			if np.count_nonzero(mask_connection==region_label)<=outlier_size:
				mask[mask_connection==region_label] = 0
	return mask

def region_valid(voxels, threshold_ratio=0.99):
	num_voxel = voxels.size
	num_tissue = voxels[voxels>-600].size
	if num_tissue < num_voxel*threshold_ratio:
		return True
	else:
		#too much tissue in this region, there's little possibility for nodules to exist
		return False

def make_patchs(voxels):
	width, length, height = voxels.shape
	patch_size = np.min(voxels.shape)
	patchs = np.zeros(shape=(9,patch_size,patch_size), dtype = float)
	patchs[0] = voxels[:,:,int(height/2)]
	patchs[1] = voxels[:,int(length/2),:]
	patchs[2] = voxels[int(width/2),:,:]
	for h in range(height):
		patchs[3,:,h] = voxels[:,h,h]
		patchs[4,h,:] = voxels[h,:,h]
		patchs[5,:,h] = voxels[:,h,height-h-1]
		patchs[6,h,:] = voxels[h,:,height-h-1]
	for w in range(width):
		patchs[7,w,:] = voxels[w,w,:]
		patchs[8,w,:] = voxels[width-w-1,w,:]
	
	return patchs

def concatenate_patchs(patchs, num_channels=3):
	img_height = patchs.shape[1]
	img_width = patchs.shape[2]
	half_width = int(img_width/2)
	half_height = int(img_height/2)

	#3 orders of patch indices
	patch_indices_list = []
	patch_indices = [ind for ind in range(9)]
	patch_indices_list.append(patch_indices)
	patch_indices_list.append(copy.copy(patch_indices))
	patch_indices_list.append(copy.copy(patch_indices))
	#random.shuffle(patch_indices_list[0])
	random.shuffle(patch_indices_list[1])
	patch_indices_list[2].reverse()

	patch_chan = np.zeros(shape=(4*img_height,4*img_width,num_channels), dtype=float)
	for c in range(num_channels):
		patch_indices = patch_indices_list[c]
		aug_patch = np.ndarray(shape=(9*img_height, 9*img_width), dtype=float)
		for h in range(3):
			for w in range(3):
				for hi in range(3):
					for wi in range(3):
						aug_patch[(h*3+hi)*img_height:(h*3+hi+1)*img_height, (w*3+wi)*img_width:(w*3+wi+1)*img_width] = patchs[patch_indices[hi*3+wi]]
		patch_chan[:,:,c] = aug_patch[2*img_height+half_height:7*img_height-half_height, 2*img_width+half_width:7*img_width-half_width]

	return patch_chan

def box_sample(full_scan, nodule_coordinate, box_size, resize_factor=None):
	if not isinstance(box_size, np.ndarray):
		box_size = np.repeat(box_size, full_scan.ndim)
	box_prehalf = np.int_(box_size/2)
	box_posthalf = box_size - box_prehalf
	if resize_factor is not None:
		box_size = np.int_(np.ceil(box_size / resize_factor))
	'''
	v_center = np.array(nodule_coordinate)
	zyx_1 = v_center - box_size
	zyx_2 = v_center + box_size
	if zyx_1.min()<0 or (np.array(full_scan.shape)-zyx_2).min()<0:
		full_scan = np.pad(full_scan, ((box_size[0], box_size[0]), (box_size[1], box_size[1]), (box_size[2], box_size[2])), 'constant', constant_values = ((config['MIN_BOUND'], config['MIN_BOUND']), (config['MIN_BOUND'], config['MIN_BOUND']), (config['MIN_BOUND'], config['MIN_BOUND'])))
		zyx_1 = v_center
		zyx_2 = v_center + 2 * box_size
	vol_crop = full_scan[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]  # ---截取立方体
	'''
	vol_crop = local_crop(full_scan, nodule_coordinate, box_size)
	if resize_factor is not None:
        	vol_crop = scipy.ndimage.interpolation.zoom(vol_crop, resize_factor, mode='nearest')
	bottom_coord = np.int_(np.array(vol_crop.shape)/2-box_prehalf)
	top_coord = np.int_(np.array(vol_crop.shape)/2+box_posthalf)
	nodule_box = vol_crop[bottom_coord[0]:top_coord[0], bottom_coord[1]:top_coord[1], bottom_coord[2]:top_coord[2]]
	#nodule_box = vol_crop[int(vol_crop.shape[0]/2-box_prehalf):int(vol_crop.shape[0]/2+box_size-box_prehalf), int(vol_crop.shape[1]/2-box_prehalf):int(vol_crop.shape[1]/2+box_size-box_prehalf), int(vol_crop.shape[2]/2-box_prehalf):int(vol_crop.shape[2]/2+box_size-box_prehalf)]  # ---将截取的立方体置于nodule_box
	return nodule_box

def spherical_sample(full_scan, nodule_coordinate, spacing, splitnum, spacesize_mm=50, imgsize=224):
	samples = []
	box_size = np.repeat(spacesize_mm*2, 3)
	box_shape = np.ceil(box_size / np.array(spacing)).astype(int)
	volume = local_crop(full_scan, nodule_coordinate, box_shape)
	new_spacing = np.repeat(np.min(spacing), 3)
	volume, _ = resample(volume, spacing, new_spacing)
	resize = np.min(spacing)*imgsize / float(spacesize_mm)
	imgsh = int(imgsize/2)
	for t in range(splitnum):
		for p in range(splitnum):
			volumerot = volume
			theta = 180 / float(splitnum) * t
			phi = 180 / float(splitnum) * p
			if phi!=0:
				volumerot = scipy.ndimage.interpolation.rotate(volumerot, phi, (1, 2))
			if theta!=0:
				volumerot = scipy.ndimage.interpolation.rotate(volumerot, theta, (0, 1))
			image = volumerot[int(volumerot.shape[0]/2.0+0.5),:,:]
			if resize is not None:
        			image = scipy.ndimage.interpolation.zoom(image, resize, mode='nearest')
			imgc = np.int_(np.array(image.shape) / 2 + 0.5)
			image = image[imgc[0]-imgsh:imgc[0]+imgsize-imgsh, imgc[1]-imgsh:imgc[1]+imgsize-imgsh]
			samples.append(image)
	return samples

def volume2image(volume, box_size):
	box_half = int(box_size/2)
	rangesize = int(math.sqrt(box_size))
	slicenum = rangesize * rangesize
	halfslicenum = int(slicenum/2)
	image = np.empty((rangesize*box_size, rangesize*box_size))
	ccenter = np.int_(np.array(volume.shape)/2+0.5)
	vision_crop = volume[ccenter[0]-halfslicenum:ccenter[0]+slicenum-halfslicenum, ccenter[1]-box_half:ccenter[1]+box_size-box_half, ccenter[2]-box_half:ccenter[2]+box_size-box_half]
	for py in range(rangesize):
		for px in range(rangesize):
			image[py*box_size:(py+1)*box_size, px*box_size:(px+1)*box_size] = vision_crop[py*rangesize+px]

	return image	

def nodule_cluster(nodule_centers, scale, iterate=False):
	print("Clustering:")
	clusters=[]
	l=len(nodule_centers)
	if l==0:
		return clusters
	center_index_cluster = 0 - np.ones(len(nodule_centers), dtype=int)
	#initial clustering
	point = nodule_centers[l-1]	#point is a list
	center_index_cluster[l-1] = 0
	clusters.append([point, point, 1])
	for i in range(l-1):
		point = nodule_centers[i] #The current point to be clustered
		flag = 0
		nearsqdist = scale * scale
		nearcand = -1
		#find the older cluster
		for j in range(len(clusters)):
			#calculate the distance with only coordination but prediction
			sqdist = (point[0]-clusters[j][0][0])*(point[0]-clusters[j][0][0]) + (point[1]-clusters[j][0][1])*(point[1]-clusters[j][0][1]) + (point[2]-clusters[j][0][2])*(point[2]-clusters[j][0][2])
			if sqdist<scale*scale and sqdist<nearsqdist: #which means we should add the point into this cluster
				#Notice the type that cluster is a list so we need to write a demo
				nearsqdist = sqdist
				nearcand = j
				flag=1
		if flag==1:
			clusters[nearcand][1] = [(clusters[nearcand][1][0]+point[0]),
                				(clusters[nearcand][1][1]+point[1]),
                				(clusters[nearcand][1][2]+point[2]),
						(clusters[nearcand][1][3]+point[3])]
			clusters[nearcand][2] = clusters[nearcand][2]+1
			clusters[nearcand][0] = [(clusters[nearcand][1][0])/clusters[nearcand][2],
                				(clusters[nearcand][1][1])/clusters[nearcand][2],
                				(clusters[nearcand][1][2])/clusters[nearcand][2],
						(clusters[nearcand][1][3])/clusters[nearcand][2]]
			center_index_cluster[i] = nearcand
		else:
			# create a new cluster
			center_index_cluster[i] = len(clusters)
			clusters.append([point, point, 1])
	
	if iterate:
		#rearrange the clusters by iterations
		converge = False
		while not converge:
			converge = True
			for i in range(l):
				point = nodule_centers[i] #The current point to be clustered
				nearsqdist = scale*scale
				nearcand = -1
				#find the older cluster
				for j in range(len(clusters)):
					if clusters[j][2]<=0:
		    				continue
					#calculate the distance with only coordination but prediction
					sqdist = (point[0]-clusters[j][0][0])*(point[0]-clusters[j][0][0]) + (point[1]-clusters[j][0][1])*(point[1]-clusters[j][0][1]) + (point[2]-clusters[j][0][2])*(point[2]-clusters[j][0][2])
					if sqdist<nearsqdist: #which means we should add the point into this cluster
						#Notice the type that cluster is a list so we need to write a demo
						nearsqdist = sqdist
						nearcand = j
				if nearcand>=0 and nearcand!=center_index_cluster[i]:
					converge = False
					oldcand = center_index_cluster[i]
					if oldcand>=0:
						clusters[oldcand][1] = [(clusters[oldcand][1][0] - point[0]),
												 (clusters[oldcand][1][1] - point[1]),
												 (clusters[oldcand][1][2] - point[2]),
												 (clusters[oldcand][1][3] - point[3])]
						clusters[oldcand][2] = clusters[oldcand][2] - 1
						clusters[oldcand][0] = [(clusters[oldcand][1][0]) / clusters[oldcand][2],
												(clusters[oldcand][1][1]) / clusters[oldcand][2],
												(clusters[oldcand][1][2]) / clusters[oldcand][2],
												(clusters[oldcand][1][3]) / clusters[oldcand][2]]
					clusters[nearcand][1] = [(clusters[nearcand][1][0]+point[0]),
								(clusters[nearcand][1][1]+point[1]),
								(clusters[nearcand][1][2]+point[2]),
								(clusters[nearcand][1][3]+point[3])]
					clusters[nearcand][2] = clusters[nearcand][2]+1
					clusters[nearcand][0] = [(clusters[nearcand][1][0]) / clusters[nearcand][2],
											 (clusters[nearcand][1][1]) / clusters[nearcand][2],
											 (clusters[nearcand][1][2]) / clusters[nearcand][2],
											 (clusters[nearcand][1][3]) / clusters[nearcand][2]]
					center_index_cluster[i] = nearcand
	solid_clusters = [c for c in clusters if c[2]>0]
	print('Clustering Done')

	return solid_clusters
