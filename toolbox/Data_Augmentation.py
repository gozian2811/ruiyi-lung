import scipy.ndimage as sni
import numpy as np
import random
import math
import copy
from . import BasicTools as bt
from . import MITools as mt

rotation_augment_number = 9
flip_augment_number = 3

def scale_translation_augment(volume_overbound, volume_shape, scales=[1.0], transrange=None, transnum=5):
	v_center = np.int_(np.array(volume_overbound.shape)/2)
	if transrange is None:
		translations = np.array([[0,0,0]])
	else:
		translations = np.zeros((transnum+1, 3))
		translations[1:,:] = np.random.uniform(transrange[0], transrange[1], (transnum,3))
	out_batch = []
	for s in range(len(scales)):
		scale = scales[s]
		for t in range(len(translations)):
			translation = translations[t]
			box_size = np.int_(np.ceil(volume_shape*scale))
			window_size = np.array(box_size/2, dtype=int)
			zyx_1 = np.int_(np.rint(v_center + translation - window_size))  #the order of indices is [Z, Y, X]
			zyx_2 = np.int_(np.rint(v_center + translation + box_size - window_size))
			if mt.coord_overflow(zyx_1, volume_overbound.shape) or mt.coord_overflow(zyx_2, volume_overbound.shape, topopen=True):
				#print('diameter:{} scale:{} translation:{} the region is out of the bound of the volume' .format(nodule_diameter, scale, translation))
				continue
			nodule_box = np.zeros(shape=volume_shape, dtype=int)  # ---nodule_box_size = 45
			img_crop = volume_overbound[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]
			img_crop[img_crop<-1024] = -1024  # the voxel value below -1024 is set to -1024
			if scale==1.0:
				img_crop_rescaled = img_crop
			else:
				img_crop_rescaled, rescaled_spacing = mt.resample(img_crop, np.array([1,1,1]), np.array([scale,scale,scale]))
			padding_shape = np.array((img_crop_rescaled.shape-volume_shape)/2, dtype=int)
			nodule_box = img_crop_rescaled[padding_shape[0]:padding_shape[0]+volume_shape[0], padding_shape[1]:padding_shape[1]+volume_shape[1], padding_shape[2]:padding_shape[2]+volume_shape[2]]
			out_batch.append(nodule_box)
	return out_batch
	
def rotation_flip_augment(data, rotation_augment=True, flip_augment=True):
	out_batch = []
	if type(data)==list:
		for volume in data:
			out_batch.extend(rotation_flip_augment(volume, rotation_augment, flip_augment))
	elif type(data)==np.ndarray:
		if len(data.shape)==3:
			out_batch.append(data)
			if rotation_augment:
				out_batch.append(np.rot90(data, k=1, axes=(2, 1)))
				out_batch.append(np.rot90(data, k=2, axes=(2, 1)))
				out_batch.append(np.rot90(data, k=3, axes=(2, 1)))
				out_batch.append(np.rot90(data, k=1, axes=(2, 0)))
				out_batch.append(np.rot90(data, k=2, axes=(2, 0)))
				out_batch.append(np.rot90(data, k=3, axes=(2, 0)))
				out_batch.append(np.rot90(data, k=1, axes=(1, 0)))
				out_batch.append(np.rot90(data, k=2, axes=(1, 0)))
				out_batch.append(np.rot90(data, k=3, axes=(1, 0)))
			if flip_augment:
				out_batch.append(data[::-1,:,:])
				out_batch.append(data[:,::-1,:])
				out_batch.append(data[:,:,::-1])
	else:
		return data
	return out_batch
	
def volume_extract_augment(index, volume_overbound, volume_shape, keep_origin=True, translation_num=0, translation_range=(-6,6), rotation_augment=False, flip_augment=False):
	if rotation_augment:
		rotation_num = rotation_augment_number
	else:
		rotation_num = 0
	if flip_augment:
		flip_num = flip_augment_number
	else:
		flip_num = 0
	transind = int(index / (rotation_num + flip_num + 1))
	rotflipind = index % (rotation_num + flip_num + 1)
	if keep_origin and transind==0:
		translation = np.array([0,0,0])
	else:
		translation = np.random.uniform(translation_range[0], translation_range[1], 3)
	
	v_center = np.int_(np.array(volume_overbound.shape)/2)
	box_size = np.int_(volume_shape)
	window_size = np.array(box_size/2, dtype=int)
	zyx_1 = np.rint(v_center + translation - window_size).astype(int)  #the order of indices is [Z, Y, X]
	zyx_2 = np.rint(v_center + translation + box_size - window_size).astype(int)
	if mt.coord_overflow(zyx_1, volume_overbound.shape) or mt.coord_overflow(zyx_2, volume_overbound.shape, topopen=True):
		#print('diameter:{} scale:{} translation:{} the region is out of the bound of the volume' .format(nodule_diameter, scale, translation))
		return None
	nodule_box = np.zeros(shape=volume_shape, dtype=int)  # ---nodule_box_size = 45
	volume_crop = volume_overbound[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]
	
	if rotflipind==0:
		volume_augmented = volume_crop
	elif rotation_augment and rotflipind<10:
		rotind = rotflipind - 1
		rotaxeses = ((1, 0), (2, 0), (2, 1))
		rottime = int(rotind / 3) + 1
		axesind = rotind % 3
		volume_augmented = np.rot90(volume_crop, k=rottime, axes=rotaxeses[axesind])
	else:
		if rotflipind>=10:
			flipind = rotflipind - 10
		else:
			flipind = rotflipind - 1
		volume_augmented = np.flip(volume_crop, flipind)
	return volume_augmented
	
def volume_extract_augment_random(index, volume_overbound, volume_shape, translation_num=0, translation_range=(-6,6), rotation_num=0, flip_num=0):
	#transind = int(index / (max(rotation_num,1)*max(flip_num,1)))
	#rotflipind = index % (max(rotation_num,1)*max(flip_num,1))
	#rotind = rotflipind / max(flip_num,1)
	#flipind = rotflipind % max(flip_num,1)
	if translation_num==0:
		translation = np.array([0,0,0])
	else:
		translation = np.random.uniform(translation_range[0], translation_range[1], 3)
	
	v_center = np.int_(np.array(volume_overbound.shape)/2)
	box_size = np.zeros(3, dtype=int)
	box_size[:] = volume_shape
	window_size = np.array(box_size/2, dtype=int)
	zyx_1 = np.rint(v_center + translation - window_size).astype(int)  #the order of indices is [Z, Y, X]
	zyx_2 = np.rint(v_center + translation + box_size - window_size).astype(int)
	if mt.coord_overflow(zyx_1, volume_overbound.shape) or mt.coord_overflow(zyx_2, volume_overbound.shape, topopen=True):
		#print('diameter:{} scale:{} translation:{} the region is out of the bound of the volume' .format(nodule_diameter, scale, translation))
		return None
	#nodule_box = np.zeros(shape=volume_shape, dtype=int)  # ---nodule_box_size = 45
	volume_crop = volume_overbound[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]
	
	volume_augmented = volume_crop
	if rotation_num>0:
		randrotind = random.randint(0, rotation_augment_number)
		if randrotind>0:
			rotaxeses = ((1, 0), (2, 0), (2, 1))
			rottime = int((randrotind-1) / 3) + 1
			axesind = (randrotind-1) % 3
			volume_augmented = np.rot90(volume_augmented, k=rottime, axes=rotaxeses[axesind])
	if flip_num>0:
		randflipind = random.randint(0, flip_augment_number)
		if randflipind>0:
			volume_augmented = np.flip(volume_augmented, randflipind-1)
	return volume_augmented
	
def extract_augment_random(index, image_overbound, image_size, translation_num=0, translation_range=(-6,6), rotation_num=0, flip_num=0, shear_num=0):
	#the order of index decomposition is (translation, rotation, flipping and shearing)
	if isinstance(image_overbound, np.ndarray):
		image_dimension = image_overbound.ndim
		image_shape = image_overbound.shape
	else:
		image_dimension = image_overbound[0].ndim
		image_shape = image_overbound[1].shape
	if translation_num==0:
		translation = np.zeros(image_dimension)
	else:
		translation = np.random.uniform(translation_range[0], translation_range[1], image_dimension)
		index = int(index/translation_num)
	rottime = None
	rotaxes = None
	if rotation_num>0:
		rotation_augment_number = 3**(image_dimension-1)
		if rotation_num!=10:
			rotind = random.randint(0, rotation_augment_number)
		else:
			rotind = index % rotation_num
			index = int(index/rotation_num)
		if rotind>0:
			rotaxeses = bt.array_couples_combine(np.arange(image_dimension))
			rottime = int((rotind-1) / len(rotaxeses)) + 1
			axesind = (rotind-1) % len(rotaxeses)
			rotaxes = rotaxeses[axesind]
			#image_augmented = np.rot90(image_augmented, k=rottime, axes=rotaxeses[axesind])
	flipdim = None
	if flip_num>0:
		flip_augment_number = image_dimension
		if flip_num!=4:
			flipind = random.randint(0, flip_augment_number)
		else:
			flipind = index % flip_num
			index = int(index/flip_num)
		if flipind>0:
			flipdim = flipind - 1
			#image_augmented = np.flip(image_augmented, flipind-1)
	shear_matrix = None
	if shear_num>0:
		shear_axis = random.randint(0, image_dimension-1)
		shear_angles = (np.random.rand(image_dimension) - 0.5) * 2 * math.pi / 3	#restrict the range of shear angles to [-2*pi/3, 2*pi/3]
		identity_matrix = np.identity(image_dimension+1)
		shear_matrix = copy.copy(identity_matrix)
		for d in range(image_dimension):
			if d != shear_axis:
				shear_matrix[shear_axis, d] = -math.sin(shear_angles[d])
				shear_matrix[d, d] = math.cos(shear_angles[d])
		#affine_matrix = np.random.rand(image_dimension, image_dimension) - 0.5
		#affine_matrix /= np.linalg.norm(affine_matrix, 2, axis=0)
		#shear_matrix[:-1, :-1] = affine_matrix
		offset_matrix = np.zeros((image_dimension+1, image_dimension+1))
		offset_matrix[:-1, -1] = np.array(image_shape) / 2
		shear_matrix = np.dot(np.dot(identity_matrix+offset_matrix, shear_matrix), identity_matrix-offset_matrix)
	if isinstance(image_overbound, np.ndarray):
		return make_augment(image_overbound, image_size, translation, rotaxes, rottime, flipdim, shear_matrix)
	else:
		image_augmented = []
		for image in image_overbound:
			image_augmented.append(make_augment(image, image_size, translation, rotaxes, rottime, flipdim, shear_matrix))
		return image_augmented
	
def make_augment(image_overbound, image_size, translation=0, rotation_axes=None, rotation_time=None, flip_dim=None, shear_matrix=None):
	v_center = np.array(image_overbound.shape)/2.0
	trans_center = np.rint(v_center + translation).astype(int)
	crop_size = np.int_(image_size)
	window_size = np.array(crop_size/2, dtype=int)
	cbottom = trans_center - window_size
	ctop = trans_center + crop_size - window_size
	if mt.coord_overflow(cbottom, image_overbound.shape) or mt.coord_overflow(ctop, image_overbound.shape, topopen=True):
		return None
	image_augmented = image_overbound
	if shear_matrix is not None:
		affine_matrix = shear_matrix[:-1, :-1]
		offset = shear_matrix[:-1, -1]
		image_augmented = sni.interpolation.affine_transform(image_augmented, affine_matrix, offset)
	for d in range(image_overbound.ndim):
		condition = np.zeros(image_overbound.shape[d], dtype=bool)
		condition[cbottom[d]:ctop[d]] = True
		image_augmented = image_augmented.compress(condition, axis=d)
	#image_augmented = image_overbound[cbottom[0]:ctop[0], cbottom[1]:ctop[1], cbottom[2]:ctop[2]]
	if rotation_axes is not None and rotation_time is not None:
		image_augmented = np.rot90(image_augmented, k=rotation_time, axes=rotation_axes)
	if flip_dim is not None:
		image_augmented = np.flip(image_augmented, flip_dim)
	return image_augmented
		
def extract_volumes(volume_overbound, volume_shape, nodule_diameter=-1, centering=True, scale_augment=False, translation_augment=False, rotation_augment=False, flip_augment=False):
	#volume_shape = np.int_(np.array(volume_overbound.shape)/2)
	v_center = np.int_(np.array(volume_overbound.shape)/2)
	if scale_augment:
		#the scale indicates the real size of the cropped box
		if nodule_diameter>44:
			scales = [1.0, 1.25]
		elif nodule_diameter<10 and nodule_diameter>0:
			scales = [0.8, 1.0]
		else:
			scales = [0.8,1.0,1.25]
	else:
		scales = [1.0]
	if translation_augment and nodule_diameter>=0:
		translations = np.array([[0,0,0],[0,0,1],[0,0,-1], [0,1,0],[0,math.sqrt(0.5),math.sqrt(0.5)],[0,math.sqrt(0.5),-math.sqrt(0.5)], [0,-1,0],[0,-math.sqrt(0.5),math.sqrt(0.5)],[0,-math.sqrt(0.5),-math.sqrt(0.5)],
				         [1,0,0],[math.sqrt(0.5),0,math.sqrt(0.5)],[math.sqrt(0.5),0,-math.sqrt(0.5)], [math.sqrt(0.5),math.sqrt(0.5),0],[math.sqrt(0.3333),math.sqrt(0.3333),math.sqrt(0.3333)],[math.sqrt(0.3333),math.sqrt(0.3333),-math.sqrt(0.3333)], [math.sqrt(0.5),-math.sqrt(0.5),0],[math.sqrt(0.3333),-math.sqrt(0.3333),math.sqrt(0.3333)],[math.sqrt(0.3333),-math.sqrt(0.3333),-math.sqrt(0.3333)],
				         [-1,0,0],[-math.sqrt(0.5),0,math.sqrt(0.5)],[-math.sqrt(0.5),0,-math.sqrt(0.5)], [-math.sqrt(0.5),math.sqrt(0.5),0],[-math.sqrt(0.3333),math.sqrt(0.3333),math.sqrt(0.3333)],[-math.sqrt(0.3333),math.sqrt(0.3333),-math.sqrt(0.3333)], [-math.sqrt(0.5),-math.sqrt(0.5),0],[-math.sqrt(0.3333),-math.sqrt(0.3333),math.sqrt(0.3333)],[-math.sqrt(0.3333),-math.sqrt(0.3333),-math.sqrt(0.3333)]])
		#translations = np.array([[0,0,0],[0,0,1],[0,0,-1],[0,1,0],[0,-1,0],[1,0,0],[-1,0,0]])
		#translations = np.random.uniform(-6, 6, (5,3))
		num_translations = 27
		rt = np.zeros(num_translations, dtype=int)
		rt[1:num_translations] = np.random.choice(range(1,len(translations)), num_translations-1, False)
		rt = np.sort(rt)
	else:
		translations = np.array([[0,0,0]])
		rt = np.array([0])

	#num_translations = min(5, len(translations))
	for s in range(len(scales)):
		#rt = np.zeros(num_translations, dtype=int)
		#rt[1:num_translations] = np.random.choice(range(1,len(translations)), num_translations-1, False)
		#rt = np.sort(rt)
		for t in range(rt.size):
			scale = scales[s]
			box_size = np.int_(np.ceil(volume_shape*scale))
			window_size = np.array(box_size/2, dtype=int)
			if nodule_diameter>=0 and t!=0:
				if nodule_diameter>0:
					if centering:
						transscales = np.array([nodule_diameter * 0.3])
					else:
						tsnum = math.sqrt(box_size.max()/nodule_diameter)
						step = box_size.max() / 2 / tsnum
						transscales = np.arange(1.0, tsnum) * step
				else:
					if centering:
						transscales = np.array([1])
					else:
						tsnum = 2
						step = box_size.max() / 2 / tsnum
						transscales = np.arange(1.0, tsnum) * step
			else:
				transscales = np.array([0])
			for ts in range(len(transscales)):
				transscale = transscales[ts]
				translation = np.array(transscale*translations[rt[t]], dtype=int)	#the translation step cooperating with the nodule_diameter to ensure the translation being within the range of the nodule boundary
				'''
				tnz = (np.absolute(translation)>1).nonzero()[0]
				if tnz.size==0 and t!=0:	#the translation is too tiny to distinguish
					#print('diameter:{} scale:{} translation:{} the translation is invisible' .format(nodule_diameter, scale, translation))
					continue
				'''
				tob = ((box_size/2-translation)>nodule_diameter/2).nonzero()[0]
				if not centering and tob.size==0:
					#print('diameter:{} scale:{} translation:{} nodule out of box' .format(nodule_diameter, scale, translation))
					continue

				zyx_1 = v_center + translation - window_size  #the order of indices is [Z, Y, X]
				zyx_2 = v_center + translation + box_size - window_size
				if mt.coord_overflow(zyx_1, volume_overbound.shape) or mt.coord_overflow(zyx_2, volume_overbound.shape, topopen=True):
					#print('diameter:{} scale:{} translation:{} the region is out of the bound of the volume' .format(nodule_diameter, scale, translation))
					continue
				nodule_box = np.zeros(shape=volume_shape, dtype=int)  # ---nodule_box_size = 45
				img_crop = volume_overbound[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]
				img_crop[img_crop<-1024] = -1024  # the voxel value below -1024 is set to -1024
				if scale==1.0:
					img_crop_rescaled = img_crop
				else:
					img_crop_rescaled, rescaled_spacing = resample(img_crop, np.array([1,1,1]), np.array([scale,scale,scale]))
				padding_shape = np.array((img_crop_rescaled.shape-volume_shape)/2, dtype=int)
				nodule_box = img_crop_rescaled[padding_shape[0]:padding_shape[0]+volume_shape[0], padding_shape[1]:padding_shape[1]+volume_shape[1], padding_shape[2]:padding_shape[2]+volume_shape[2]]
				if 'volume_batch' not in dir():
					volume_batch = nodule_box.reshape((1, volume_shape[0], volume_shape[1], volume_shape[2]))
				else:
					volume_batch = np.concatenate((volume_batch, nodule_box.reshape((1, volume_shape[0], volume_shape[1], volume_shape[2]))), axis=0)
				if rotation_augment:
					rot_box = np.rot90(nodule_box, k=1, axes=(2, 1))
					volume_batch = np.concatenate((volume_batch, rot_box.reshape(
						(1, volume_shape[0], volume_shape[1], volume_shape[2]))), axis=0)
					rot_box = np.rot90(nodule_box, k=2, axes=(2, 1))
					volume_batch = np.concatenate((volume_batch, rot_box.reshape(
						(1, volume_shape[0], volume_shape[1], volume_shape[2]))), axis=0)
					rot_box = np.rot90(nodule_box, k=3, axes=(2, 1))
					volume_batch = np.concatenate((volume_batch, rot_box.reshape(
						(1, volume_shape[0], volume_shape[1], volume_shape[2]))), axis=0)
					rot_box = np.rot90(nodule_box, k=1, axes=(2, 0))
					volume_batch = np.concatenate((volume_batch, rot_box.reshape(
						(1, volume_shape[0], volume_shape[1], volume_shape[2]))), axis=0)
					rot_box = np.rot90(nodule_box, k=2, axes=(2, 0))
					volume_batch = np.concatenate((volume_batch, rot_box.reshape(
						(1, volume_shape[0], volume_shape[1], volume_shape[2]))), axis=0)
					rot_box = np.rot90(nodule_box, k=3, axes=(2, 0))
					volume_batch = np.concatenate((volume_batch, rot_box.reshape(
						(1, volume_shape[0], volume_shape[1], volume_shape[2]))), axis=0)
					rot_box = np.rot90(nodule_box, k=1, axes=(1, 0))
					volume_batch = np.concatenate((volume_batch, rot_box.reshape(
						(1, volume_shape[0], volume_shape[1], volume_shape[2]))), axis=0)
					rot_box = np.rot90(nodule_box, k=2, axes=(1, 0))
					volume_batch = np.concatenate((volume_batch, rot_box.reshape(
						(1, volume_shape[0], volume_shape[1], volume_shape[2]))), axis=0)
					rot_box = np.rot90(nodule_box, k=3, axes=(1, 0))
					volume_batch = np.concatenate((volume_batch, rot_box.reshape(
						(1, volume_shape[0], volume_shape[1], volume_shape[2]))), axis=0)
				if flip_augment:
					flip_box = nodule_box[::-1,:,:]
					volume_batch = np.concatenate((volume_batch, flip_box.reshape(
						(1, volume_shape[0], volume_shape[1], volume_shape[2]))), axis=0)
					flip_box = nodule_box[:,::-1,:]
					volume_batch = np.concatenate((volume_batch, flip_box.reshape(
						(1, volume_shape[0], volume_shape[1], volume_shape[2]))), axis=0)
					flip_box = nodule_box[:,:,::-1]
					volume_batch = np.concatenate((volume_batch, flip_box.reshape(
						(1, volume_shape[0], volume_shape[1], volume_shape[2]))), axis=0)
					
	if 'volume_batch' not in dir():
		print('volume extraction failed')
		np.save('error.npy', volume_overbound)
	return volume_batch
	
def add_noise(image, noise_range, unsigned=True):
	image_noised = image + np.random.uniform(*noise_range, image.shape)
	anegative = np.zeros_like(image)-1
	if anegative.min()>0:
		#the data type is unsigned int
		image_noised[image_noised<0] = 0
		image_noised = np.minimum(image_noised, anegative)
	image_noised = image_noised.astype(image.dtype)
	return image_noised
