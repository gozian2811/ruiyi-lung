import os
import math
import random
import copy
import array
import scipy.ndimage
import matplotlib.pyplot as plt
import numpy as np

def read_environment(envfile):
	ef = open(envfile)
	environment = ef.readlines()
	img_width = int(environment[0])
	img_height = int(environment[1])
	num_view = int(environment[2])
	max_bound = int(environment[3])
	min_bound = int(environment[4])
	pixel_mean = float(environment[5])
	ef.close()
	return img_width, img_height, num_view, max_bound, min_bound, pixel_mean

def get_serie_uid(filepath):
	filename = os.path.basename(filepath)
	fileparts = os.path.splitext(filename)
	return fileparts[0]
	
def write_mhd_file(mhdfile, data, dsize):
	def write_meta_header(filename, meta_dict):
		header = ''
		# do not use tags = meta_dict.keys() because the order of tags matters
		tags = ['ObjectType', 'NDims', 'BinaryData', 'NoduleDiameter'
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
		#---write data to file
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

def set_window_width(image, MIN_BOUND=-1024.0):
	image[image < MIN_BOUND] = MIN_BOUND
	return image
	
def coord_overflow(coord, shape):
	upbound = shape - coord	
	if coord[coord<0].size>0 or upbound[upbound<=0].size>0:
		return True
	else:
		return False

def resample(image, old_spacing, new_spacing=[1, 1, 1]):
	resize_factor = old_spacing / new_spacing
	new_real_shape = image.shape * resize_factor
	new_shape = np.round(new_real_shape)
	real_resize_factor = new_shape / image.shape
	new_spacing = old_spacing / real_resize_factor
	if image.shape[0]<1000:
        	image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
	else:
		num_batch = int(math.ceil(image.shape[0]/1000))
		for b in range(num_batch):
			image_batch = image[b*1000:min((b+1)*1000,image.shape[0]), :, :]
			image_batch = scipy.ndimage.interpolation.zoom(image_batch, real_resize_factor, mode='nearest')
			if 'new_image' in dir():
				new_image = np.append(new_image, image_batch, axis=0)
			else:
				new_image = image_batch
		image = new_image

	return image, new_spacing

def make_patchs(voxel):
	width, length, height = voxel.shape
	patch_size = np.min(voxel.shape)
	patchs = np.zeros(shape=(9,patch_size,patch_size), dtype = float)
	patchs[0] = voxel[:,:,int(height/2)]
	patchs[1] = voxel[:,int(length/2),:]
	patchs[2] = voxel[int(width/2),:,:]
	for h in range(height):
		patchs[3,:,h] = voxel[:,h,h]
		patchs[4,h,:] = voxel[h,:,h]
		patchs[5,:,h] = voxel[:,h,height-h-1]
		patchs[6,h,:] = voxel[h,:,height-h-1]
	for w in range(width):
		patchs[7,w,:] = voxel[w,w,:]
		patchs[8,w,:] = voxel[width-w-1,w,:]
	
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