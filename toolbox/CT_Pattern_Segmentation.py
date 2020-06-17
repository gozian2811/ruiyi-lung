#-*-coding:utf-8-*-

import SimpleITK as sitk
import pandas as pd
import numpy as np
import skimage
import time
import os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import convex_hull_image, disk, ball
from skimage.morphology import binary_dilation as bd
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
from scipy.ndimage.morphology import binary_dilation, binary_closing, generate_binary_structure
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.misc
from tqdm import tqdm

data_path = "../TIANCHI_data/train/"
scan_file = data_path + "LKDS-00001.mhd"

def load_scan(file):
	full_scan = sitk.ReadImage(scan_file)
	img_array = sitk.GetArrayFromImage(full_scan)  #numpy数组，z,y,x
	origin = np.array(full_scan.GetOrigin())[::-1]   #世界坐标原点 z,y,x
	old_spacing = np.array(full_scan.GetSpacing())[::-1]   #原体素间距
	return img_array, origin, old_spacing


def resample(image, old_spacing, new_spacing=[1, 1, 1]):
	'''
	将体素间距设为(1, 1, 1)
	'''
	resize_factor = old_spacing / new_spacing
	new_shape = image.shape * resize_factor
	new_shape = np.round(new_shape)
	real_resize_factor = new_shape / image.shape
	new_spacing = old_spacing / real_resize_factor
	image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

	return image, new_spacing
    

def plot_3d(image, threshold=-600):
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
	plt.show()

def largest_label_image(image, bg=-1):
	vals, counts = np.unique(image, return_counts=True)    # 统计image中的值及频率
	counts = counts[vals != bg]
	vals = vals[vals != bg]
	if len(counts) > 0:
		maxind = np.argmax(counts)
		return vals[maxind], counts[maxind]
	else:
		return None, None

def segment_lung_slice(ima,fill_lung_structures=True):
	#make through the ima
	for yt in range(ima.shape[0]):
		ima[yt,0] = ima[0,0]
	for xt in range(ima.shape[1]):
		ima[-1,xt] = ima[0,0]

	shape=ima.shape
	binary_image = np.array(ima > -320, dtype=np.int8) + 1
	#calculate the connected region
	labels = measure.label(binary_image)
	#fill the air around the person
	background_label = labels[0,0]
	# Fill the air around the person

	#一开始相当于按照>-320与<-320进行了区分,这里把外部空气区域也认为是2了，这里人身体外就都是纯白了
	binary_image[background_label == labels] = 2

	# Method of filling the lung structures (比形态选择要好)

	if fill_lung_structures:
		# For every slice we determine the largest solid structure
		labeling = measure.label(binary_image-1)
		l_max = largest_label_image(labeling, bg=0)
		if l_max is not None:  # This slice contains some other things (needn't be lung)
			binary_image[labeling != l_max] = 1

	binary_image -= 1  # Make the image actual binary
	binary_image = 1 - binary_image  # Invert it, lungs are now 1,background are 0

	# Remove other air pockets insided body
	labels = measure.label(binary_image, background=0)
	l_max = largest_label_image(labels, bg=0)

	if l_max is not None:  # if This image has lungs
		binary_image[labels != l_max] = 0
	#Here we have make no lung part all zero
	#Lung with inner sth 1 ,other 0

	area = float(shape[0] * shape[1])
	val, count = np.unique(binary_image, return_counts=True)
	if count[val == 0 ] / area < 0.97:  # which means the area of lung in the picture is over 2%
		flag = 1
	else:
		flag = 0

	##find some special points
	#######################################
	#for being careful ,I notice many details here,to avoid many other things

	segimage=deepcopy(ima)
	segimage[binary_image==0]=1024 # This step turned the non-lung position in this image to 0，and save the lung position
	return segimage, binary_image, flag
	
def segment_lung_mask_slice_fast(image, hu_threshold, fill_lung_structures=True):
	# not actually binary, but 1 and 2.
	# 0 is treated as background, which we do not want
	binary_image = np.array(image > hu_threshold, dtype=np.int8) + 1
	#get through the light line at the bottom
	binary_image = np.pad(binary_image, ((2,0),(0,0)), 'constant', constant_values=((2,2),(2,2)))
	binary_image[0,:] = 1
	binary_image[:,0] = 1
	binary_image[:,-1] = 1
	# 获得阈值图像
	labels = measure.label(binary_image, connectivity=1)
	# label()函数标记连通区域
	# Pick the pixel in the very corner to determine which label is air.
	#   Improvement: Pick multiple background labels from around the patient
	#   More resistant to "trays" on which the patient lays cutting the air
	#   around the person in half
	background_label = labels[0,0]
	#Fill the air around the person
	binary_image[labels==background_label] = 2
	# Method of filling the lung structures (that is superior to something like
	# morphological closing)
	binary_image -= 1 #Make the image actual binary
	if fill_lung_structures:
		labeling = measure.label(binary_image)
		l_max, _ = largest_label_image(labeling, bg=labeling[0,0])
		if l_max is not None: #This slice contains some lung
			binary_image[np.logical_and(labeling!=l_max, labeling!=labeling[0,0])] = 1
	binary_image = 1 - binary_image #Invert it, lungs are now 1
	# Remove other air pockets insided body
	labels = measure.label(binary_image, background=0)
	l_max, maxcount = largest_label_image(labels, bg=0)
	if l_max is not None: # There are air pockets
		background_label = labels[0,0]
		min_label = labels.min()
		_, labelcounts = np.unique(labels, return_counts=True)
		maxlabel = labels.max()
		label_coords = np.where(labels!=background_label)
		for lci in range(len(label_coords[0])):
			label = labels[label_coords[0][lci]][label_coords[1][lci]] - min_label
			if labelcounts[label]<maxcount/4:
				binary_image[label_coords[0][lci]][label_coords[1][lci]] = 0

	if maxcount is None or maxcount < image.size * 0.01:
		return np.zeros_like(image, dtype=np.int8)
	binary_image = np.delete(binary_image, [0,1], axis=0)
	return binary_image
	
def segment_lung_mask_fast(image, hu_threshold=-600, fill_lung_structures=True, dilation=True):
	# not actually binary, but 1 and 2.
	# 0 is treated as background, which we do not want
	binary_image = np.array(image>hu_threshold, dtype=np.int8) + 1
	#binary_image = np.array(np.logical_or(image>hu_threshold, image<-1024), dtype=np.int8) + 1
	#get through the light line at the bottom
	binary_image = np.pad(binary_image, ((2,0),(0,0),(0,0)), 'constant', constant_values=((2,2),(2,2),(2,2)))
	binary_image[0,:,:] = 1
	binary_image[:,0,:] = 1
	binary_image[:,-1,:] = 1
	# 获得阈值图像
	labels = measure.label(binary_image, connectivity=1)
	# label()函数标记连通区域
	# Pick the pixel in the very corner to determine which label is air.
	#   Improvement: Pick multiple background labels from around the patient
	#   More resistant to "trays" on which the patient lays cutting the air
	#   around the person in half
	background_label = labels[0,0,0]
	#Fill the air around the person
	binary_image[labels==background_label] = 2
	# Method of filling the lung structures (that is superior to something like
	# morphological closing)
	if fill_lung_structures:
		# For every slice we determine the largest solid structure
		for i, axial_slice in enumerate(binary_image):
			axial_slice = axial_slice - 1
			labeling = measure.label(axial_slice)
			l_max, _ = largest_label_image(labeling, bg=labeling[0,0])
			if l_max is not None: #This slice contains some lung
				binary_image[i][np.logical_and(labeling!=l_max, labeling!=labeling[0,0])] = 1
		#labeling = measure.label(binary_image - 1)
		#l_max = largest_label_image(labeling, bg=0)
		#if l_max is not None:
		#    binary_image[labeling!=l_max] = 1
	binary_image -= 1 #Make the image actual binary
	binary_image = 1 - binary_image #Invert it, lungs are now 1
	# Remove other air pockets insided body
	labels = measure.label(binary_image, background=0)
	l_max, maxcount = largest_label_image(labels, bg=0)
	if l_max is not None: # There are air pockets
		background_label = labels[0,0,0]
		min_label = labels.min()
		_, labelcounts = np.unique(labels, return_counts=True)
		maxlabel = labels.max()
		label_coords = np.where(labels!=background_label)
		for lci in range(len(label_coords[0])):
			label = labels[label_coords[0][lci]][label_coords[1][lci]][label_coords[2][lci]] - min_label
			if labelcounts[label]<maxcount/4:
				binary_image[label_coords[0][lci]][label_coords[1][lci]][label_coords[2][lci]] = 0

	binary_image = np.delete(binary_image, [0,1], axis=0)
	if maxcount < image.size * 0.001:
		binary_image = np.zeros_like(image, dtype=np.int8)
		for z in range(len(binary_image)):
			binary_image[z] = segment_lung_mask_slice_fast(image[z], hu_threshold)
		if (binary_image>0).sum() < image.size * 0.001:
			return None
	if np.count_nonzero(binary_image[:,-1,:])>0:
		#wrong segmentation, which need correction
		labeling = measure.label(binary_image, background=0)
		for l in np.unique(labeling[:,-1,:]):
			if l!=0:
				binary_image[labeling==l] = 0
	if dilation:
		binary_image = extend_mask(binary_image, ball_size=3, iterations=6)
	return binary_image

def segment_body_mask(image, hu_threshold=-600):
	# 0 is treated as background, which we do not want
	binary_image = np.array(image > hu_threshold, dtype=np.int8) + 1
	#get through the light line at the bottom
	binary_image = np.pad(binary_image, ((1,0),(0,0),(0,0)), 'constant', constant_values=((1,1),(1,1),(1,1)))
	binary_image[:,0,:] = 1
	binary_image[:,-1,:] = 1
	# 获得阈值图像
	labels = measure.label(binary_image, connectivity=1)
	background_label = labels[0,0,0]
	#Fill the air around the person
	binary_image[np.logical_and(binary_image==1, labels!=background_label)] = 2
	binary_image -= 1

	labels = measure.label(binary_image, background=0)
	l_max, maxcount = largest_label_image(labels, bg=0)
	if l_max is not None and maxcount>image.size*0.001:	#There are air pockets
		background_label = labels[0,0,0]
		min_label = labels.min()
		_, labelcounts = np.unique(labels, return_counts=True)
		maxlabel = labels.max()
		label_coords = np.where(labels!=background_label)
		for lci in range(len(label_coords[0])):
			label = labels[label_coords[0][lci]][label_coords[1][lci]][label_coords[2][lci]] - min_label
			#if label==labels[1, 300, 163]-min_label:
			#	print(label)
			if labelcounts[label]<maxcount/4:
				binary_image[label_coords[0][lci]][label_coords[1][lci]][label_coords[2][lci]] = 0
		binary_image = np.delete(binary_image, 0, axis=0)
		return binary_image
	else:
		return None

def process_mask(mask):
	convex_mask = np.copy(mask)
	for i_layer in range(convex_mask.shape[0]):
		mask1  = np.ascontiguousarray(mask[i_layer])
		if np.sum(mask1)>0:
			mask2 = convex_hull_image(mask1)
			if np.sum(mask2)>2*np.sum(mask1):
				mask2 = mask1
		else:
			mask2 = mask1
		convex_mask[i_layer] = mask2
	struct = generate_binary_structure(3,1)  
	dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10) 
	return dilatedMask

def extend_bounding(img):
	selem = disk(3)
	mask = bd(img, selem)

	return mask

def extend_mask(imagemask, ball_size=4, iterations=15):
	padding = ball_size * iterations
	mask = np.zeros(shape=(imagemask.shape[0]+2*padding, imagemask.shape[1]+2*padding, imagemask.shape[2]+2*padding), dtype=bool)
	mask[padding:imagemask.shape[0]+padding, padding:imagemask.shape[1]+padding, padding:imagemask.shape[2]+padding][imagemask==1] = imagemask[imagemask==1]
	#mask = np.pad(imagemask, ((padding,padding), (padding,padding), (padding, padding)), 'constant', ((0,0),(0,0),(0,0)))
	#mask = np.copy(imagemask)
	#for i_layer in range(mask.shape[0]):
	#    slice_img = mask[i_layer]
	#    mask_img = extend_bounding(slice_img)
	#    mask[i_layer] = mask_img

	#struct = generate_binary_structure(3,1)
	struct = ball(ball_size)
	#dilatedMask = binary_dilation(mask,structure=struct,iterations=10)
	mask = binary_closing(mask, structure=struct, iterations=iterations)
	return mask[padding:imagemask.shape[0]+padding, padding:imagemask.shape[1]+padding, padding:imagemask.shape[2]+padding]

def del_surplus(lung_mask, image):
	for z in range(lung_mask.shape[0]):
		slice_z = lung_mask[z,:,:]       #轴向切片
		sum_z = slice_z.sum()            #计算轴向切片的和
		if(sum_z != 0):                  #和不等于0，说明包含需保留的组织
			z_top = z                    #确定该位置
			lung_mask = lung_mask[z_top:,:,:]          #保留z_top以下的部分
			image = image[z_top:,:,:]
			break

	for z in range(0, lung_mask.shape[0])[::-1]:
		slice_z = lung_mask[z,:,:]
		sum_z = slice_z.sum()
		if(sum_z != 0):
			z_bottom = z
			lung_mask = lung_mask[:z_bottom+1,:,:]     #保留z_bottom以上的部分
			image = image[:z_bottom+1,:,:]
			break

	for y in range(lung_mask.shape[1]):
		slice_y = lung_mask[:,y,:]
		sum_y = slice_y.sum()
		if(sum_y != 0):
			y_top = y
			lung_mask = lung_mask[:,y_top:,:]         #保留y_top以下的部分
			image = image[:,y_top:,:]
			break

	for y in range(0, lung_mask.shape[1])[::-1]:
		slice_y = lung_mask[:,y,:]
		sum_y = slice_y.sum()
		if(sum_y != 0):
			y_bottom = y
			lung_mask = lung_mask[:,:y_bottom+1,:]     #保留y_bottom以上的部分
			image = image[:,:y_bottom+1,:]
			break

	for x in range(lung_mask.shape[2]):
		slice_x = lung_mask[:,:,x]
		sum_x = slice_x.sum()
		if(sum_x != 0):
			x_top = x 
			lung_mask = lung_mask[:,:,x_top:]           #保留x_top以下的部分
			image = image[:,:,x_top:]
			break

	for x in range(0, lung_mask.shape[2])[::-1]:
		slice_x = lung_mask[:,:,x]
		sum_x = slice_x.sum()
		if(sum_x != 0):
			x_bottom = x
			lung_mask = lung_mask[:,:,:x_bottom+1]       # 保留x_bottom以上的部分
			image = image[:,:,:x_bottom+1]
			break

	return lung_mask, image

def mask_boundbox(lung_mask):
	class BoundBox:
		def __init__(self):
			self.z_top = -1
			self.z_bottom = -1
			self.y_top = -1
			self.y_bottom = -1
			self.x_top = -1
			self.x_bottom = -1
	
	boundbox = BoundBox()
	for z in range(lung_mask.shape[0]):
		slice_z = lung_mask[z,:,:]       #轴向切片
		sum_z = slice_z.sum()            #计算轴向切片的和
		if(sum_z != 0):                  #和不等于0，说明包含需保留的组织
			boundbox.z_top = z                    #确定该位置
			break

	for z in range(0, lung_mask.shape[0])[::-1]:
		slice_z = lung_mask[z,:,:]
		sum_z = slice_z.sum()
		if(sum_z != 0):
			boundbox.z_bottom = z
			break

	for y in range(lung_mask.shape[1]):
		slice_y = lung_mask[:,y,:]
		sum_y = slice_y.sum()
		if(sum_y != 0):
			boundbox.y_top = y
			break

	for y in range(0, lung_mask.shape[1])[::-1]:
		slice_y = lung_mask[:,y,:]
		sum_y = slice_y.sum()
		if(sum_y != 0):
			boundbox.y_bottom = y
			break

	for x in range(lung_mask.shape[2]):
		slice_x = lung_mask[:,:,x]
		sum_x = slice_x.sum()
		if(sum_x != 0):
			boundbox.x_top = x 
			break

	for x in range(0, lung_mask.shape[2])[::-1]:
		slice_x = lung_mask[:,:,x]
		sum_x = slice_x.sum()
		if(sum_x != 0):
			boundbox.x_bottom = x
			break
	#return z_top, z_bottom, y_top, y_bottom, x_top, x_bottom
	return boundbox

if __name__ == '__main__':
	img_array, origin, old_spacing = load_scan(scan_file)
	image, new_spacing = resample(img_array, old_spacing)
	#segmented_lungs = segment_lung_mask(image, False)
	segmented_lungs_filled = segment_lung_mask(image, True)
	#cv.view_CT(segmented_lungs_filled)

	process_lungs = extend_mask(segmented_lungs_filled)
	process_lungs, image = del_surplus(process_lungs, image)
	#cv.view_CT(process_lungs)
	seg_lung = image * process_lungs
	#plot_3d(process_lungs, 0)
	np.save("1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.npy", seg_lung)
