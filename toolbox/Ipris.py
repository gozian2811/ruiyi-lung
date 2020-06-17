import math
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

#Some of the gradient magnitude are of negative values, and masks of LIDC-IDRI-0002 contain some outliers.
#the management of negative gradient magnitude may be improved in the future, and the normal calculation method is not convincing and need to be further proved.

def mask_compare(mask1, mask2):
	if mask1==mask2:
		return 0
	if mask1>mask2 or mask1==1:
		#the mask1 indicates a region inner than mask2
		return -1
	else:
		#the mask1 indicates a region outer than mask2
		return 1

def approx_normal(mask_field):
	label_field = mask_field == mask_field[1][1]
	label_index = label_field.nonzero()
	if len(label_index[0])==3:
		#calculate the normal with coordinates of two adjacent points
		hollow_field = label_field.copy()
		hollow_field[1, 1] = 0
		adjacents = hollow_field.nonzero()
		surfdir = (adjacents[0][0]-adjacents[0][1], adjacents[1][0]-adjacents[1][1])
		normal = (surfdir[1], -surfdir[0])
	else:
		normal = ((label_index[0] - 1).sum(), (label_index[1] - 1).sum())
	#if max(normal) == 0 and min(normal) == 0:
	#	return None
	if max(normal) == 0 and min(normal) == 0:
		# the points near the center are distributed symetric
		hollow_field = label_field.copy()
		hollow_field[1, 1] = 0
		segment_labels = measure.label(hollow_field, background=0)
		oneside = np.where(segment_labels == 1)
		surfdir = ((oneside[0] - 1).sum(), (oneside[1] - 1).sum())
		if max(surfdir) == 0 and min(surfdir) == 0:
			#print('error:could not find normal')
			return None
		normal = (surfdir[1], -surfdir[0])
	normal_normalized = np.array(normal) / math.sqrt(normal[0] ** 2 + normal[1] ** 2)
	normal_point = (int(1+normal_normalized[0]+0.5), int(1+normal_normalized[1]+0.5))
	reverse_point = (int(1-normal_normalized[0]+0.5), int(1-normal_normalized[1]+0.5))
	normal_mask = mask_field[normal_point[0]][normal_point[1]]
	reverse_mask = mask_field[reverse_point[0]][reverse_point[1]]
	normal_judge = mask_compare(normal_mask, reverse_mask)
	if normal_judge==1:
		return normal_normalized
	elif normal_judge==-1:
		return -normal_normalized
	else:
		return None
	#if mask_field[normal_point[0]][normal_point[1]]>mask_field[1][1] or mask_field[normal_point[0]][normal_point[1]]==1:
		#the normal points to the inside, flip it
		#normal_normalized = -normal_normalized

def gradient_magnitude(nodule_slice, pcoord, normal):
	pcoord = np.array(pcoord)
	normal = np.array(normal)
	aftpoint = pcoord + normal
	prepoint = pcoord - normal
	aftvalue = nodule_slice[int(aftpoint[0]+0.5)][int(aftpoint[1]+0.5)]
	prevalue = nodule_slice[int(prepoint[0]+0.5)][int(prepoint[1]+0.5)]
	gm = (prevalue - aftvalue) / 2.0	#assuming the length of normal is 2
	if gm < 0:
		gm = gm
	return gm

def average_difference(nodule_slice, pcoord, normal, mode, samplenum=5):
	#the value of mode lies in {'gradient', 'intensity'}
	sidesnum = int(samplenum/2)
	ad = 0
	for q in range(sidesnum):
		step = normal * (q + 1)
		outpoint = (int(pcoord[0]+step[0]+0.5), int(pcoord[1]+step[1]+0.5))
		inpoint = (int(pcoord[0]-step[0]+0.5), int(pcoord[1]-step[1]+0.5))
		if mode=='gradient':
			#outvalue = nodule_slice[outpoint[0]][outpoint[1]] - nodule_slice[pcoord[0]][pcoord[1]]
			#invalue = nodule_slice[inpoint[0]][inpoint[1]] - nodule_slice[pcoord[0]][pcoord[1]]
			outvalue = gradient_magnitude(nodule_slice, outpoint, normal)
			invalue = gradient_magnitude(nodule_slice, inpoint, normal)
		elif mode=='intensity':
			outvalue = nodule_slice[outpoint[0]][outpoint[1]]
			invalue = nodule_slice[inpoint[0]][inpoint[1]]
		else:
			print('Failed: unknown type of mode.')
			exit()
		ad += (invalue - outvalue) / float((q + 1) * 2)
	ad /= float(sidesnum)
	return ad

def average_gradient(nodule_slice, pcoord, normal, mode, samplenum=5):
	# the value of mode lies in {'sharpness', 'entropy'}
	pcoord = np.array(pcoord)
	normal = np.array(normal)
	sidesnum = int(samplenum / 2)
	ag = 0
	#startcoord = pcoord - normal * sidesnum
	for r in range(-sidesnum, sidesnum+1):
		#aftcoord = startcoord + normal * (r + 1)
		#precoord = startcoord + normal * r
		#aftvalue = nodule_slice[int(aftcoord[0]+0.5)][int(aftcoord[1]+0.5)]
		#prevalue = nodule_slice[int(precoord[0] + 0.5)][int(precoord[1] + 0.5)]
		#magnitude = prevalue - aftvalue
		currcoord = pcoord + normal * r
		magnitude = gradient_magnitude(nodule_slice, currcoord, normal)
		if mode=='sharpness':
			ag += magnitude
		elif mode=='entropy':
			if magnitude > 0:
				ag += magnitude * math.log(magnitude) / math.log(2)
			elif magnitude < 0:
				ag -= (-magnitude) * math.log(-magnitude) / math.log(2)
		else:
			print('Failed: unknown type of mode.')
			exit()
	if mode=='sharpness':
		ag /= float(sidesnum * 2)
	return ag

def statistic(values):
	value_array = np.array(values)
	mean = value_array.mean()
	maximum = value_array.max()
	minimum = value_array.min()
	standard_deviation = value_array.std()
	return mean, standard_deviation, minimum, maximum

def ipris_feature(volume, mask):
	gray_profile_2 = []
	gradient_sharpness_1 = []
	gradient_magnitude_entropy_1 = []
	neighbors = [[0,1], [1,0], [0,-1], [-1,0]]
	for si in range(len(volume)):
		nodule_slice = volume[si]
		mask_slice = mask[si]
		inner = np.where(mask_slice==1)
		if inner[0].size==0: continue
		
		secondshell = []
		for y, x in np.nditer([inner[0], inner[1]]):
			x = int(x)
			y = int(y)
			for neighbor in neighbors:
				if mask_slice[y+neighbor[0]][x+neighbor[1]]==2:
					#pixel of the second shell
					mask_slice[y][x] = 3
					secondshell.append((y, x))
					break

		for y, x in secondshell:
			'''
			label_field = mask_slice[y-1:y+2,x-1:x+2]==mask_slice[y, x]
			label_index = label_field.nonzero()
			normal = ((label_index[0]-1).mean(), (label_index[1]-1).mean())
			if max(normal)==0 and min(normal)==0:
				#the points near the center are distributed symetric
				hollow_field = label_field.copy()
				hollow_field[1,1] = 0
				segment_labels = measure.label(hollow_field, background=0)
				oneside = np.where(segment_labels==1)
				surfdir = ((oneside[0]-1).mean(), (oneside[1]-1).mean())
				if max(surfdir)==0 and min(surfdir)==0:
					print('error:could not find normal')
					exit()
				normal = (surfdir[1], -surfdir[0])
			'''
			normal = approx_normal(mask_slice[y-1:y+2,x-1:x+2])
			if normal is None:
				print("({},{},{}) unable to generate normal" .format(si, y, x))
			else:
				gray_profile_2.append(average_difference(nodule_slice, (y, x), normal, 'intensity'))
		
		outshell = np.where(mask_slice==2)
		for y, x in np.nditer([outshell[0], outshell[1]]):
			x = int(x)
			y = int(y)
			if si==60 and y==328 and x==432:
				print(y, x)
			normal = approx_normal(mask_slice[y - 1:y + 2, x - 1:x + 2])
			if normal is None:
				print("({},{},{}) unable to generate normal" .format(si, y, x))
			else:
				gradient_magnitude_entropy_1.append(average_gradient(nodule_slice, (y, x), normal, 'entropy'))
				gradient_sharpness_1.append(average_gradient(nodule_slice, (y, x), normal, 'sharpness'))

	gray_profile_2_statistic = statistic(gray_profile_2)
	gradient_magnitude_entropy_1_statistic = statistic(gradient_magnitude_entropy_1)
	gradient_sharpness_1_statistic = statistic(gradient_sharpness_1)
	result_feature = np.concatenate((np.array(gray_profile_2_statistic), np.array(gradient_magnitude_entropy_1_statistic), np.array(gradient_sharpness_1_statistic)), axis=0)
	return result_feature