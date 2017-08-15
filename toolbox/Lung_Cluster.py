# import the necessary packages
from skimage import data, io, segmentation, color
from skimage.future import graph
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.segmentation import watershed
from skimage.util import img_as_float
from random import random
import matplotlib.pyplot as plt
import argparse
import SimpleITK as sitk
import numpy as np
from . import MITools as mt
from . import CTViewer as cv

NODULE_THRESHOLD = -600

# load the image and convert it to a floating point data type
def normalization(x):
	x = np.array(x, dtype=float)
	Min = np.min(x)
	Max = np.max(x)
	x = (x - Min) / (Max - Min)
	return x

def weight_mean_color(graph, src, dst, n):
	diff = graph.node[dst]['mean color'] - graph.node[n]['mean color']
	diff = np.linalg.norm(diff)
	return {'weight': diff}

def merge_mean_color(graph, src, dst):
	graph.node[dst]['total color'] += graph.node[src]['total color']
	graph.node[dst]['pixel count'] += graph.node[src]['pixel count']
	graph.node[dst]['mean color'] = (graph.node[dst]['total color'] / graph.node[dst]['pixel count'])

def cluster_filter(volume, labels):
	labels_filtered = labels.copy()
	num_labels = labels.max() + 1	#the default range of labels is 0 to max
	for l in range(num_labels):
		clvalues = volume[labels==l]
		if clvalues.size>0 and clvalues.max()<NODULE_THRESHOLD:
			labels_filtered[labels==l] = -1
	return labels_filtered

def cluster_merge(volume, labels):
	g = graph.rag_mean_color(volume, labels)
	labels_merged = graph.merge_hierarchical(labels, g, thresh=0.03, rag_copy=False, in_place_merge=True,
					   merge_func=merge_mean_color, weight_func=weight_mean_color)
	return labels_merged
	
def threshold_mask(segimage, lungmask, threshold=NODULE_THRESHOLD):
	shape = segimage.shape
	tissuemask = np.zeros(shape,dtype=int)
	for i in range(shape[0]):
		for j in range(shape[1]):
			for k in range(shape[2]):
				#Do judge
				if(lungmask[i][j][k]==1 and segimage[i][j][k]>threshold):
					tissuemask[i][j][k]=1
					#order z,y,x
	np.where(tissuemask == 1)
	return tissuemask
    
def cluster_centers(cluster_labels, eliminate_upper_size=20000):
	num_labels = cluster_labels.max() + 1	#the default range of labels is 0 to max
	cluster_sizes = np.zeros(shape=(num_labels), dtype=int)
	centers = np.zeros(shape=(num_labels, 3), dtype=float)
	for z in range(cluster_labels.shape[0]):
		for y in range(cluster_labels.shape[1]):
			for x in range(cluster_labels.shape[2]):
				label = cluster_labels[z][y][x]
				if label>=0 and cluster_sizes[label] >= 0:
					cluster_sizes[label] += 1
					centers[label] += np.array([z, y, x])
					if cluster_sizes[label] > eliminate_upper_size:
						# no longer to caluculate the center of this cluster for its too large
						cluster_sizes[label] = -1

	clcenters = []
	cllabels = []
	for i in range(num_labels):
		if cluster_sizes[i] > 0:
			center = np.array(np.round(centers[i]/cluster_sizes[i]), dtype=int)
			clcenters.append(center)
			cllabels.append(i)

	return clcenters, cllabels

def segment_vision(volume, labels):
	if volume.min()<0 or volume.max()>1:
		volume = normalization(volume)
	segvision = np.zeros(shape=(volume.shape[0], volume.shape[1], volume.shape[2], 3), dtype=np.float64)
	#cv.view_CT(labels)
	for z in range(volume.shape[0]):
		segvision[z] = mark_boundaries(volume[z], labels[z], mode='inner')
	return segvision

def seed_coord_cluster(index, clsize):
	numcoords = len(index[0])
	if numcoords == 0:
		return []
	coords = []
	for i in range(len(index[0])):
		coords.append([index[0][i], index[1][i], index[2][i]])

	clnum = 0
	index_cluster = 0 - np.ones(numcoords, dtype=int)
	steps = [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, 1, 1], [0, 1, -1], [0, -1, 0], [0, -1, 1], [0, -1, -1],
		 [1, 0, 0], [1, 0, 1], [1, 0, -1], [1, 1, 0], [1, 1, 1], [1, 1, -1], [1, -1, 0], [1, -1, 1],
		 [1, -1, -1],
		 [-1, 0, 0], [-1, 0, 1], [-1, 0, -1], [-1, 1, 0], [-1, 1, 1], [-1, 1, -1], [-1, -1, 0], [-1, -1, 1],
		 [-1, -1, -1]]
	# steps = [[0,0,1],[0,0,-1],[0,1,0],[0,-1,0],[1,0,0],[-1,0,0]]
	# clustering by seeds
	clusters = []
	for i in range(0, numcoords):
		if index_cluster[i] < 0:
			print("%d" % (i))
			clnum += 1
			cluster_stack = [i]
			index_cluster[i] = clnum - 1
			clusters.append([i])
			size = 1
			while len(cluster_stack) > 0 and size <= clsize:
				pind = cluster_stack.pop(0)
				for step in steps:
					neighbor = [coords[i][0] + step[0], coords[i][1] + step[1],
						    coords[i][2] + step[2]]
					if coords.count(neighbor) > 0:
						nind = coords.index(neighbor)
						if index_cluster[nind] < 0:
							size += 1
							cluster_stack.append(nind)
							index_cluster[nind] = clnum - 1
							clusters[-1].append(nind)

	# calculate the cluster center
	clind = 0
	clend = False
	clcenters = []
	while index_cluster.count(clind) > 0:
		summary = [0.0, 0.0, 0.0]
		size = 0
		for i in range(len(index_cluster)):
			if index_cluster[i] == clind:
				size += 1
				summary = [summary[0] + coords[i][0], summary[1] + coords[i][1],
					   summary[2] + coords[i][2]]
		center = np.array([round(summary[0] / size), round(summary[1] / size), round(summary[2] / size)],
				  dtype=int)
		clcenters.append(center)

	# the coordination order is z, y, x
	return clcenters

def seed_volume_cluster(nodule_matrix, cluster_size=-1, eliminate_lower_size=5, result_vision=False):
	clnum = 0
	index_cluster = 0 - np.ones(nodule_matrix.shape, dtype=int)
	steps = [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, 1, 1], [0, 1, -1], [0, -1, 0], [0, -1, 1], [0, -1, -1],
		 [1, 0, 0], [1, 0, 1], [1, 0, -1], [1, 1, 0], [1, 1, 1], [1, 1, -1], [1, -1, 0], [1, -1, 1],
		 [1, -1, -1],
		 [-1, 0, 0], [-1, 0, 1], [-1, 0, -1], [-1, 1, 0], [-1, 1, 1], [-1, 1, -1], [-1, -1, 0], [-1, -1, 1],
		 [-1, -1, -1]]
	# steps = [[0,0,1],[0,0,-1],[0,1,0],[0,-1,0],[1,0,0],[-1,0,0]]
	# clustering by seeds
	#clusters = []
	for z in range(nodule_matrix.shape[0]):
		for y in range(nodule_matrix.shape[1]):
			for x in range(nodule_matrix.shape[2]):
				#print("%d %d %d" %(z, y, x))
				if nodule_matrix[z][y][x] > 0 and index_cluster[z][y][x] < 0:
					clnum += 1
					cluster_stack = [[z, y, x]]
					index_cluster[z][y][x] = clnum - 1
					#clusters.append([[z, y, x]])
					size = 1
					while len(cluster_stack) > 0:
						coord = cluster_stack.pop(0)
						for step in steps:
							neighbor = np.array([coord[0] + step[0], coord[1] + step[1],
									     coord[2] + step[2]], dtype=int)
							if not mt.coord_overflow(neighbor, nodule_matrix.shape) and \
									nodule_matrix[
										neighbor[0], neighbor[1], neighbor[
											2]] > 0 and index_cluster[
								neighbor[0], neighbor[1], neighbor[2]] < 0:
								size += 1
								cluster_stack.append(neighbor)
								index_cluster[neighbor[0], neighbor[1], neighbor[
									2]] = clnum - 1
								#clusters[-1].append(neighbor)
						if cluster_size > 0 and size > cluster_size:
							break
	# calculate the cluster center
	#clcenters = []
	# cloutput = open("clsizes.txt", "w")
	#for cluster in clusters:
		# cloutput.write("%d " %(len(cluster)))
		#summary = np.array([0.0, 0.0, 0.0])
		#for point in cluster:
		#	summary = [summary[0] + point[0], summary[1] + point[1], summary[2] + point[2]]
		#if len(cluster) <= eliminate_lower_size:
		#	continue
		#center = np.array([round(summary[0] / len(cluster)), round(summary[1] / len(cluster)),
		#		   round(summary[2] / len(cluster))], dtype=int)
		# center = [int(round(center[0]/len(cluster))), int(round(center[1]/len(cluster))), int(round(center[2]/len(cluster)))]
		#clcenters.append(center)
	# cloutput.close()

	if result_vision:
		rv = np.zeros(shape=(nodule_matrix.shape[0], nodule_matrix.shape[1], nodule_matrix.shape[2], 3))
		cluster_colors = np.random.rand(index_cluster.max()+1, 3)
		for z in range(nodule_matrix.shape[0]):
			for y in range(nodule_matrix.shape[1]):
				for x in range(nodule_matrix.shape[2]):
					cind = index_cluster[z, y, x]
					if cind>0:
						rv[z, y, x] = cluster_colors[cind]
		'''
		for cl in clusters:
			r = round(random.random(), 4)
			g = round(random.random(), 4)
			b = round(random.random(), 4)
			color = np.array([r, g, b])
			for coord in cl:
				rv[coord[0], coord[1], coord[2]] = color
		'''
		cv.view_CT(rv)

	# the coordination order is z, y, x
	#return clcenters, index_cluster
	return index_cluster

def slic_segment(volume, num_segments=500000, compactness=0.001, merge_cluster=False, result_output=False, view_result=False):
	volume = normalization(volume)
	labels = slic(volume, n_segments=num_segments, sigma=1, multichannel=False, compactness=compactness, slic_zero=True,
		      max_iter=15)

	if merge_cluster:
		labels = cluster_merge(volume, labels)
	'''
	g = graph.rag_mean_color(volume, labels)
	labels_merged = graph.merge_hierarchical(labels, g, thresh=0.03, rag_copy=False, in_place_merge=True,
					   merge_func=merge_mean_color, weight_func=weight_mean_color)
	segresult = np.zeros(shape=(volume.shape[0], volume.shape[1], volume.shape[2], 3), dtype=np.float64)
	for z in range(volume.shape[0]):
		segresult[z] = mark_boundaries(volume[z], labels_merged[z], mode='inner')
	'''
	if result_output or view_result:
		segresult = segment_vision(volume, labels)
		if result_output:
			np.save('detection_vision/slic_result.npy', segresult)
		if view_result:
			cv.view_CT(segresult)
		
	return labels
	
def seed_segment(volume, lung_mask, cluster_size=-1, view_result=False):
	organ_mask = threshold_mask(volume, lung_mask)
	labels = seed_volume_cluster(organ_mask, cluster_size)
	labels_merged = cluster_merge(volume, labels)
	if view_result:
		segresult = segment_vision(volume, labels_merged)
		cv.view_CT(segresult)
		
	return labels_merged