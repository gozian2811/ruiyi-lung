import math
import numpy as np
import functools as ft
import pandas as pd
from skimage import measure
from scipy.ndimage.interpolation import zoom
from . import basicconfig
from . import MITools as mt
from . import Lung_Cluster as lc
from . import CT_Pattern_Segmentation as cps
try:
	import torch
except:
	print('torch not installed')
try:
	from tqdm import tqdm # long waits are not fun
except:
	print('tqdm not installed')
	tqdm = lambda x : x

MAX_BOUND = basicconfig.config["MAX_BOUND"]
MIN_BOUND = basicconfig.config["MIN_BOUND"]
PIXEL_MEAN = basicconfig.config["PIXEL_MEAN"]
NORM_CROP = basicconfig.config["NORM_CROP"]

medical_normalization = ft.partial(mt.medical_normalization, max_bound=MAX_BOUND, min_bound=MIN_BOUND, pixel_mean=PIXEL_MEAN, crop=NORM_CROP, input_copy=False)

def slic_candidate(image, clsize=27, focus_area='lung'):
	if focus_area=='body':
		segmask = cps.segment_body_mask(image)
	elif focus_area=='lung':
		segmask = cps.segment_lung_mask_fast(image)
		#segmask = cps.extend_mask(segmask)
	else:
		segmask = np.ones_like(image, dtype=bool)
	if segmask is None:
		print('Lung segmentation failed')
		#patient_evaluations.write('%d/%d patient %s lung segmentation failed\n' %(p+1, len(test_patients), uid))
		return None
	#print('lung segment. time{}s' .format(time.time()-start_time))

	#np.save('segmask.npy', segmask)
	#print('segment mask saved')
	mbb = cps.mask_boundbox(segmask)
	'''
	mbb.z_top = 290
	mbb.z_bottom = 320
	mbb.y_top = 225
	mbb.y_bottom = 255
	mbb.x_top = 290
	mbb.x_bottom = 320
	'''
	image_bounded = image[mbb.z_top:mbb.z_bottom+1, mbb.y_top:mbb.y_bottom+1, mbb.x_top:mbb.x_bottom+1]
	#print('Lung Segmentation Done. time:{}s' .format(time.time()-start_time))

	#nodule_matrix, cindex = cd.candidate_detection(segimage)
	#cluster_labels = lc.seed_volume_cluster(nodule_matrix, cluster_size=30, result_vision=False)
	num_segments = int(image_bounded.size / clsize)
	#num_segments = 20
	#print('cluster number:%d' %(num_segments))
	cluster_labels = 0 - np.ones(shape=image.shape, dtype=int)
	cluster_labels_bounded = lc.slic_segment(image_bounded, compactness=0.01, num_segments=num_segments)
	#print('Clustering Done. time:{}s' .format(time.time()-start_time))
	cluster_labels[mbb.z_top:mbb.z_bottom+1, mbb.y_top:mbb.y_bottom+1, mbb.x_top:mbb.x_bottom+1] = cluster_labels_bounded
	cluster_labels[np.logical_or(segmask==0, image<-600)] = -1
	#np.save('slic_result_colored.npy', lc.segment_color_vision(cluster_labels))
	#np.save('slic_result.npy', lc.segment_vision(image, cluster_labels))
	candidate_coords, candidate_labels = lc.cluster_centers(cluster_labels)
	#print('Candidate Done. time:{}s' .format(time.time()-start_time))
	
	return candidate_coords, candidate_labels, cluster_labels
	
def nodule_context_slic(image, nodule_coords):
	segmask = cps.segment_lung_mask_fast(image)
	if segmask is None:
		print('Lung segmentation failed')
		#patient_evaluations.write('%d/%d patient %s lung segmentation failed\n' %(p+1, len(test_patients), uid))
		return None
	#print('lung segment. time{}s' .format(time.time()-start_time))
	segmask = cps.extend_mask(segmask)
	cluster_labels = 0 - np.ones(shape=image.shape, dtype=int)
	for ncoord in nodule_coords:
		#for ci in range(len(ncoord)):
		#	if ncoord[ci] < 20:
		#		ncoord[ci] = 20
		#	if ncoord[ci] > image.shape[ci] - 20:
		#		ncoord[ci] = image.shape[ci] - 20
		boundbox = {}
		boundbox["z_bottom"] = max(int(ncoord[0]-20), 0)
		boundbox["z_top"] = min(int(ncoord[0]+20), image.shape[0])
		boundbox["y_bottom"] = max(int(ncoord[1]-20), 0)
		boundbox["y_top"] = min(int(ncoord[1]+20), image.shape[1])
		boundbox["x_bottom"] = max(int(ncoord[2]-20), 0)
		boundbox["x_top"] = min(int(ncoord[2]+20), image.shape[2])
		image_bounded = image[boundbox["z_bottom"]:boundbox["z_top"], boundbox["y_bottom"]:boundbox["y_top"], boundbox["x_bottom"]:boundbox["x_top"]]
		num_segments = int(image_bounded.shape[0] * image_bounded.shape[1] * image_bounded.shape[2] / 15)	#the volume of a 3mm nodule is 27 voxels
		cluster_labels_bounded = lc.slic_segment(image_bounded, compactness=0.01, num_segments=num_segments)
		cluster_labels_bounded += cluster_labels.max() + 1
		cluster_labels[boundbox["z_bottom"]:boundbox["z_top"], boundbox["y_bottom"]:boundbox["y_top"], boundbox["x_bottom"]:boundbox["x_top"]] = cluster_labels_bounded
	cluster_labels[np.logical_or(segmask==0, image<-600)] = -1
	candidate_coords, candidate_labels = lc.cluster_centers(cluster_labels)
	#print('Candidate Done. time:{}s' .format(time.time()-start_time))
	
	return candidate_coords, candidate_labels, cluster_labels
	
def luna_candidate(image, uid, origin, spacing, candidate_file, lung_segment=False, vision_path=None):
	if lung_segment:
		segmask = cps.segment_lung_mask_fast(image)
		if segmask is None: print('Lung segmentation failed')
		else: segmask = cps.extend_mask(segmask)
		if vision_path is not None: np.save(vision_path+'/'+uid+'_segmask.npy', segmask)
	else:
		segmask = None
	candidates = pd.read_csv(candidate_file)
	candlines = (candidates["seriesuid"].values.astype(type(uid))==uid).nonzero()[0]
	candidate_coords = []
	#candidate_coords = np.empty((len(candlines), 3), dtype=int)
	for ci in tqdm(range(len(candlines))):
		candline = candlines[ci]
		zcoord = candidates["coordZ"].values[candline]
		ycoord = candidates["coordY"].values[candline]
		xcoord = candidates["coordX"].values[candline]
		real_coord = np.array([zcoord, ycoord, xcoord], dtype=float)
		#candidate_coords[ci] = np.abs(real_coord-origin) / spacing + 0.5
		ccoord = np.array(np.abs(real_coord-origin) / spacing + 0.5, dtype=int)
		if segmask is None or segmask[ccoord[0]][ccoord[1]][ccoord[2]]: candidate_coords.append(ccoord)
	candidate_coords = np.array(candidate_coords, dtype=int)
	
	return candidate_coords
	
def predictions_map(cluster_labels, predictions, labels):
	result_labels = 0 - np.ones(shape=cluster_labels.shape, dtype=int)
	result_predictions = np.zeros(shape=cluster_labels.shape, dtype=float)
	print("predictions map:")
	for label_prediction_enum in enumerate(tqdm(np.nditer([predictions, labels]))):
		prediction, label = label_prediction_enum[1]
		result_labels[cluster_labels==label] = label
		result_predictions[cluster_labels==label] = prediction
	return result_predictions, result_labels
	
def predictions_map_fast(cluster_labels, predictions, labels):
	result_labels = 0 - np.ones(shape=cluster_labels.shape, dtype=int)
	result_predictions = np.zeros(shape=cluster_labels.shape, dtype=float)
	label_coords = np.where(cluster_labels>=0)
	print("predictions map:")
	for lci in tqdm(range(len(label_coords[0]))):
		coordz = label_coords[0][lci]
		coordy = label_coords[1][lci]
		coordx = label_coords[2][lci]
		labelindices = np.where(labels==cluster_labels[coordz][coordy][coordx])
		if len(labelindices[0])>0:
			labelindex = labelindices[0][0]
			result_labels[coordz][coordy][coordx] = labels[labelindex]
			result_predictions[coordz][coordy][coordx] = predictions[labelindex]
	return result_predictions, result_labels
	
def prediction_cluster(prediction_volume, difference_threshold=0.2, maxclsize=64000, minclsize=10):
	nodule_detections = []
	predorder = prediction_volume.argsort(axis=None)[::-1]
	ordercoords = np.unravel_index(predorder, prediction_volume.shape)
	num_coords = len(np.where(prediction_volume>0)[0])
	cluster_labels = 0 - np.ones_like(prediction_volume, dtype=bool)
	steps = np.int_([[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, 1, 1], [0, 1, -1], [0, -1, 0], [0, -1, 1], [0, -1, -1], [1, 0, 0],
			 [1, 0, 1], [1, 0, -1], [1, 1, 0], [1, -1, 0], [-1, 0, 0], [-1, 0, 1], [-1, 0, -1], [-1, 1, 0], [-1, -1, 0]])
	labelcount = 0
	print("prediction clustering:")
	for oi in tqdm(range(num_coords)):
		maxcoord = np.int_([ordercoords[0][oi], ordercoords[1][oi], ordercoords[2][oi]])
		#if prediction_volume[ordercoords[0][oi], ordercoords[1][oi], ordercoords[2][oi]]<=0:
		#	break
		if cluster_labels[ordercoords[0][oi], ordercoords[1][oi], ordercoords[2][oi]]<0:
			cluster_labels[maxcoord[0], maxcoord[1], maxcoord[2]] = labelcount
			maxpred = prediction_volume[maxcoord[0], maxcoord[1], maxcoord[2]]
			centercoord = np.zeros(3, dtype=float)
			cluster_size = 0
			diffusion_list = [maxcoord]
			while len(diffusion_list) > 0:
				seed = diffusion_list.pop(0)
				centercoord += seed
				cluster_size += 1
				for si in range(len(steps)):
					neighbor = seed + steps[si]
					if not mt.coord_overflow(neighbor, prediction_volume.shape) and cluster_labels[neighbor[0],neighbor[1],neighbor[2]]<0 and prediction_volume[neighbor[0],neighbor[1],neighbor[2]]>0 and abs(maxpred-prediction_volume[neighbor[0],neighbor[1],neighbor[2]])<=difference_threshold:
						cluster_labels[neighbor[0], neighbor[1], neighbor[2]] = labelcount
						diffusion_list.append(neighbor)
			labelcount += 1
			if (maxclsize<=0 or cluster_size<=maxclsize) and (minclsize<=0 or cluster_size>=minclsize):
				centercoord /= cluster_size
				nodule_detections.append([centercoord[0], centercoord[1], centercoord[2], maxpred])
	return nodule_detections, cluster_labels
	
def diameter_calc(prediction_volume, nodule_coords):
	nodule_diameters = []
	cluster_labels = 0 - np.ones_like(prediction_volume, dtype=bool)
	steps = np.int_([[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, 1, 1], [0, 1, -1], [0, -1, 0], [0, -1, 1], [0, -1, -1], [1, 0, 0],
			 [1, 0, 1], [1, 0, -1], [1, 1, 0], [1, -1, 0], [-1, 0, 0], [-1, 0, 1], [-1, 0, -1], [-1, 1, 0], [-1, -1, 0]])
	for oi in tqdm(range(len(nodule_coords))):
		nodulecoord = np.int_([nodule_coords[oi][0], nodule_coords[oi][1], nodule_coords[oi][2]])
		#if prediction_volume[ordercoords[0][oi], ordercoords[1][oi], ordercoords[2][oi]]<=0:
		#	break
		if cluster_labels[nodulecoord[0], nodulecoord[1], nodulecoord[2]] < 0:
			cluster_labels[nodulecoord[0], nodulecoord[1], nodulecoord[2]] = oi
			maxpred = prediction_volume[nodulecoord[0], nodulecoord[1], nodulecoord[2]]
			voxcoords = []
			diffusion_list = [nodulecoord]
			while len(diffusion_list) > 0:
				seed = diffusion_list.pop(0)
				voxcoords.append(seed)
				for si in range(len(steps)):
					neighbor = seed + steps[si]
					if not mt.coord_overflow(neighbor, prediction_volume.shape) and cluster_labels[neighbor[0],neighbor[1],neighbor[2]]<0 and prediction_volume[neighbor[0],neighbor[1],neighbor[2]]>0:
						cluster_labels[neighbor[0], neighbor[1], neighbor[2]] = oi
						diffusion_list.append(neighbor)
			
			maxsqdist = 0
			for vc in range(len(voxcoords)):
				for vcc in range(vc+1, len(voxcoords)):
					diff = voxcoords[vc] - voxcoords[vcc]
					sqdist = (diff*diff).sum()
					if maxsqdist < sqdist:
						maxsqdist = sqdist
			maxdist = math.sqrt(maxsqdist)
			nodule_diameters.append(maxdist)
		else:
			print("nodule {} fused with others" .format(nodule_coords[oi][:3]))
	return nodule_diameters
	
def prediction_centering(prediction_volume, prediction_threshold=0.5, maxclsize=-1, minclsize=10):
	nodule_detections = []
	nodulespixlabels = prediction_volume > prediction_threshold
	connectedlabels = measure.label(nodulespixlabels, connectivity=2)
	backgroundlabel = connectedlabels[0, 0, 0]
	# num_labels = connectedlabels.max() + 1
	labels = np.unique(connectedlabels)
	maxlabel = labels.max()
	print("prediction centering:")
	for label in enumerate(tqdm(labels)):
		#print('combination process:%d/%d' % (label, maxlabel))
		label = label[1]
		if label != backgroundlabel:
			prediction = prediction_volume[connectedlabels == label].max()
			coords = (connectedlabels == label).nonzero()
			clsize = len(coords[0])
			if clsize == 0 or (maxclsize > 0 and clsize > maxclsize) or (
					minclsize >= 0 and clsize < minclsize):
				continue
			#nodule_center = [int(coords[0].mean() + 0.5), int(coords[1].mean() + 0.5), int(coords[2].mean() + 0.5)]
			nodule_detections.append([coords[0].mean(), coords[1].mean(), coords[2].mean(), prediction])
	return nodule_detections

def prediction_centering_fast(prediction_volume, maxclsize=-1, minclsize=10):
	nodule_detections = []
	nodulespixlabels = prediction_volume > 0
	connectedlabels = measure.label(nodulespixlabels, connectivity=2)
	backgroundlabel = connectedlabels[0, 0, 0]
	labels, lcounts = np.unique(connectedlabels, return_counts=True)
	num_labels = labels.max() + 1
	centersz = np.zeros(num_labels, dtype=float)
	centersy = np.zeros(num_labels, dtype=float)
	centersx = np.zeros(num_labels, dtype=float)
	centerspred = np.zeros(num_labels, dtype=float)
	lnums = np.zeros(num_labels, dtype=int)
	prediction_coords = np.where(connectedlabels!=backgroundlabel)
	print("prediction centering:")
	for pci in tqdm(range(prediction_coords[0].size)):
		coordz = prediction_coords[0][pci]
		coordy = prediction_coords[1][pci]
		coordx = prediction_coords[2][pci]
		label  = connectedlabels[coordz][coordy][coordx]
		if lcounts[label] == 0 or (maxclsize >= 0 and lcounts[label] > maxclsize) or (minclsize >= 0 and lcounts[label] < minclsize):
			continue
		centersz[label] += coordz
		centersy[label] += coordy
		centersx[label] += coordx
		lnums[label] += 1
		if centerspred[label] < prediction_volume[coordz][coordy][coordx]:
			centerspred[label] = prediction_volume[coordz][coordy][coordx]
	labels_valid = lnums > 0
	cnums = lnums[labels_valid]
	nodule_detections = np.transpose(np.array([centersz[labels_valid]/cnums, centersy[labels_valid]/cnums, centersx[labels_valid]/cnums, centerspred[labels_valid]]))
	return nodule_detections
	
def prediction_combine(coords, predictions, distance_threshold=3):
	indsets = []
	for ci in range(len(coords)):
		findneighbor = False
		for indset in indsets:
			for ind in indset:
				step = coords[ci] - coords[ind]
				distancesquare = (step*step).sum()
				if distancesquare <= distance_threshold * distance_threshold:
					indset.append(ci)
					findneighbor = True
					break
			if findneighbor:
				break
		if not findneighbor:
			indsets.append([ci])
	centers = np.zeros((len(indsets), 3), dtype=float)
	cpreds = np.zeros((len(indsets), 1), dtype=float)
	for csi in range(len(indsets)):
		for ind in indsets[csi]:
			centers[csi] = centers[csi] + coords[ind]
			if cpreds[csi] < predictions[ind]:
				cpreds[csi] = predictions[ind]
		centers[csi] = centers[csi] / len(indsets[csi])
	nodule_center_predictions = np.concatenate((centers, cpreds), axis=1)
	return nodule_center_predictions
	
def precise_detection_pt(volume, region_size, candidate_coords, model, resize_factor=None, candidate_batch=10, use_gpu=True, prediction_threshold=0.5):
	#model.eval()
	data_shape = np.array([region_size, region_size, region_size], dtype=int)
	data_prehalf = np.int_(data_shape/2)
	if resize_factor is None:
		region_shape = np.array([region_size, region_size, region_size])
	else:
		region_shape = np.ceil(np.array([region_size, region_size, region_size])/np.array(resize_factor)).astype(int)
	region_prehalf = np.int_(region_shape/2)
	volume_padded = MIN_BOUND * np.ones((volume.shape[0]+region_shape[0], volume.shape[1]+region_shape[1], volume.shape[2]+region_shape[2]), dtype=int)
	volume_padded[region_prehalf[0]:region_prehalf[0]+volume.shape[0], region_prehalf[1]:region_prehalf[1]+volume.shape[1], region_prehalf[2]:region_prehalf[2]+volume.shape[2]] = volume
	test_data = np.zeros(shape=(candidate_batch, data_shape[0], data_shape[1], data_shape[2]), dtype=float)
	if isinstance(model, torch.nn.Module):
		candidate_predictions = np.zeros(len(candidate_coords), dtype=float)
	else:
		candidate_predictions = [np.zeros(len(candidate_coords), dtype=float) for m in range(len(model))]

	for cb in tqdm(range(0, len(candidate_coords), candidate_batch)):
		candbatchend = min(cb+candidate_batch, len(candidate_coords))
		for cc in range(cb, candbatchend):
			coord = candidate_coords[cc]
			local_region = volume_padded[coord[0]:coord[0]+region_shape[0], coord[1]:coord[1]+region_shape[1], coord[2]:coord[2]+region_shape[2]]
			if resize_factor is not None:
				local_region_resized = zoom(local_region, resize_factor, mode='nearest')
				resized_center = np.int_(np.array(local_region_resized.shape)/2)
				crop_bottom = resized_center - data_prehalf
				crop_top = resized_center + data_shape - data_prehalf
				local_region = local_region_resized[crop_bottom[0]:crop_top[0], crop_bottom[1]:crop_top[1], crop_bottom[2]:crop_top[2]]
			test_data[cc-cb] = medical_normalization(local_region)
		test_data_clipped = test_data[:candbatchend-cb]
		test_data_reshaped = test_data_clipped.reshape((candbatchend-cb, 1, region_size, region_size, region_size))
		model_input = torch.autograd.Variable(torch.from_numpy(test_data_reshaped).float())
		if use_gpu: model_input = model_input.cuda()
		if isinstance(model, torch.nn.Module):
			net_outs = model(model_input)
			#predictions = torch.nn.functional.softmax(net_outs, dim=1).data.cpu().numpy()
			predictions = torch.nn.functional.softmax(net_outs, dim=1).data[:, 1]
			if use_gpu: predictions = predictions.cpu()
			predictions = predictions.numpy()
			candidate_predictions[cb:candbatchend][predictions>prediction_threshold] = predictions[predictions>prediction_threshold]
			'''
			for p in range(len(predictions)):
				if predictions[p][1]>prediction_threshold:
					candidate_predictions[cb+p] = predictions[p][1]
			'''
		else:
			for m in range(len(model)):
				net_outs = model[m](model_input)
				predictions = torch.nn.functional.softmax(net_outs, dim=1).data[:, 1]
				if use_gpu: predictions = predictions.cpu()
				predictions = predictions.numpy()
				candidate_predictions[m][cb:candbatchend][predictions>prediction_threshold] = predictions[predictions>prediction_threshold]

	return candidate_predictions
	
def precise_detection_tf(volume, region_size, candidate_coords, sess, input_tensor, output_tensor, candidate_batch=10, prediction_threshold=0.5):
	data_shape = np.array([region_size, region_size, region_size], dtype=int)
	region_shape = np.array([region_size, region_size, region_size], dtype=int)
	region_prehalf = np.int_(region_shape/2)
	volume_padded = MIN_BOUND * np.ones((volume.shape[0]+region_shape[0], volume.shape[1]+region_shape[1], volume.shape[2]+region_shape[2]), dtype=int)
	volume_padded[region_prehalf[0]:region_prehalf[0]+volume.shape[0], region_prehalf[1]:region_prehalf[1]+volume.shape[1], region_prehalf[2]:region_prehalf[2]+volume.shape[2]] = volume
	test_data = np.zeros(shape=(candidate_batch, data_shape[0], data_shape[1], data_shape[2]), dtype=float)
	candidate_predictions = np.zeros(len(candidate_coords), dtype=float)
	#nodule_centers = []

	#predictions_output = open('detection_vision/candidates2/predictions.txt', 'w')
	for cb in tqdm(range(0, len(candidate_coords), candidate_batch)):
		candbatchend = min(cb+candidate_batch, len(candidate_coords))
		for cc in range(cb, candbatchend):
			coord = candidate_coords[cc]
			local_region = volume_padded[coord[0]:coord[0]+region_shape[0], coord[1]:coord[1]+region_shape[1], coord[2]:coord[2]+region_shape[2]]
			#np.save('detection_vision/candidates/region'+str(cc)+'.npy', local_region)
			#if not mt.region_valid(local_region):
			#	continue
			test_data[cc-cb] = medical_normalization(local_region)
		predictions = sess.run(output_tensor, feed_dict={input_tensor:test_data[:candbatchend-cb]})
		#predictions = np.random.rand(test_data.shape[0], 1)
		#predictions = np.concatenate((predictions, 1 - predictions), axis=1)
		for p in range(len(predictions)):
			#pdata = test_data[p]
			#np.save('detection_vision/candidates2/region_'+str(predictions[p][0])+'.npy', pdata)
			#predictions_output.write('%f\n' %(predictions[p][0]))
			if predictions[p][0]>prediction_threshold:
				candidate_predictions[cb+p] = predictions[p][0]
	#predictions_output.close()

	return candidate_predictions
	
def precise_detection_multilevel(volume, region_sizes, candidate_coords, sess, input_tensors, output_tensor, candidate_batch=10, augmentation=False, prediction_threshold=0.5):
	max_region_size = max(region_sizes)
	max_region_prehalf = int(max_region_size/2)
	region_shapes = np.array([[region_sizes[0], region_sizes[0], region_sizes[0]],
				[region_sizes[1], region_sizes[1], region_sizes[1]],
				[region_sizes[2], region_sizes[2], region_sizes[2]]], dtype=int)
	region_prehalfs = region_shapes / 2
	volume_padded = MIN_BOUND * np.ones((volume.shape[0]+max_region_size, volume.shape[1]+max_region_size, volume.shape[2]+max_region_size), dtype=int)
	volume_padded[max_region_prehalf:max_region_prehalf+volume.shape[0], max_region_prehalf:max_region_prehalf+volume.shape[1], max_region_prehalf:max_region_prehalf+volume.shape[2]] = volume
	aug_proportion = int(augmentation) * 12 + 1
	cand_batch_size = int(candidate_batch/float(aug_proportion)+0.5)
	data_batch_size = cand_batch_size * aug_proportion
	test_datas = [np.zeros(shape=(data_batch_size, region_shapes[0][0], region_shapes[0][1], region_shapes[0][2]), dtype=float),
		     np.zeros(shape=(data_batch_size, region_shapes[1][0], region_shapes[1][1], region_shapes[1][2]), dtype=float),
		     np.zeros(shape=(data_batch_size, region_shapes[2][0], region_shapes[2][1], region_shapes[2][2]), dtype=float)]
	candidate_predictions = np.zeros(len(candidate_coords), dtype=float)
	#nodule_centers = []

	#predictions_output = open('detection_vision/candidates2/predictions.txt', 'w')
	for cb in tqdm(range(0, len(candidate_coords), cand_batch_size)):
		candbatchend = min(cb+cand_batch_size, len(candidate_coords))
		for cc in range(cb, candbatchend):
			coord = np.int_(candidate_coords[cc])
			coord_bottoms = (coord-region_prehalfs[0]+max_region_prehalf, coord-region_prehalfs[1]+max_region_prehalf, coord-region_prehalfs[2]+max_region_prehalf)
			coord_tops = (coord-region_prehalfs[0]+region_shapes[0]+max_region_prehalf, coord-region_prehalfs[1]+region_shapes[1]+max_region_prehalf, coord-region_prehalfs[2]+region_shapes[2]+max_region_prehalf)
			local_regions = (volume_padded[coord_bottoms[0][0]:coord_tops[0][0], coord_bottoms[0][1]:coord_tops[0][1], coord_bottoms[0][2]:coord_tops[0][2]],
					 volume_padded[coord_bottoms[1][0]:coord_tops[1][0], coord_bottoms[1][1]:coord_tops[1][1], coord_bottoms[1][2]:coord_tops[1][2]],
					 volume_padded[coord_bottoms[2][0]:coord_tops[2][0], coord_bottoms[2][1]:coord_tops[2][1], coord_bottoms[2][2]:coord_tops[2][2]])
			if augmentation:
				local_regions = (mt.extract_volumes(local_regions[0], region_shapes[0], rotation_augment=True, flip_augment=True),
						 mt.extract_volumes(local_regions[1], region_shapes[1], rotation_augment=True, flip_augment=True),
						 mt.extract_volumes(local_regions[2], region_shapes[2], rotation_augment=True, flip_augment=True))
			tdind = cc - cb
			test_datas[0][tdind*aug_proportion:(tdind+1)*aug_proportion] = medical_normalization(local_regions[0])
			test_datas[1][tdind*aug_proportion:(tdind+1)*aug_proportion] = medical_normalization(local_regions[1])
			test_datas[2][tdind*aug_proportion:(tdind+1)*aug_proportion] = medical_normalization(local_regions[2])
		predictions = sess.run(output_tensor, feed_dict={input_tensors[0]:test_datas[0][:(candbatchend-cb)*aug_proportion], input_tensors[1]:test_datas[1][:(candbatchend-cb)*aug_proportion], input_tensors[2]:test_datas[2][:(candbatchend-cb)*aug_proportion]})
		for pb in range(candbatchend-cb):
			#pdata = test_data[p]
			#np.save('detection_vision/candidates2/region_'+str(predictions[p][0])+'.npy', pdata)
			#predictions_output.write('%f\n' %(predictions[p][0]))
			prediction = predictions[pb*aug_proportion:(pb+1)*aug_proportion, 0].mean()
			if prediction>prediction_threshold:
				candidate_predictions[cb+pb] = prediction
	#predictions_output.close()

	return candidate_predictions
	
def precise_detection_multilevel_old(volume, region_sizes, candidate_coords, sess, input_tensors, output_tensor, candidate_batch=10, prediction_threshold=0.5):
	max_region_size = max(region_sizes)
	max_region_prehalf = int(max_region_size/2)
	region_shapes = np.array([[region_sizes[0], region_sizes[0], region_sizes[0]],
				[region_sizes[1], region_sizes[1], region_sizes[1]],
				[region_sizes[2], region_sizes[2], region_sizes[2]]], dtype=int)
	region_prehalfs = region_shapes / 2
	volume_padded = MIN_BOUND * np.ones((volume.shape[0]+max_region_size, volume.shape[1]+max_region_size, volume.shape[2]+max_region_size), dtype=int)
	volume_padded[max_region_prehalf:max_region_prehalf+volume.shape[0], max_region_prehalf:max_region_prehalf+volume.shape[1], max_region_prehalf:max_region_prehalf+volume.shape[2]] = volume
	test_datas = [np.zeros(shape=(candidate_batch, region_shapes[0][0], region_shapes[0][1], region_shapes[0][2]), dtype=float),
		     np.zeros(shape=(candidate_batch, region_shapes[1][0], region_shapes[1][1], region_shapes[1][2]), dtype=float),
		     np.zeros(shape=(candidate_batch, region_shapes[2][0], region_shapes[2][1], region_shapes[2][2]), dtype=float)]
	candidate_predictions = np.zeros(len(candidate_coords), dtype=float)
	#nodule_centers = []

	#predictions_output = open('detection_vision/candidates2/predictions.txt', 'w')
	for cb in tqdm(range(0, len(candidate_coords), candidate_batch)):
		candbatchend = min(cb+candidate_batch, len(candidate_coords))
		for cc in range(cb, candbatchend):
			coord = np.int_(candidate_coords[cc])
			coord_bottoms = (coord-region_prehalfs[0]+max_region_prehalf, coord-region_prehalfs[1]+max_region_prehalf, coord-region_prehalfs[2]+max_region_prehalf)
			coord_tops = (coord-region_prehalfs[0]+region_shapes[0]+max_region_prehalf, coord-region_prehalfs[1]+region_shapes[1]+max_region_prehalf, coord-region_prehalfs[2]+region_shapes[2]+max_region_prehalf)
			local_regions = (volume_padded[coord_bottoms[0][0]:coord_tops[0][0], coord_bottoms[0][1]:coord_tops[0][1], coord_bottoms[0][2]:coord_tops[0][2]],
					 volume_padded[coord_bottoms[1][0]:coord_tops[1][0], coord_bottoms[1][1]:coord_tops[1][1], coord_bottoms[1][2]:coord_tops[1][2]],
					 volume_padded[coord_bottoms[2][0]:coord_tops[2][0], coord_bottoms[2][1]:coord_tops[2][1], coord_bottoms[2][2]:coord_tops[2][2]])
			test_datas[0][cc-cb] = medical_normalization(local_regions[0])
			test_datas[1][cc-cb] = medical_normalization(local_regions[1])
			test_datas[2][cc-cb] = medical_normalization(local_regions[2])
		predictions = sess.run(output_tensor, feed_dict={input_tensors[0]:test_datas[0][:candbatchend-cb], input_tensors[1]:test_datas[1][:candbatchend-cb], input_tensors[2]:test_datas[2][:candbatchend-cb]})
		for p in range(len(predictions)):
			#pdata = test_data[p]
			#np.save('detection_vision/candidates2/region_'+str(predictions[p][0])+'.npy', pdata)
			#predictions_output.write('%f\n' %(predictions[p][0]))
			if predictions[p][0]>prediction_threshold:
				candidate_predictions[cb+p] = predictions[p][0]
	#predictions_output.close()

	return candidate_predictions

print("program config:")	
print("max bound:{}" .format(MAX_BOUND))
print("min bound:{}" .format(MIN_BOUND))
print("pixel mean:{}" .format(PIXEL_MEAN))
print("normalization crop:{}" .format(NORM_CROP))
