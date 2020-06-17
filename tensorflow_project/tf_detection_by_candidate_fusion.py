#!/usr/bin/env python
# encoding: utf-8

import os
import copy
import time
import shutil
import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
#import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import measure
from glob import glob
from toolbox import BasicTools as bt
from toolbox import MITools as mt
from toolbox import CTViewer_Multiax as cvm
from toolbox import Lung_Pattern_Segmentation as lps
from toolbox import Lung_Cluster as lc
from toolbox import Nodule_Detection as nd
from toolbox import TensorflowTools as tft
from toolbox import Evaluations as eva
try:
	from tqdm import tqdm # long waits are not fun
except:
	print('tqdm 是一个轻量级的进度条小包。。。')
	tqdm = lambda x : x

'''
ENVIRONMENT_FILE = "./constants.txt"
IMG_WIDTH, IMG_HEIGHT, NUM_VIEW, MAX_BOUND, MIN_BOUND, PIXEL_MEAN = mt.read_environment(ENVIRONMENT_FILE)
WINDOW_SIZE = min(IMG_WIDTH, IMG_HEIGHT)
NUM_CHANNELS = 3
'''
constants = bt.read_constants("./constants.txt")
#REGION_SIZE = constants["REGION_SIZE"]
MAX_BOUND = float(constants["MAX_BOUND"])
MIN_BOUND = float(constants["MIN_BOUND"])
PIXEL_MEAN = constants["PIXEL_MEAN"]
REGION_SIZES = (20, 30, 40)
CANDIDATE_BATCH = 15
RESULT_VISION = False
FUSION_MODE = 'committe'

def precise_detection(volume, candidate_coords, candidate_labels, sess, input_tensors, output_tensor):
	max_region_size = max(REGION_SIZES)
	max_region_prehalf = int(max_region_size/2)
	region_sizes = np.array([[REGION_SIZES[0], REGION_SIZES[0], REGION_SIZES[0]],
				[REGION_SIZES[1], REGION_SIZES[1], REGION_SIZES[1]],
				[REGION_SIZES[2], REGION_SIZES[2], REGION_SIZES[2]]], dtype=int)
	region_prehalfs = region_sizes / 2
	volume_padded = MAX_BOUND * np.ones((volume.shape[0]+max_region_size, volume.shape[1]+max_region_size, volume.shape[2]+max_region_size), dtype=int)
	volume_padded[max_region_prehalf:max_region_prehalf+volume.shape[0], max_region_prehalf:max_region_prehalf+volume.shape[1], max_region_prehalf:max_region_prehalf+volume.shape[2]] = volume
	test_datas = [np.zeros(shape=(CANDIDATE_BATCH, region_sizes[0][0], region_sizes[0][1], region_sizes[0][2]), dtype=float),
		     np.zeros(shape=(CANDIDATE_BATCH, region_sizes[1][0], region_sizes[1][1], region_sizes[1][2]), dtype=float),
		     np.zeros(shape=(CANDIDATE_BATCH, region_sizes[2][0], region_sizes[2][1], region_sizes[2][2]), dtype=float)]
	test_labels = []
	label_predictions = []
	#nodule_centers = []
	batch_index = 0

	#predictions_output = open('detection_vision/candidates2/predictions.txt', 'w')
	for cc in tqdm(range(len(candidate_coords))):
		coord = np.array(candidate_coords[cc], dtype=int)
		coord_bottoms = (coord-region_prehalfs[0]+max_region_prehalf, coord-region_prehalfs[1]+max_region_prehalf, coord-region_prehalfs[2]+max_region_prehalf)
		coord_tops = (coord-region_prehalfs[0]+region_sizes[0]+max_region_prehalf, coord-region_prehalfs[1]+region_sizes[1]+max_region_prehalf, coord-region_prehalfs[2]+region_sizes[2]+max_region_prehalf)
		local_regions = (volume_padded[coord_bottoms[0][0]:coord_tops[0][0], coord_bottoms[0][1]:coord_tops[0][1], coord_bottoms[0][2]:coord_tops[0][2]],
				 volume_padded[coord_bottoms[1][0]:coord_tops[1][0], coord_bottoms[1][1]:coord_tops[1][1], coord_bottoms[1][2]:coord_tops[1][2]],
				 volume_padded[coord_bottoms[2][0]:coord_tops[2][0], coord_bottoms[2][1]:coord_tops[2][1], coord_bottoms[2][2]:coord_tops[2][2]])
				 
		#np.save('detection_vision/candidates/region'+str(cc)+'.npy', local_region)
		#if not mt.region_valid(local_region):
		#	continue
		test_datas[0][batch_index] = (local_regions[0] - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
		test_datas[1][batch_index] = (local_regions[1] - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
		test_datas[2][batch_index] = (local_regions[2] - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
		test_labels.append(candidate_labels[cc])
		batch_index += 1
		if batch_index==CANDIDATE_BATCH:
			predictions = sess.run(output_tensor, feed_dict={input_tensors[0]:test_datas[0], input_tensors[1]:test_datas[1], input_tensors[2]:test_datas[2]})
			#predictions = np.random.rand(test_data.shape[0], 1)
			#predictions = np.concatenate((predictions, 1 - predictions), axis=1)
			for p in range(predictions.shape[0]):
				#pdata = test_data[p]
				#np.save('detection_vision/candidates2/region_'+str(predictions[p][0])+'.npy', pdata)
				#predictions_output.write('%f\n' %(predictions[p][0]))
				if predictions[p][0]>predictions[p][1]:
					label_predictions.append([test_labels[p], predictions[p][0]])
					#nodule_centers.append([candidate_coords[p][0], candidate_coords[p][1], candidate_coords[p][2], predictions[p][0]])
			test_labels = []
			batch_index = 0
	if batch_index>0:
		test_datas[0] = test_datas[0][:batch_index]
		test_datas[1] = test_datas[1][:batch_index]
		test_datas[2] = test_datas[2][:batch_index]
		predictions = sess.run(output_tensor, feed_dict={input_tensors[0]:test_datas[0], input_tensors[1]:test_datas[1], input_tensors[2]:test_datas[2]})
		#predictions = np.random.rand(test_data.shape[0], 1)
		#predictions = np.concatenate((predictions,1-predictions), axis=1)
		for p in range(predictions.shape[0]):
			#pdata = test_data[p]
			#np.save('detection_vision/candidates2/region_'+str(predictions[p][0])+'.npy', pdata)
			#predictions_output.write('%f\n' %(predictions[p][0]))
			if predictions[p][0]>predictions[p][1]:
				label_predictions.append([test_labels[p], predictions[p][0]])
				#nodule_centers.append([candidate_coords[p][0], candidate_coords[p][1], candidate_coords[p][2], predictions[p][0]])
	#predictions_output.close()

	#return the centers of detected nodules and the labels corresponding to the pixel cluster in this nodule
	#return nodule_centers, label_predictions
	return label_predictions
	
def precise_detection_with_labels(volume, cluster_labels, sess, input_tensors, output_tensor):
	max_region_size = max(REGION_SIZES)
	max_region_prehalf = int(max_region_size/2)
	region_sizes = np.array([[REGION_SIZES[0], REGION_SIZES[0], REGION_SIZES[0]],
				[REGION_SIZES[1], REGION_SIZES[1], REGION_SIZES[1]],
				[REGION_SIZES[2], REGION_SIZES[2], REGION_SIZES[2]]], dtype=int)
	region_prehalfs = region_sizes / 2
	volume_padded = MAX_BOUND * np.ones((volume.shape[0]+max_region_size, volume.shape[1]+max_region_size, volume.shape[2]+max_region_size), dtype=int)
	volume_padded[max_region_prehalf:max_region_prehalf+volume.shape[0], max_region_prehalf:max_region_prehalf+volume.shape[1], max_region_prehalf:max_region_prehalf+volume.shape[2]] = volume
	test_datas = [np.zeros(shape=(CANDIDATE_BATCH, region_sizes[0][0], region_sizes[0][1], region_sizes[0][2]), dtype=float),
		     np.zeros(shape=(CANDIDATE_BATCH, region_sizes[1][0], region_sizes[1][1], region_sizes[1][2]), dtype=float),
		     np.zeros(shape=(CANDIDATE_BATCH, region_sizes[2][0], region_sizes[2][1], region_sizes[2][2]), dtype=float)]
	test_labels = []
	label_predictions = []
	# nodule_centers = []
	batch_index = 0

	# predictions_output = open('detection_vision/candidates2/predictions.txt', 'w')
	labels = np.unique(cluster_labels)
	if labels[0]<0:
		labels = np.delete(labels, 0)
	print('num labels:%d' %(len(labels)))
	for label in enumerate(tqdm(labels)):
		label = label[1]
		coords = np.where(cluster_labels==label)
		coord = np.int_([coords[0].mean()+0.5, coords[1].mean()+0.5, coords[2].mean()+0.5])
		coord_bottoms = (coord-region_prehalfs[0]+max_region_prehalf, coord-region_prehalfs[1]+max_region_prehalf, coord-region_prehalfs[2]+max_region_prehalf)
		coord_tops = (coord-region_prehalfs[0]+region_sizes[0]+max_region_prehalf, coord-region_prehalfs[1]+region_sizes[1]+max_region_prehalf, coord-region_prehalfs[2]+region_sizes[2]+max_region_prehalf)
		local_regions = (volume_padded[coord_bottoms[0][0]:coord_tops[0][0], coord_bottoms[0][1]:coord_tops[0][1], coord_bottoms[0][2]:coord_tops[0][2]],
				 volume_padded[coord_bottoms[1][0]:coord_tops[1][0], coord_bottoms[1][1]:coord_tops[1][1], coord_bottoms[1][2]:coord_tops[1][2]],
				 volume_padded[coord_bottoms[2][0]:coord_tops[2][0], coord_bottoms[2][1]:coord_tops[2][1], coord_bottoms[2][2]:coord_tops[2][2]])
		# np.save('detection_vision/candidates/region'+str(cc)+'.npy', local_region)
		# if not mt.region_valid(local_region):
		#	continue
		test_datas[0][batch_index] = (local_regions[0] - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
		test_datas[1][batch_index] = (local_regions[1] - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
		test_datas[2][batch_index] = (local_regions[2] - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
		test_labels.append(label)
		batch_index += 1
		if batch_index == CANDIDATE_BATCH:
			predictions = sess.run(output_tensor, feed_dict={input_tensors[0]: test_datas[0], input_tensors[1]: test_datas[1], input_tensors[2]: test_datas[2]})
			# predictions = np.random.rand(test_data.shape[0], 1)
			# predictions = np.concatenate((predictions, 1 - predictions), axis=1)
			for p in range(predictions.shape[0]):
				#pdata = test_data[p]
				# np.save('detection_vision/candidates2/region_'+str(predictions[p][0])+'.npy', pdata)
				# predictions_output.write('%f\n' %(predictions[p][0]))
				if predictions[p][0] > predictions[p][1]:
					label_predictions.append([test_labels[p], predictions[p][0]])
				# nodule_centers.append([candidate_coords[p][0], candidate_coords[p][1], candidate_coords[p][2], predictions[p][0]])
			test_labels = []
			batch_index = 0
	if batch_index>0:
		test_datas[0] = test_datas[0][:batch_index]
		test_datas[1] = test_datas[1][:batch_index]
		test_datas[2] = test_datas[2][:batch_index]
		predictions = sess.run(output_tensor, feed_dict={input_tensors[0]: test_datas[0], input_tensors[1]: test_datas[1], input_tensors[2]: test_datas[2]})
		# predictions = np.random.rand(test_data.shape[0], 1)
		# predictions = np.concatenate((predictions,1-predictions), axis=1)
		for p in range(predictions.shape[0]):
			#pdata = test_data[p]
			# np.save('detection_vision/candidates2/region_'+str(predictions[p][0])+'.npy', pdata)
			# predictions_output.write('%f\n' %(predictions[p][0]))
			if predictions[p][0] > predictions[p][1]:
				label_predictions.append([test_labels[p], predictions[p][0]])
			# nodule_centers.append([candidate_coords[p][0], candidate_coords[p][1], candidate_coords[p][2], predictions[p][0]])
	# predictions_output.close()

	# return the centers of detected nodules and the labels corresponding to the pixel cluster in this nodule
	# return nodule_centers, label_predictions
	return label_predictions

if __name__ == "__main__":
	test_paths = ["./LUNA16/subset9"]
	#test_filelist = './models_tensorflow/luna_tianchi_slh_3D_l3454-512-2_bn2_stage3/pfilelist.log'
	net_files = ["models_tensorflow/luna_slh_3D_bndo_flbias_l5_20_aug_stage3/epoch20/epoch20",
		     "models_tensorflow/luna_slh_3D_bndo_flbias_l5_30_aug2_stage2/epoch10/epoch10", 
		     "models_tensorflow/luna_slh_3D_bndo_flbias_l6_40_aug_stage2/epoch28/epoch28"]
	fusion_file = "models_tensorflow/luna_slh_3D_fusion/epoch6/epoch6"
	bn_files = ["models_tensorflow/luna_slh_3D_bndo_flbias_l5_20_aug_stage3/batch_normalization_statistic.npy",
		   "models_tensorflow/luna_slh_3D_bndo_flbias_l5_30_aug2/batch_normalization_statistic.npy",
		   "models_tensorflow/luna_slh_3D_bndo_flbias_l6_40_aug_stage2/batch_normalization_statistic1.npy"]
	annotation_file = "LUNA16/csvfiles/annotations.csv"
	candidate_file = "LUNA16/csvfiles/candidates.csv"
	vision_path = "./detection_vision/test"
	result_path = "./results"
	evaluation_path = result_path + "/experiments2/evaluation_" + FUSION_MODE + "fusion_test"
	result_file = evaluation_path + "/result.csv"

	if "test_paths" in dir():
		all_patients = []
		for path in test_paths:
			all_patients += glob(path + "/*.mhd")
		if len(all_patients)<=0:
			print("No patient found")
			exit()
	elif "test_filelist" in dir():
		validation_rate = 0.2
		pfilelist_file = open(test_filelist, "r")
		pfiles = pfilelist_file.readlines()
		for pfi in range(len(pfiles)):
			pfiles[pfi] = pfiles[pfi][:-1]
		pfilelist_file.close()
		validation_num = int(len(pfiles)*validation_rate)
		val_files = pfiles[-validation_num:]
	else:
		print("No test data")
		exit()
	if not os.access(vision_path, os.F_OK):
		os.makedirs(vision_path)
	if os.access(evaluation_path, os.F_OK):
		shutil.rmtree(evaluation_path)
	os.makedirs(evaluation_path)

	inputs = [tf.placeholder(tf.float32, [None, REGION_SIZES[0], REGION_SIZES[0], REGION_SIZES[0]]),
		  tf.placeholder(tf.float32, [None, REGION_SIZES[1], REGION_SIZES[1], REGION_SIZES[1]]),
		  tf.placeholder(tf.float32, [None, REGION_SIZES[2], REGION_SIZES[2], REGION_SIZES[2]])]
	inputs_reshape = [tf.reshape(inputs[0], [-1, REGION_SIZES[0], REGION_SIZES[0], REGION_SIZES[0], 1]),
			  tf.reshape(inputs[1], [-1, REGION_SIZES[1], REGION_SIZES[1], REGION_SIZES[1], 1]),
			  tf.reshape(inputs[2], [-1, REGION_SIZES[2], REGION_SIZES[2], REGION_SIZES[2], 1])]
	if "bn_files" in dir():
		bn_params = [np.load(bn_files[0]), np.load(bn_files[1]), np.load(bn_files[2])]
	else:
		bn_params = [None, None, None]
	outputs0, variables0, _ = tft.volume_bndo_flbias_l5_20(inputs_reshape[0], dropout_rate=0.0, batch_normalization_statistic=False, bn_params=bn_params[0])
	outputs1, variables1, _ = tft.volume_bndo_flbias_l5_30(inputs_reshape[1], dropout_rate=0.0, batch_normalization_statistic=False, bn_params=bn_params[1])
	outputs2, variables2, _ = tft.volume_bndo_flbias_l6_40(inputs_reshape[2], dropout_rate=0.0, batch_normalization_statistic=False, bn_params=bn_params[2])
	if FUSION_MODE == 'vote':
		predictions = [outputs0['sm_out'], outputs1['sm_out'], outputs2['sm_out']]
		combined_prediction = tft.vote_fusion(predictions)
		combined_prediction = tf.reshape(combined_prediction, [-1,1])
	elif FUSION_MODE == 'committe':
		predictions = [outputs0['sm_out'], outputs1['sm_out'], outputs2['sm_out']]
		combined_prediction = tft.committe_fusion(predictions)
	elif FUSION_MODE == 'late':
		features = [outputs0['flattened_out'], outputs1['flattened_out'], outputs2['flattened_out']]
		_, combined_prediction, variables_fusion = tft.late_fusion(features, False)
	else:
		print("unknown fusion mode")
		exit()
	
	saver0 = tf.train.Saver(variables0)
	saver1 = tf.train.Saver(variables1)
	saver2 = tf.train.Saver(variables2)
	if FUSION_MODE == 'late':
		saver_fusion = tf.train.Saver(variables_fusion)
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	saver0.restore(sess, net_files[0])
	saver1.restore(sess, net_files[1])
	saver2.restore(sess, net_files[2])
	if FUSION_MODE == 'late':
		saver_fusion.restore(sess, fusion_file)

	#ktb.set_session(mt.get_session(0.5))
	start_time = time.time()
	#patient_evaluations = open(evaluation_path + "/patient_evaluations.log", "w")
	results = []
	CPMs = []
	CPMs2 = []
	test_patients = all_patients[:1]
	bt.filelist_store(all_patients, evaluation_path + "/patientfilelist.log")
	#random.shuffle(test_patients)
	for p in range(len(test_patients)):
		result = []
		patient = test_patients[p]
		#patient = "./LUNA16/subset9/1.3.6.1.4.1.14519.5.2.1.6279.6001.227968442353440630355230778531.mhd"
		patient = "./LUNA16/subset9/1.3.6.1.4.1.14519.5.2.1.6279.6001.102681962408431413578140925249.mhd"
		#patient = "./TIANCHI_examples/LKDS-00005.mhd"
		uid = mt.get_serie_uid(patient)
		annotations = mt.get_annotations(uid, annotation_file)
		if len(annotations)==0:
			print('%d/%d patient %s has no annotations, ignore it.' %(p+1, len(test_patients), uid))
			#patient_evaluations.write('%d/%d patient %s has no annotations, ignore it\n' %(p+1, len(test_patients), uid))
			continue

		print('%d/%d processing patient:%s' %(p+1, len(test_patients), uid))
		full_image_info = sitk.ReadImage(patient)
		full_scan = sitk.GetArrayFromImage(full_image_info)
		origin = np.array(full_image_info.GetOrigin())[::-1]	#the order of origin and old_spacing is initially [z,y,x]
		old_spacing = np.array(full_image_info.GetSpacing())[::-1]
		image, new_spacing = mt.resample(full_scan, old_spacing)	#resample
		print('Resample Done. time:{}s' .format(time.time()-start_time))

		#make a real nodule visualization
		real_nodules = []
		for annotation in annotations:
			real_nodule = np.int_([(annotation[2]-origin[0])/new_spacing[0], (annotation[1]-origin[1])/new_spacing[1], (annotation[0]-origin[2])/new_spacing[2]])
			real_nodules.append(real_nodule)
		if RESULT_VISION:
			annotation_vision = cvm.view_coordinations(image, real_nodules, window_size=10, reverse=False, slicewise=False, show=False)
			np.save(vision_path+"/"+uid+"_annotations.npy", annotation_vision)

		segmask = lps.segment_lung_mask_fast(image)
		if segmask is None:
			print('Lung segmentation failed')
			#patient_evaluations.write('%d/%d patient %s lung segmentation failed\n' %(p+1, len(test_patients), uid))
			continue
		#print('lung segment. time{}s' .format(time.time()-start_time))
		segmask = lps.extend_mask(segmask)
		#segresult = lc.segment_vision(image, segmask)
		#np.save('temp.npy', segresult)
		#del segresult
		#segresult = lc.segment_vision(image, tempmask)
		#np.save('temp2.npy', segresult)
		#del segresult
		mbb = lps.mask_boundbox(segmask)
		'''
		mbb.z_top = 290
		mbb.z_bottom = 320
		mbb.y_top = 225
		mbb.y_bottom = 255
		mbb.x_top = 290
		mbb.x_bottom = 320
		'''
		image_bounded = image[mbb.z_top:mbb.z_bottom+1, mbb.y_top:mbb.y_bottom+1, mbb.x_top:mbb.x_bottom+1]
		print('Lung Segmentation Done. time:{}s' .format(time.time()-start_time))

		#nodule_matrix, cindex = cd.candidate_detection(segimage)
		#cluster_labels = lc.seed_volume_cluster(nodule_matrix, cluster_size=30, result_vision=False)
		num_segments = int(image_bounded.shape[0] * image_bounded.shape[1] * image_bounded.shape[2] / 27)	#the volume of a 3mm nodule is 27 voxels
		#num_segments = 20
		print('cluster number:%d' %(num_segments))
		cluster_labels = 0 - np.ones(shape=image.shape, dtype=int)
		cluster_labels_bounded = lc.slic_segment(image_bounded, compactness=0.01, num_segments=num_segments)
		print('Clustering Done. time:{}s' .format(time.time()-start_time))
		cluster_labels[mbb.z_top:mbb.z_bottom+1, mbb.y_top:mbb.y_bottom+1, mbb.x_top:mbb.x_bottom+1] = cluster_labels_bounded
		cluster_labels[np.logical_or(segmask==0, image<-600)] = -1
		#cluster_labels_filtered_fast = lc.cluster_filter_fast(image, cluster_labels)
		#print('Cluster Filtering Fast Done. time:{}s'.format(time.time() - start_time))
		#cluster_labels_filtered = lc.cluster_filter(image, cluster_labels)	#the clusters with no tissue are filetered out
		#candidate_coords_slow, candidate_labels_slow = lc.cluster_centers_fast(cluster_labels_filtered)
		#volume_candidated = cvm.view_coordinations(image, candidate_coords_slow, window_size=10, reverse=False, slicewise=True, show=False)
		#np.save(vision_path+"/"+uid+"_candidate_slow.npy", volume_candidated)
		#print('candidate slow number:%d' %(len(candidate_coords_slow)))
		#print('Cluster Filtering Done. time:{}s'.format(time.time() - start_time))

		#segresultfast = lc.segment_vision(image, cluster_labels_filtered_fast)
		#np.save(vision_path+"/"+uid+"_segresultfast.npy", segresultfast)
		if RESULT_VISION:
			np.save(vision_path + "/" + uid + "_segmask.npy", cluster_labels)
			segresult = lc.segment_vision(image, cluster_labels)
			np.save(vision_path + "/" + uid + "_segresult.npy", segresult)
		candidate_coords, candidate_labels = lc.cluster_centers(cluster_labels)
		#print("centering done {}" .format(time.time() - start_time))
		#if RESULT_VISION:
		#	volume_candidated = cvm.view_coordinations(image, candidate_coords, window_size=10, reverse=False, slicewise=True, show=False)
		#	np.save(vision_path+"/"+uid+"_candidate.npy", volume_candidated)
		print('Candidate Done. time:{}s' .format(time.time()-start_time))

		print('candidate number:%d' %(len(candidate_coords)))
		label_predictions = precise_detection(image, candidate_coords, candidate_labels, sess, inputs, combined_prediction)
		#label_predictions = precise_detection_with_labels(image, cluster_labels, sess, inputs, combined_prediction)
		result_labels = 0 - np.ones(shape=cluster_labels.shape, dtype=int)
		result_predictions = np.zeros(shape=cluster_labels.shape, dtype=float)
		for label, prediction in label_predictions:
			result_labels[cluster_labels==label] = label
			result_predictions[cluster_labels==label] = prediction
		#np.save(vision_path+"/"+uid+"_detlabels.npy", result_labels)
		if RESULT_VISION:
			detresult = lc.segment_vision(image, result_labels)
			np.save(vision_path+"/"+uid+"_detresult.npy", detresult)
		nodule_centers = nd.prediction_centering(result_predictions)
		print('Detection Done. time:{}s' .format(time.time()-start_time))

		if RESULT_VISION:
			nodules = []
			for nc in range(len(nodule_centers)):
				nodules.append(nodule_centers[nc][0:3])
			volume_predicted = cvm.view_coordinations(result_predictions*1000, nodules, window_size=10, reverse=False, slicewise=False, show=False)
			np.save(vision_path+"/"+uid+"_prediction.npy", volume_predicted)
		'''
		#randomly create a result for testing
		nodule_centers = []
		for nc in range(10):
			nodule_centers.append([random.randint(0,image.shape[0]-1), random.randint(0,image.shape[1]-1), random.randint(0,image.shape[2]-1), random.random()])
		'''
		print('Nodule coordinations:')
		if len(nodule_centers)<=0:
			print('none')
		for nc in range(len(nodule_centers)):
			#the output coordination order is [x,y,z], while the order for volume image should be [z,y,x]
			result.append([uid, (nodule_centers[nc][2]*new_spacing[2])+origin[2], (nodule_centers[nc][1]*new_spacing[1])+origin[1], (nodule_centers[nc][0]*new_spacing[0])+origin[0], nodule_centers[nc][3]])
			results.append([uid, (nodule_centers[nc][2]*new_spacing[2])+origin[2], (nodule_centers[nc][1]*new_spacing[1])+origin[1], (nodule_centers[nc][0]*new_spacing[0])+origin[0], nodule_centers[nc][3]])
			print('%d %d %d %f' %(nodule_centers[nc][0], nodule_centers[nc][1], nodule_centers[nc][2], nodule_centers[nc][3]))
		output_frame = pd.DataFrame(data=results, columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability'])
		output_frame.to_csv(result_file, index=False, float_format='%.4f')
		#if len(results)<=0:
		#	print("No results to evaluate, continue")
		#	continue
		assessment = eva.detection_assessment(results, annotation_file, candidate_file)
		if assessment is None:
			print('assessment failed')
			#patient_evaluations.write('%d/%d patient %s assessment failed\n' %(p+1, len(test_patients), uid))
			continue
		num_scans, FPsperscan, sensitivities, CPMscore, FPsperscan2, sensitivities2, CPMscore2, nodules_detected = assessment

		if len(FPsperscan)<=0 or len(sensitivities)<=0:
			print("No results to evaluate, continue")
		else:
			eva.evaluation_vision(CPMs, num_scans, FPsperscan, sensitivities, CPMscore, nodules_detected, CPM_output = evaluation_path + "/CPMscores.log", FROC_output = evaluation_path + "/froc_" + str(num_scans) + "scans.png")

		if len(FPsperscan2)<=0 or len(sensitivities2)<=0:
			print("No results to evaluate, continue")
		else:
			eva.evaluation_vision(CPMs2, num_scans, FPsperscan2, sensitivities2, CPMscore2, nodules_detected, CPM_output = evaluation_path + "/CPMscores2.log", FROC_output = evaluation_path + "/froc2_" + str(num_scans) + "scans.png")

		#patient_evaluations.write('%d/%d patient %s CPM score:%f\n' %(p+1, len(test_patients), uid, single_assessment[6]))
		print('Evaluation Done. time:{}s' .format(time.time()-start_time))

	#sess = ktb.get_session()
	#patient_evaluations.close()
	sess.close()
	output_frame = pd.DataFrame(data=results, columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability'])
	output_frame.to_csv(result_file, index=False, float_format='%.4f')
	print('Overall Detection Done')
