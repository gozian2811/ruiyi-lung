#!/usr/bin/env python
# encoding: utf-8

import os
import sys
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
sys.path.append('/home/fyl/programs/lung_project')
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
REGION_SIZES = (20, 30, 40)
CANDIDATE_BATCH = 15
AUGMENTATION = False
FUSION_MODE = 'committe'

if __name__ == "__main__":
	test_path = "../datasets/weiyi_lung"
	net_files = ["models_tensorflow/luna_slh_3D_bndo_flbias_l5_20_aug_stage3/epoch20/epoch20",
		     "models_tensorflow/luna_slh_3D_bndo_flbias_l5_30_aug2_stage2/epoch10/epoch10", 
		     "models_tensorflow/luna_slh_3D_bndo_flbias_l6_40_aug_stage2/epoch25/epoch25"]
	fusion_file = "models_tensorflow/luna_slh_3D_fusion2/epoch6/epoch6"
	bn_files = ["models_tensorflow/luna_slh_3D_bndo_flbias_l5_20_aug_stage3/batch_normalization_statistic.npy",
		   "models_tensorflow/luna_slh_3D_bndo_flbias_l5_30_aug2_stage2/batch_normalization_statistic.npy",
		   "models_tensorflow/luna_slh_3D_bndo_flbias_l6_40_aug_stage2/batch_normalization_statistic_25.npy"]
	annotation_file = "../datasets/LUNA16/csvfiles/annotations_corrected.csv"
	exclude_file = "../datasets/LUNA16/csvfiles/annotations_excluded_corrected.csv"
	vision_path = "./detection_vision/weiyi_2"
	result_path = "./results"
	evaluation_path = result_path + "/experiments1/evaluation_" + FUSION_MODE + "fusion_weiyi_2"
	result_file = evaluation_path + "/result.csv"

	all_patients = []
	for subdir in os.listdir(test_path):
		all_patients.append(test_path + '/' + subdir)
		
	if 'vision_path' in dir() and 'vision_path' is not None and not os.access(vision_path, os.F_OK):
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
		features = [outputs0['flattened_out'], outputs1['flattened_out'], outputs2['fc1_out']]
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
	test_patients = all_patients
	bt.filelist_store(all_patients, evaluation_path + "/patientfilelist.log")
	#random.shuffle(test_patients)
	for p in range(len(test_patients)):
		result = []
		patient = test_patients[p]
		#uid = mt.get_mhd_uid(patient)
		#annotations = mt.get_luna_annotations(uid, annotation_file)
		#if len(annotations)==0:
			#print('%d/%d patient %s has no annotations, ignore it.' %(p+1, len(test_patients), uid))
			#continue
		
		full_scan, image_info = mt.read_dicom_scan(patient)
		uid = image_info['uid']
		print('%d/%d processing patient:%s' %(p+1, len(test_patients), uid))

		origin = image_info['origin']
		old_spacing = image_info['spacing']
		image, new_spacing = mt.resample(full_scan, old_spacing)	#resample
		print('Resample Done. time:{}s' .format(time.time()-start_time))

		#make a real nodule visualization
		real_nodules = []
		#for annotation in annotations:
			#real_nodule = np.int_([abs(annotation[2]-origin[0])/new_spacing[0], abs(annotation[1]-origin[1])/new_spacing[1], abs(annotation[0]-origin[2])/new_spacing[2]])
			#real_nodules.append(real_nodule)
		if 'vision_path' in dir() and 'vision_path' is not None:
			annotation_vision = cvm.view_coordinates(image, real_nodules, window_size=10, reverse=False, slicewise=False, show=False)
			np.save(vision_path+"/"+uid+"_annotations.npy", annotation_vision)

		candidate_results = nd.slic_candidate(image)
		if candidate_results is None:
			continue
		candidate_coords, candidate_labels, cluster_labels = candidate_results
		print('Candidate Done. time:{}s' .format(time.time()-start_time))

		print('candidate number:%d' %(len(candidate_coords)))
		candidate_predictions = nd.precise_detection_multilevel(image, REGION_SIZES, candidate_coords, sess, inputs, combined_prediction, CANDIDATE_BATCH, AUGMENTATION, 0.5)
		valid_predictions = candidate_predictions > 0
		result_predictions, result_labels = nd.predictions_map_fast(cluster_labels, candidate_predictions[valid_predictions], candidate_labels[valid_predictions])
		if 'vision_path' in dir() and 'vision_path' is not None:
			np.save(vision_path+"/"+uid+"_detlabels.npy", result_labels)
			np.save(vision_path+"/"+uid+"_detpredictions.npy", result_predictions)
			#detresult = lc.segment_vision(image, result_labels)
			#np.save(vision_path+"/"+uid+"_detresult.npy", detresult)
		nodule_center_predictions = nd.prediction_centering_fast(result_predictions)
		#nodule_center_predictions, prediction_labels = nd.prediction_cluster(result_predictions)
		print('Detection Done. time:{}s' .format(time.time()-start_time))

		if 'vision_path' in dir() and 'vision_path' is not None:
			nodules = []
			for nc in range(len(nodule_center_predictions)):
				nodules.append(np.int_(nodule_center_predictions[nc][0:3]))
			volume_predicted = cvm.view_coordinates(result_predictions*1000, nodules, window_size=10, reverse=False, slicewise=False, show=False)
			np.save(vision_path+"/"+uid+"_prediction.npy", volume_predicted)
			if 'prediction_labels' in dir():
				prediction_cluster_vision = lc.segment_color_vision(prediction_labels)
				np.save(vision_path+"/"+uid+"_prediction_clusters.npy", prediction_cluster_vision)
		'''
		#randomly create a result for testing
		nodule_center_predictions = []
		for nc in range(10):
			nodule_center_predictions.append([random.randint(0,image.shape[0]-1), random.randint(0,image.shape[1]-1), random.randint(0,image.shape[2]-1), random.random()])
		'''
		print('Nodule coordinations:')
		if len(nodule_center_predictions)<=0:
			print('none')
		for nc in range(len(nodule_center_predictions)):
			#the output coordination order is [x,y,z], while the order for volume image should be [z,y,x]
			results.append([uid, (nodule_center_predictions[nc][2]*new_spacing[2])+origin[2], (nodule_center_predictions[nc][1]*new_spacing[1])+origin[1], (nodule_center_predictions[nc][0]*new_spacing[0])+origin[0], nodule_center_predictions[nc][3]])
			print('{} {} {} {}' .format(nodule_center_predictions[nc][0], nodule_center_predictions[nc][1], nodule_center_predictions[nc][2], nodule_center_predictions[nc][3]))
		output_frame = pd.DataFrame(data=results, columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability'])
		output_frame.to_csv(result_file, index=False, float_format='%.4f')
		#if len(results)<=0:
		#	print("No results to evaluate, continue")
		#	continue
		'''
		assessment = eva.detection_assessment(results, annotation_file, exclude_file)
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
		'''

	sess.close()
	output_frame = pd.DataFrame(data=results, columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability'])
	output_frame.to_csv(result_file, index=False, float_format='%.4f')
	print('Overall Detection Done')
