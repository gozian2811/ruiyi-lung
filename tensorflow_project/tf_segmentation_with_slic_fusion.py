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
sys.path.append("/home/fyl/programs/lung_project")
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
	#test_paths = ["./LUNA16/subset9"]
	test_paths = ["../datasets/SPH_data/0721"]
	#test_filelist = "results/evaluation_committefusion_segmentation/patientfilelist.log"
	net_files = ["models_tensorflow/luna_slh_3D_bndo_flbias_l5_20_aug_stage3/epoch20/epoch20",
		     "models_tensorflow/luna_slh_3D_bndo_flbias_l5_30_aug2_stage2/epoch10/epoch10", 
		     "models_tensorflow/luna_slh_3D_bndo_flbias_l6_40_aug_stage2/epoch25/epoch25"]
	fusion_file = "models_tensorflow/luna_slh_3D_fusion2/epoch6/epoch6"
	bn_files = ["models_tensorflow/luna_slh_3D_bndo_flbias_l5_20_aug_stage3/batch_normalization_statistic.npy",
		   "models_tensorflow/luna_slh_3D_bndo_flbias_l5_30_aug2_stage2/batch_normalization_statistic.npy",
		   "models_tensorflow/luna_slh_3D_bndo_flbias_l6_40_aug_stage2/batch_normalization_statistic_25.npy"]
	#annotation_file = "LUNA16/csvfiles/annotations_corrected.csv"
	annotation_file = "../datasets/SPH_data/annotations.xlsx"
	#exclude_file = "LUNA16/csvfiles/annotations_excluded.csv"
	vision_path = "./detection_vision/sph"
	result_path = "./results"
	evaluation_path = result_path + "/evaluation_" + FUSION_MODE + "fusion_segmentation_sph"
	result_file = evaluation_path + "/result.csv"

	if "test_filelist" in dir() and os.access(test_filelist, os.F_OK):
		all_patients = bt.filelist_load(test_filelist)
	elif "test_paths" in dir():
		all_patients = []
		#for path in test_paths:
		#	all_patients += glob(path + "/*.mhd")
		for path in test_paths:
			all_patients.extend(bt.get_dirs(path))
		if len(all_patients)<=0:
			print("No patient found")
			exit()
		#random.shuffle(all_patients)
		bt.filelist_store(all_patients, evaluation_path + "_filelist.log")
	else:
		print("No test data")
		exit()
	if 'vision_path' in dir() and 'vision_path' is not None and not os.access(vision_path, os.F_OK):
		os.makedirs(vision_path)
	if os.access(evaluation_path, os.F_OK):
		shutil.rmtree(evaluation_path)
	if not os.access(evaluation_path, os.F_OK):
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
	test_patients = all_patients
	#test_patients = ["./LUNA16/subset9/1.3.6.1.4.1.14519.5.2.1.6279.6001.195557219224169985110295082004.mhd"]
	for p in range(len(test_patients)):
		result = []
		patient = test_patients[p]
		#patient = "./LUNA16/subset9/1.3.6.1.4.1.14519.5.2.1.6279.6001.300270516469599170290456821227.mhd"
		#patient = "./LUNA16/subset9/1.3.6.1.4.1.14519.5.2.1.6279.6001.212608679077007918190529579976.mhd"
		#patient = "./LUNA16/subset9/1.3.6.1.4.1.14519.5.2.1.6279.6001.102681962408431413578140925249.mhd"
		#patient = "./TIANCHI_examples/LKDS-00005.mhd"
		#uid = mt.get_sample_uid(patient)
		#annotations = mt.get_luna_annotations(uid, annotation_file)
		full_scan, full_image_info, uid = mt.read_sph_scan(patient)
		origin = np.array(full_image_info.GetOrigin())[::-1]
		old_spacing = np.array(full_image_info.GetSpacing())[::-1]
		annotations = mt.get_sph_annotations(uid, annotation_file)
		if len(annotations)==0:
			print('%d/%d patient %s has no annotations, ignore it.' %(p+1, len(test_patients), uid))
			#patient_evaluations.write('%d/%d patient %s has no annotations, ignore it\n' %(p+1, len(test_patients), uid))
			continue

		print('%d/%d processing patient:%s' %(p+1, len(test_patients), uid))
		#full_image_info = sitk.ReadImage(patient)
		#full_scan = sitk.GetArrayFromImage(full_image_info)
		#origin = np.array(full_image_info.GetOrigin())[::-1]	#the order of origin and old_spacing is initially [z,y,x]
		#old_spacing = np.array(full_image_info.GetSpacing())[::-1]
		image, new_spacing = mt.resample(full_scan, old_spacing)	#resample
		#image = full_scan
		#new_spacing = old_spacing
		print('Resample Done. time:{}s' .format(time.time()-start_time))

		#make a real nodule visualization
		real_nodules = []
		for annotation in annotations:
			#real_nodule = np.int_([abs(annotation[2]-origin[0])/new_spacing[0], abs(annotation[1]-origin[1])/new_spacing[1], abs(annotation[0]-origin[2])/new_spacing[2]])
			real_nodule = np.int_(annotation[::-1]*old_spacing/new_spacing)
			real_nodules.append(real_nodule)
		if 'vision_path' in dir() and 'vision_path' is not None:
			annotation_vision = cvm.view_coordinates(image, real_nodules, window_size=10, reverse=False, slicewise=False, show=False)
			np.save(vision_path+"/"+uid+"_annotations.npy", annotation_vision)

		candidate_results = nd.nodule_context_slic(image, real_nodules)
		if candidate_results is None:
			continue
		candidate_coords, candidate_labels, cluster_labels = candidate_results
		print('Candidate Done. time:{}s' .format(time.time()-start_time))

		print('candidate number:%d' %(len(candidate_coords)))
		candidate_predictions = nd.precise_detection_multilevel(image, REGION_SIZES, candidate_coords, sess, inputs, combined_prediction, CANDIDATE_BATCH, AUGMENTATION, 0.1)
		valid_predictions = candidate_predictions > 0
		result_predictions, result_labels = nd.predictions_map_fast(cluster_labels, candidate_predictions[valid_predictions], candidate_labels[valid_predictions])
		if 'vision_path' in dir() and 'vision_path' is not None:
			np.save(vision_path+"/"+uid+"_detlabels.npy", result_labels)
			np.save(vision_path+"/"+uid+"_detpredictions.npy", result_predictions)
			#detresult = lc.segment_vision(image, result_labels)
			#np.save(vision_path+"/"+uid+"_detresult.npy", detresult)
		diameters = nd.diameter_calc(result_predictions, real_nodules)
		for ani in range(len(annotations)):
			print("calculation comparison:{} {}" .format(annotations[ani], diameters[ani]))
			#results.append([uid, annotations[ani][0], annotations[ani][1], annotations[ani][2], annotations[ani][3], diameters[ani]])
			results.append([uid, annotations[ani][0], annotations[ani][1], annotations[ani][2], diameters[ani]])
		#output_frame = pd.DataFrame(data=results, columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_anno', 'diameter_pred'])
		output_frame = pd.DataFrame(data=results, columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_pred'])
		output_frame.to_csv(result_file, index=False, float_format='%.4f')
		np.save(evaluation_path+"/"+uid+"_segmentations.npy", result_labels>=0)
		#nodule_center_predictions, prediction_labels = nd.prediction_cluster(result_predictions)
		print('Segmentation Done. time:{}s' .format(time.time()-start_time))

	sess.close()
	print('Overall Detection Done')
