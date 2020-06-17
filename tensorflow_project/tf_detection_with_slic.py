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
from toolbox import CandidateDetection as cd
from toolbox import Lung_Cluster as lc
from toolbox import TensorflowTools as tft
from toolbox import Nodule_Detection as nd
from toolbox import Evaluations as eva
try:
	from tqdm import tqdm # long waits are not fun
except:
	print('tqdm 是一个轻量级的进度条小包。。。')
	tqdm = lambda x : x

REGION_SIZE = 30
CANDIDATE_BATCH = 10

if __name__ == "__main__":
	test_paths = ["./LUNA16/subset9"]
	#test_filelist = './models_tensorflow/luna_tianchi_slh_3D_l3454-512-2_bn2_stage3/pfilelist.log'
	net_file = "models_tensorflow/luna_3D_bndo_flbias_l5_30_aug/epoch10/epoch10"
	bn_file = "models_tensorflow/luna_3D_bndo_flbias_l5_30_aug/epoch10/batch_normalization_statistic.npy"
	annotation_file = "LUNA16/csvfiles/annotations_corrected.csv"
	exclude_file = "LUNA16/csvfiles/annotations_excluded_corrected.csv"
	#vision_path = "./detection_vision/test"
	result_path = "./results"
	evaluation_path = result_path + "/experiments3/evaluations_30"
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
	if 'vision_path' in dir() and 'vision_path' is not None and not os.access(vision_path, os.F_OK):
		os.makedirs(vision_path)
	if os.access(evaluation_path, os.F_OK):
		shutil.rmtree(evaluation_path)
	os.makedirs(evaluation_path)

	x = tf.placeholder(tf.float32, [None, REGION_SIZE, REGION_SIZE,
						REGION_SIZE])
	x_image = tf.reshape(x, [-1, REGION_SIZE, REGION_SIZE,
				 REGION_SIZE, 1])
	bn_params = np.load(bn_file)
	outputs, _, _ = tft.volume_bndo_flbias_l5_30(x_image, dropout_rate=0.0, batch_normalization_statistic=False, bn_params=bn_params)
	prediction_out = outputs['sm_out']
	saver = tf.train.Saver()
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	saver.restore(sess, net_file)

	start_time = time.time()
	#patient_evaluations = open(evaluation_path + "/patient_evaluations.log", "w")
	results = []
	CPMs = []
	CPMs2 = []
	test_patients = all_patients
	bt.filelist_store(all_patients, evaluation_path + "/patientfilelist.log")
	random.shuffle(test_patients)
	for p in range(len(test_patients)):
		patient = test_patients[p]
		#patient = "./LUNA16/subset9/1.3.6.1.4.1.14519.5.2.1.6279.6001.212608679077007918190529579976.mhd"
		#patient = "./LUNA16/subset9/1.3.6.1.4.1.14519.5.2.1.6279.6001.102681962408431413578140925249.mhd"
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
			real_nodule = np.int_([abs(annotation[2]-origin[0])/new_spacing[0], abs(annotation[1]-origin[1])/new_spacing[1], abs(annotation[0]-origin[2])/new_spacing[2]])
			real_nodules.append(real_nodule)
		if 'vision_path' in dir() and 'vision_path' is not None:
			annotation_vision = cvm.view_coordinations(image, real_nodules, window_size=10, reverse=False, slicewise=False, show=False)
			np.save(vision_path+"/"+uid+"_annotations.npy", annotation_vision)

		candidate_results = nd.slic_candidate(image)
		if candidate_results is None:
			continue
		candidate_coords, candidate_labels, cluster_labels = candidate_results
		if 'vision_path' in dir() and vision_path is not None:
			np.save(vision_path + "/" + uid + "_segmask.npy", cluster_labels)
			#segresult = lc.segment_vision(image, cluster_labels)
			#np.save(vision_path + "/" + uid + "_segresult.npy", segresult)
		print('Candidate Done. time:{}s' .format(time.time()-start_time))

		print('candidate number:%d' %(len(candidate_coords)))
		candidate_predictions = nd.precise_detection(image, REGION_SIZE, candidate_coords, sess, x, prediction_out, CANDIDATE_BATCH, 0.2)
		positive_predictions = candidate_predictions > 0
		result_predictions, result_labels = nd.predictions_map_fast(cluster_labels, candidate_predictions[positive_predictions], candidate_labels[positive_predictions])
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
			volume_predicted = cvm.view_coordinations(result_predictions*1000, nodules, window_size=10, reverse=False, slicewise=False, show=False)
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

	sess.close()
	#output_frame = pd.DataFrame(data=results, columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability'])
	#output_frame.to_csv(result_file, index=False, float_format='%.4f')
	print('Overall Detection Done')
