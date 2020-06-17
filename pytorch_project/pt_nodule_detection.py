#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import copy
import time
import models
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
from toolbox import CandidateDetection as cd
from toolbox import Lung_Cluster as lc
from toolbox import Nodule_Detection as nd
from toolbox import Evaluations as eva
try:
	from tqdm import tqdm # long waits are not fun
except:
	print('tqdm 是一个轻量级的进度条小包。。。')
	tqdm = lambda x : x

#START_NUM = 66
REGION_SIZE = 32
CANDIDATE_BATCH = 5

if __name__ == "__main__":
	test_paths = ["/data/fyl/datasets/LUNA16/subset0", "/data/fyl/datasets/LUNA16/subset1", "/data/fyl/datasets/LUNA16/subset2", "/data/fyl/datasets/LUNA16/subset3", "/data/fyl/datasets/LUNA16/subset4", "/data/fyl/datasets/LUNA16/subset5", "/data/fyl/datasets/LUNA16/subset6", "/data/fyl/datasets/LUNA16/subset7", "/data/fyl/datasets/LUNA16/subset8", "/data/fyl/datasets/LUNA16/subset9"]
	#test_sample_paths = ["/home/fyl/datasets/luna_64/test"]
	test_sample_filelist = "/data/fyl/models_pytorch/DensecropNet_detection_2_rfold5/filelist_val_fold4.log"
	net_file = "/data/fyl/models_pytorch/DensecropNet_detection_2_rfold5/DensecropNet_detection_2_rfold5_epoch7"
	annotation_file = "/data/fyl/datasets/LUNA16/csvfiles/annotations_corrected.csv"
	exclude_file = "/data/fyl/datasets/LUNA16/csvfiles/annotations_excluded_corrected.csv"
	#candidate_file = "/data/fyl/datasets/LUNA16/csvfiles/candidates_V2.csv"
	evaluation_path = "./experiments_dt/evaluations_sliccand_densecropnet_2_e7_fold5"
	#vision_path = evaluation_path
	result_file = evaluation_path + "/result.csv"
	hard_negatives_file = evaluation_path + "/hard_negatives.csv"

	if "test_paths" in dir():
		all_patients = []
		for path in test_paths:
			all_patients += glob(path + "/*.mhd")
		if len(all_patients)<=0:
			print("No patient found")
			exit()
	else:
		print("No test data")
		exit()
	if "test_sample_filelist" not in dir():
		for path in test_sample_paths:
			test_samples = glob(path + '/*.npy')
	else:
		test_samples = bt.filelist_load(test_sample_filelist)
	test_uids = []
	for test_sample in test_samples:
		sample_uid = os.path.basename(test_sample).split('_')[0]
		if sample_uid not in test_uids:
			test_uids.append(sample_uid)
	if 'vision_path' in dir() and vision_path is not None and not os.access(vision_path, os.F_OK):
		os.makedirs(vision_path)
	#if os.access(evaluation_path, os.F_OK): shutil.rmtree(evaluation_path)
	if not os.access(evaluation_path, os.F_OK): os.makedirs(evaluation_path)
	shutil.copyfile(net_file, evaluation_path+'/'+net_file.split('/')[-1])

	model = models.DensecropNet(input_size=REGION_SIZE, drop_rate=0, growth_rate=64, num_blocks=4, num_fin_growth=3)
	model.load(net_file)
	model.eval()
	model.cuda()

	start_time = time.time()
	#patient_evaluations = open(evaluation_path + "/patient_evaluations.log", "w")
	results = []
	CPMs = []
	CPMs2 = []
	test_patients = all_patients
	#test_count = 0
	#random.shuffle(test_patients)
	bt.filelist_store(test_patients, evaluation_path + "/patientfilelist.log")
	for p in range(len(test_patients)):
		patient = test_patients[p]
		#patient = "./LUNA16/subset9/1.3.6.1.4.1.14519.5.2.1.6279.6001.212608679077007918190529579976.mhd"
		#patient = "./LUNA16/subset9/1.3.6.1.4.1.14519.5.2.1.6279.6001.102681962408431413578140925249.mhd"
		#patient = "./TIANCHI_examples/LKDS-00005.mhd"
		uid = mt.get_mhd_uid(patient)
		if 'test_uids' not in dir() or uid not in test_uids:
			print("%d/%d patient %s not belong to test set" %(p+1, len(test_patients), uid))
			continue
		annotations = mt.get_luna_annotations(uid, annotation_file)
		exclusions = mt.get_luna_annotations(uid, exclude_file)
		if len(annotations) == 0:
			print("%d/%d patient %s has no annotations, ignore it." %(p+1, len(test_patients), uid))
			#patient_evaluations.write('%d/%d patient %s has no annotations, ignore it\n' %(p+1, len(test_patients), uid))
			continue
		#test_count += 1
		#if test_count < START_NUM:
			#the START_NUM begin from 1
			#print("%d/%d patient %s count %d/%d." %(p+1, len(test_patients), uid, test_count, START_NUM))
			#continue

		print('%d/%d processing patient:%s' %(p+1, len(test_patients), uid))
		full_image_info = sitk.ReadImage(patient)
		full_scan = sitk.GetArrayFromImage(full_image_info)
		origin = np.array(full_image_info.GetOrigin())[::-1]	#the order of origin and old_spacing is initially [z,y,x]
		old_spacing = np.array(full_image_info.GetSpacing())[::-1]
		image, new_spacing = mt.resample(full_scan, old_spacing, np.array([1, 1, 1]))
		#image, new_spacing = full_scan, old_spacing
		print('Resample Done. time:{}s' .format(time.time()-start_time))

		#make a real nodule visualization
		real_nodules = []
		for annotation in annotations:
			real_nodule = np.int_([abs(annotation[2]-origin[0])/new_spacing[0], abs(annotation[1]-origin[1])/new_spacing[1], abs(annotation[0]-origin[2])/new_spacing[2]])
			real_nodules.append(real_nodule)
		excluded_nodules = []
		for exclusion in exclusions:
			excluded_nodule = np.int_([abs(exclusion[2]-origin[0])/new_spacing[0], abs(exclusion[1]-origin[1])/new_spacing[1], abs(exclusion[0]-origin[2])/new_spacing[2]])
			excluded_nodules.append(excluded_nodule)
		
		if 'vision_path' in dir() and vision_path is not None:
			annotation_vision = cvm.view_coordinates(image, real_nodules, window_size=10, reverse=False, slicewise=False, show=False)
			np.save(vision_path+"/"+uid+"_annotations.npy", annotation_vision)
			exclusion_vision = cvm.view_coordinates(image, excluded_nodules, window_size=10, reverse=False, slicewise=False, show=False)
			np.save(vision_path+"/"+uid+"_exclusions.npy", exclusion_vision)

		if 'candidate_file' in dir():
			print('Detection with given candidates:{}' .format(candidate_file))
			candidate_coords = nd.luna_candidate(image, uid, origin, new_spacing, candidate_file, lung_segment=True, vision_path=vision_path)
			if 'vision_path' in dir() and vision_path is not None:
				volume_candidate = cvm.view_coordinates(image, candidate_coords, window_size=10, reverse=False, slicewise=False, show=False)
				np.save(vision_path+"/"+uid+"_candidate.npy", volume_candidate)
			print('Candidate Done. time:{}s' .format(time.time()-start_time))
			print('candidate number:%d' %(len(candidate_coords)))
			candidate_predictions = nd.precise_detection_pt(image, REGION_SIZE, candidate_coords, model, None, CANDIDATE_BATCH, 0.4)
			positive_predictions = candidate_predictions > 0
			predicted_coords = np.delete(candidate_coords, np.logical_not(positive_predictions).nonzero()[0], axis=0)
			predictions = candidate_predictions[positive_predictions]
			nodule_center_predictions = nd.prediction_combine(predicted_coords, predictions)
			if 'vision_path' in dir() and vision_path is not None:
				volume_predicted = cvm.view_coordinates(image, predicted_coords, window_size=10, reverse=False, slicewise=False, show=False)
				np.save(vision_path+"/"+uid+"_predicted.npy", volume_predicted)
				nodules = []
				for nc in range(len(nodule_center_predictions)):
					nodules.append(np.int_(nodule_center_predictions[nc][0:3]))
				volume_prediction = cvm.view_coordinates(image, nodules, window_size=10, reverse=False, slicewise=False, show=False)
				np.save(vision_path+"/"+uid+"_prediction.npy", volume_prediction)
		else:
			print('Detection with slic candidates')
			candidate_results = nd.slic_candidate(image, int(27/np.prod(new_spacing)+0.5))
			if candidate_results is None:
				continue
			candidate_coords, candidate_labels, cluster_labels = candidate_results
			if 'vision_path' in dir() and vision_path is not None:
				np.save(vision_path + "/" + uid + "_segmask.npy", cluster_labels)
				#segresult = lc.segment_vision(image, cluster_labels)
				#np.save(vision_path + "/" + uid + "_segresult.npy", segresult)
			print('Candidate Done. time:{}s' .format(time.time()-start_time))
			print('candidate number:%d' %(len(candidate_coords)))
			candidate_predictions = nd.precise_detection_pt(image, REGION_SIZE, candidate_coords, model, None, CANDIDATE_BATCH, 0.4)
			positive_predictions = candidate_predictions > 0
			result_predictions, result_labels = nd.predictions_map_fast(cluster_labels, candidate_predictions[positive_predictions], candidate_labels[positive_predictions])
			if 'vision_path' in dir() and vision_path is not None:
				np.save(vision_path+"/"+uid+"_detlabels.npy", result_labels)
				np.save(vision_path+"/"+uid+"_detpredictions.npy", result_predictions)
				#detresult = lc.segment_vision(image, result_labels)
				#np.save(vision_path+"/"+uid+"_detresult.npy", detresult)
			nodule_center_predictions = nd.prediction_centering_fast(result_predictions)
			#nodule_center_predictions, prediction_labels = nd.prediction_cluster(result_predictions)
			if 'vision_path' in dir() and vision_path is not None:
				nodules = []
				for nc in range(len(nodule_center_predictions)):
					nodules.append(np.int_(nodule_center_predictions[nc][0:3]))
				volume_predicted = cvm.view_coordinates(result_predictions*1000, nodules, window_size=10, reverse=False, slicewise=False, show=False)
				np.save(vision_path+"/"+uid+"_prediction.npy", volume_predicted)
				if 'prediction_labels' in dir():
					prediction_cluster_vision = lc.segment_color_vision(prediction_labels)
					np.save(vision_path+"/"+uid+"_prediction_clusters.npy", prediction_cluster_vision)
		print('Detection Done. time:{}s' .format(time.time()-start_time))

		'''
		#randomly create a result for testing
		nodule_center_predictions = []
		for nc in range(10):
			nodule_center_predictions.append([random.randint(0,image.shape[0]-1), random.randint(0,image.shape[1]-1), random.randint(0,image.shape[2]-1), random.random()])
		'''
		if len(nodule_center_predictions)<1000:
			print('Nodule coordinations:')
			if len(nodule_center_predictions)<=0:
				print('none')
			for nc in range(len(nodule_center_predictions)):
				print('{} {} {} {}' .format(nodule_center_predictions[nc][0], nodule_center_predictions[nc][1], nodule_center_predictions[nc][2], nodule_center_predictions[nc][3]))
		for nc in range(len(nodule_center_predictions)):
			#the output coordination order is [x,y,z], while the order for volume image should be [z,y,x]
			results.append([uid, (nodule_center_predictions[nc][2]*new_spacing[2])+origin[2], (nodule_center_predictions[nc][1]*new_spacing[1])+origin[1], (nodule_center_predictions[nc][0]*new_spacing[0])+origin[0], nodule_center_predictions[nc][3]])
			#if len(nodule_center_predictions)<1000:
				#print('{} {} {} {}' .format(nodule_center_predictions[nc][0], nodule_center_predictions[nc][1], nodule_center_predictions[nc][2], nodule_center_predictions[nc][3]))
		result_frame = pd.DataFrame(data=results, columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability'])
		result_frame.to_csv(result_file, index=False, float_format='%.4f')

		assessment = eva.detection_assessment(results, annotation_file, exclude_file)
		if assessment is None:
			print('assessment failed')
			#patient_evaluations.write('%d/%d patient %s assessment failed\n' %(p+1, len(test_patients), uid))
			continue
		#num_scans, FPsperscan, sensitivities, CPMscore, FPsperscan2, sensitivities2, CPMscore2, nodules_detected = assessment
		num_scans = assessment['num_scans']
		FPsperscan, sensitivities = assessment['FROC']
		CPMscore = assessment['CPM']
		prediction_order = assessment['prediction_order']
		nodules_detected = assessment['detection_cites']
		if len(FPsperscan)<=0 or len(sensitivities)<=0:
			print("No results to evaluate, continue")
		else:
			eva.evaluation_vision(CPMs, num_scans, FPsperscan, sensitivities, CPMscore, nodules_detected, evaluation_path)
		#patient_evaluations.write('%d/%d patient %s CPM score:%f\n' %(p+1, len(test_patients), uid, single_assessment[6]))
		print('Evaluation Done. time:{}s' .format(time.time()-start_time))
		
		hard_negatives = []
		num_positive = (nodules_detected>=0).nonzero()[0].size
		for ndi in range(len(nodules_detected)):
			if results[prediction_order[ndi]][4]<=0.5 or (nodules_detected[:ndi]>=0).nonzero()[0].size==num_positive: break
			if nodules_detected[ndi]==-1: hard_negatives.append(results[prediction_order[ndi]])
		hard_negatives_frame = pd.DataFrame(data=hard_negatives, columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability'])
		hard_negatives_frame.to_csv(hard_negatives_file, index=False, float_format='%.4f')
		print('Hard Negatives Extracted. time:{}s' .format(time.time()-start_time))
		
	print('Overall Detection Done')
