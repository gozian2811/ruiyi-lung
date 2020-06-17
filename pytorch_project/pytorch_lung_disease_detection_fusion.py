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
from glob import glob
from skimage import measure
from collections import OrderedDict
from configs import config_dt
opt = config_dt.DefaultConfig()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.Devices_ID
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

if __name__ == "__main__":
	test_paths = ["/data/fyl/datasets/Tianchi_Lung_Disease/testA"]
	#test_sample_filelist = "/data/fyl/models_pytorch/DensecropNet_detection_test_rfold1/filelist_val_fold0.log"
	net_files = ["/data/fyl/models_pytorch/DensecropNet_nodule_detection_rfold1/DensecropNet_nodule_detection_rfold1_epoch2",
		     "/data/fyl/models_pytorch/DensecropNet_stripe_detection_2_rfold1/DensecropNet_stripe_detection_2_rfold1_epoch1"]
	#annotation_file = "/data/fyl/datasets/Tianchi_Lung_Disease/chestCT_round1_annotation.csv"
	#candidate_file = "/data/fyl/datasets/Tianchi_Lung_Disease/candidate.csv"
	labels = [1, 5]
	result_path = "./experiments_dt/evaluations_tianchild_densecropnet_fusion_rfold1"
	#result_path = "experiments_dt/evaluations_test"
	#vision_path = result_path
	#result_file = result_path + "/result.csv"
	hard_negatives_file = result_path + "/hard_negatives.csv"
	
	region_size = opt.input_size
	batch_size = opt.batch_size
	#label_dict = {'noduleclass':1, 'stripeclass':5, 'arterioclass':31, 'lymphnodecalclass':32}
	#label = label_dict[opt.label_mode]
	use_gpu = opt.use_gpu
	#net_file = opt.load_model_path

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
	if hasattr(opt, 'filelists') and 'test' in opt.filelists.keys():
		test_samples = bt.filelist_load(opt.filelists['test'])
		test_uids = []
		for test_sample in test_samples:
			sample_uid = os.path.basename(test_sample).split('_')[0]
			if sample_uid not in test_uids:
				test_uids.append(sample_uid)
		pd.DataFrame(data=test_uids, columns=['series_uid']).to_csv(result_path+'/patients_uid.csv', index=False)
	#else:
	#	for path in opt.filelists['test']:
	#		test_samples = glob(path + '/*.mhd')
	if 'vision_path' in dir() and vision_path is not None and not os.access(vision_path, os.F_OK):
		os.makedirs(vision_path)
	#if os.access(result_path, os.F_OK): shutil.rmtree(result_path)
	if not os.access(result_path, os.F_OK): os.makedirs(result_path)

	#model = models.DensecropNet(input_size=region_size, drop_rate=0, growth_rate=64, num_blocks=4, num_fin_growth=3).eval()
	models = [getattr(models, opt.model)(input_size=region_size, **opt.model_setup).eval() for m in range(len(net_files))]
	for n in range(len(net_files)):
		models[n].load(net_files[n])
		print('model loaded from %s' %(net_files[n]))
		shutil.copyfile(net_files[n], result_path+'/'+net_files[n].split('/')[-1])
		if use_gpu: models[n].cuda()

	start_time = time.time()
	#patient_evaluations = open(result_path + "/patient_evaluations.log", "w")
	results = []
	labeled_results = [[] for l in range(len(labels))]
	CPMs = [[] for l in range(len(labels))]
	#hard_negatives = []
	test_patients = all_patients[::-1]
	#random.shuffle(test_patients)
	bt.filelist_store(test_patients, result_path + "/patientfilelist.log")
	for p in range(len(test_patients)):
		patient = test_patients[p]
		uid = mt.get_mhd_uid(patient)
		if 'test_uids' in dir() and uid not in test_uids:
			print("%d/%d patient %s not belong to test set" %(p+1, len(test_patients), uid))
			continue

		print('%d/%d processing patient:%s' %(p+1, len(test_patients), uid))
		full_image_info = sitk.ReadImage(patient)
		full_scan = sitk.GetArrayFromImage(full_image_info)
		origin = np.array(full_image_info.GetOrigin())[::-1]	#the order of origin and old_spacing is initially [z,y,x]
		old_spacing = np.array(full_image_info.GetSpacing())[::-1]
		image, new_spacing = mt.resample(full_scan, old_spacing, np.array([1, 1, 1]))
		#image = np.load(patient)
		#new_spacing = np.array([1, 1, 1])
		#origin = np.array([0, 0, 0])
		print('Resample Done. time:{}s' .format(time.time()-start_time))

		candidate_results = nd.slic_candidate(image, 20, focus_area='lung')
		if candidate_results is None:
			continue
		candidate_coords, candidate_labels, cluster_labels = candidate_results
		if 'vision_path' in dir() and vision_path is not None:
			np.save(vision_path + "/" + uid + "_segmask.npy", cluster_labels)
			#segresult = lc.segment_vision(image, cluster_labels)
			#np.save(vision_path + "/" + uid + "_segresult.npy", segresult)
		print('Candidate Done. time:{}s' .format(time.time()-start_time))
		print('candidate number:%d' %(len(candidate_coords)))
		
		for l in range(len(labels)):
			label = labels[l]
			print('label: %d' %(label))
			evaluation_path = result_path + '/' + str(label)
			if not os.access(evaluation_path, os.F_OK): os.makedirs(evaluation_path)
			if 'annotation_file' in dir():
				annotations = mt.get_challenge_annotations(uid, annotation_file, label=label)
				if len(annotations) == 0:
					print("%d/%d patient %s has no annotations, ignore it." %(p+1, len(test_patients), uid))
					#patient_evaluations.write('%d/%d patient %s has no annotations, ignore it\n' %(p+1, len(test_patients), uid))
					continue
				#make a real nodule visualization
				if 'vision_path' in dir() and vision_path is not None:
					real_nodules = []
					for annotation in annotations:
						#real_nodule = np.int_([abs(annotation[2]-origin[0])/new_spacing[0], abs(annotation[1]-origin[1])/new_spacing[1], abs(annotation[0]-origin[2])/new_spacing[2]])
						real_nodule = mt.coord_conversion(annotation[:3][::-1], origin, old_spacing, full_scan.shape, image.shape, dir_array=True)
						real_nodules.append(real_nodule)
					annotation_vision = cvm.view_coordinates(image, real_nodules, window_size=10, reverse=False, slicewise=False, show=False)
					np.save(evaluation_path+"/"+uid+"_annotations.npy", annotation_vision)		
			candidate_predictions = nd.precise_detection_pt(image, region_size, candidate_coords, models[l], None, batch_size, use_gpu=use_gpu, prediction_threshold=0.4)
			positive_predictions = candidate_predictions > 0
			result_predictions, result_labels = nd.predictions_map_fast(cluster_labels, candidate_predictions[positive_predictions], candidate_labels[positive_predictions])
			if 'vision_path' in dir() and vision_path is not None:
				np.save(evaluation_path+"/"+uid+"_detlabels.npy", result_labels)
				np.save(evaluation_path+"/"+uid+"_detpredictions.npy", result_predictions)
				#detresult = lc.segment_vision(image, result_labels)
				#np.save(evaluation_path+"/"+uid+"_detresult.npy", detresult)
			nodule_center_predictions = nd.prediction_centering_fast(result_predictions)
			#nodule_center_predictions, prediction_labels = nd.prediction_cluster(result_predictions)
			if 'vision_path' in dir() and vision_path is not None:
				nodules = []
				for nc in range(len(nodule_center_predictions)):
					nodules.append(np.int_(nodule_center_predictions[nc][0:3]))
				volume_predicted = cvm.view_coordinates(result_predictions*1000, nodules, window_size=10, reverse=False, slicewise=False, show=False)
				np.save(evaluation_path+"/"+uid+"_prediction.npy", volume_predicted)
				if 'prediction_labels' in dir():
					prediction_cluster_vision = lc.segment_color_vision(prediction_labels)
					np.save(evaluation_path+"/"+uid+"_prediction_clusters.npy", prediction_cluster_vision)
			print('Detection Done. time:{}s' .format(time.time()-start_time))

			'''
			#randomly create a result for testing
			nodule_center_predictions = []
			for nc in range(10):
				nodule_center_predictions.append([random.randint(0,image.shape[0]-1), random.randint(0,image.shape[1]-1), random.randint(0,image.shape[2]-1), random.random()])
			'''
			'''
			if len(nodule_center_predictions)<1000:
				print('Nodule coordinations:')
				if len(nodule_center_predictions)<=0:
					print('none')
				for nc in range(len(nodule_center_predictions)):
					print('{} {} {} {}' .format(nodule_center_predictions[nc][0], nodule_center_predictions[nc][1], nodule_center_predictions[nc][2], nodule_center_predictions[nc][3]))
			'''
			for nc in range(len(nodule_center_predictions)):
				#the output coordination order is [x,y,z], while the order for volume image should be [z,y,x]
				result = [uid]
				result.extend(mt.coord_conversion(nodule_center_predictions[nc][:3], origin, old_spacing, full_scan.shape, image.shape, dir_array=False)[::-1])
				if label is not None: result.append(label)
				result.append(nodule_center_predictions[nc][3])
				results.append(result)
				labeled_results[l].append(result)
				#results.append([uid, (nodule_center_predictions[nc][2]*new_spacing[2])+origin[2], (nodule_center_predictions[nc][1]*new_spacing[1])+origin[1], (nodule_center_predictions[nc][0]*new_spacing[0])+origin[0], nodule_center_predictions[nc][3]])
				#if len(nodule_center_predictions)<1000:
					#print('{} {} {} {}' .format(nodule_center_predictions[nc][0], nodule_center_predictions[nc][1], nodule_center_predictions[nc][2], nodule_center_predictions[nc][3]))
			columns = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability']
			if label is not None:
				columns.insert(4, 'class')
			result_frame = pd.DataFrame(data=labeled_results[l], columns=columns)
			result_frame.to_csv("{}/result_{}.csv".format(evaluation_path, label), index=False, float_format='%.4f')
			#np.save("{}/result_{}.npy"%(evaluation_path, label), np.array(results))

			if 'annotation_file' in dir():
				assessment = eva.detection_assessment(labeled_results[l], annotation_file, label=label)
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
					eva.evaluation_vision(CPMs[l], num_scans, FPsperscan, sensitivities, CPMscore, nodules_detected, output_path = evaluation_path)
				#patient_evaluations.write('%d/%d patient %s CPM score:%f\n' %(p+1, len(test_patients), uid, single_assessment[6]))
				print('Evaluation Done. time:{}s' .format(time.time()-start_time))
			
				'''
				num_positive = (nodules_detected>=0).nonzero()[0].size
				for ndi in range(len(nodules_detected)):
					if results[prediction_order[ndi]][-1]<=0.5 or (nodules_detected[:ndi]>=0).nonzero()[0].size==num_positive: break
					if nodules_detected[ndi]==-1: hard_negatives.append(results[prediction_order[ndi]])
				hard_negatives_frame = pd.DataFrame(data=hard_negatives, columns=columns)
				hard_negatives_frame.to_csv(hard_negatives_file, index=False, float_format='%.4f')
				print('Hard Negatives Extracted. time:{}s' .format(time.time()-start_time))
				'''
		columns = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability']
		if label is not None:
			columns.insert(4, 'class')
		result_frame = pd.DataFrame(data=results, columns=columns)
		result_frame.to_csv(result_path+'/result.csv', index=False, float_format='%.4f')
		np.save(result_path+'/result.npy', np.array(results))
		
	print('Overall Detection Done')
