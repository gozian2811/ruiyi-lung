#!/usr/bin/env python
# encoding: utf-8

import os
import shutil
import numpy as np
import pandas as pd
import SimpleITK as sitk
import keras.backend.tensorflow_backend as ktb
from keras.models import load_model
from glob import glob
from toolbox import MITools as mt
from toolbox import CTViewer as cv
from toolbox import CandidateDetection as cd
from toolbox import Lung_Pattern_Segmentation as lps
from toolbox import Lung_Cluster as lc
try:
    from tqdm import tqdm # long waits are not fun
except:
    print('tqdm 是一个轻量级的进度条小包。。。')
    tqdm = lambda x : x


ENVIRONMENT_FILE = "./constants.txt"
IMG_WIDTH, IMG_HEIGHT, NUM_VIEW, MAX_BOUND, MIN_BOUND, PIXEL_MEAN = mt.read_environment(ENVIRONMENT_FILE)
WINDOW_SIZE = min(IMG_WIDTH, IMG_HEIGHT)
NUM_CHANNELS = 3
CANDIDATE_BATCH = 5000

test_paths = ["./TIANCHI_data/train"]
net_file = "./models_keras/tianchi-vgg-2D-v3_solid/tianchi-vgg-2D-v3_solid.h5"
vision_path = "./detection_vision/train"
result_file = "./result5.csv"

all_patients = []
for path in test_paths:
	all_patients += glob(path + "/*.mhd")
if len(all_patients)<=0:
	print("No patient found")
	exit()

if not os.access(vision_path, os.F_OK):
	os.makedirs(vision_path)

#nodules_crop = NodulesCrop("./", "./TIANCHI_data/train/", "./csv_files/train/annotations.csv")

#ktb.set_session(mt.get_session(0.5))
net_model = load_model(net_file)
results = []
for patient in enumerate(tqdm(all_patients)):
	patient = patient[1]
	#patient = "./TIANCHI_data/train/LKDS-00825.mhd"
	uid = mt.get_serie_uid(patient)
	print('Processing patient:%s' %(uid))
	full_image_info = sitk.ReadImage(patient)
	full_scan = sitk.GetArrayFromImage(full_image_info)
	origin = np.array(full_image_info.GetOrigin())[::-1]	#the order of origin and old_spacing is initially [z,y,x]
	old_spacing = np.array(full_image_info.GetSpacing())[::-1]
	image, new_spacing = mt.resample(full_scan, old_spacing)	#resample
	print('Resample Done')

	segmask = lps.segment_lung_mask(image)
	#segmask = segmask[195:200]
	#image = image[195:200]
	segimage = image.copy()
	segimage[segmask==0] = 1024		#segment the lung at the image
	print('Lung Segmentation Done')

	#cluster_labels = lc.seed_segment(image, segmask, cluster_size=3000, view_result=True)
	nodule_matrix, cindex = cd.candidate_detection(segimage)
	cluster_labels = lc.seed_volume_cluster(nodule_matrix, cluster_size=30, result_vision=False)
	print('Clustering Done')
	cluster_labels_merged = lc.cluster_merge(segimage, cluster_labels)
	print('Clusters Merged')
	lc.view_segment(image, cluster_labels_merged)
	candidate_coords = lc.cluster_centers(cluster_labels)
	volume_candidated = cv.view_coordinations(image, candidate_coords, window_size=10, reverse=False, slicewise=True, show=True)
	mt.write_mhd_file(vision_path+"/"+uid+"_candidate.mhd", volume_candidated, volume_candidated.shape[::-1])
	print('Candidate Done')
	exit()
	
	window_prehalf = int(WINDOW_SIZE/2)
	window_afterhalf = WINDOW_SIZE - window_prehalf
	image_padded = MIN_BOUND * np.ones((image.shape[0]+WINDOW_SIZE, image.shape[1]+WINDOW_SIZE, image.shape[2]+WINDOW_SIZE), dtype=int)
	image_padded[window_prehalf:window_prehalf+image.shape[0], window_prehalf:window_prehalf+image.shape[1], window_prehalf:window_prehalf+image.shape[2]] = image
	nodule_centers = []
	print('candidate number:%d' %(len(candidate_coords)))
	for tb in range(0, len(candidate_coords), CANDIDATE_BATCH):
		batch_size = min(CANDIDATE_BATCH, len(candidate_coords)-tb)
		test_data = np.zeros(shape=(batch_size, 4*IMG_HEIGHT, 4*IMG_WIDTH, NUM_CHANNELS), dtype=float)
		for i in range(batch_size):
			coord = candidate_coords[tb+i]
			local_region = image_padded[coord[0]:coord[0]+WINDOW_SIZE, coord[1]:coord[1]+WINDOW_SIZE, coord[2]:coord[2]+WINDOW_SIZE]
			patchs = mt.make_patchs(local_region)
			patch_channels = mt.concatenate_patchs(patchs, NUM_CHANNELS)
			test_data[i] = (patch_channels - MIN_BOUND) / (MAX_BOUND-MIN_BOUND) - PIXEL_MEAN

		predictions = net_model.predict(test_data, batch_size=50)
		for p in range(predictions.shape[0]):
			if predictions[p][0]>predictions[p][1]:
				nodule_centers.append([candidate_coords[p][0], candidate_coords[p][1], candidate_coords[p][2], predictions[p][0]])
				#nodules.append([candidate_coords[p][0], candidate_coords[p][1], candidate_coords[p][2]])
	print('Prediction Done')
	
	nodules = []
	for nc in range(len(nodule_centers)):
		nodules.append(nodule_centers[nc][0:3])
	volume_predicted = cv.view_coordinations(image, nodules, window_size=10, reverse=False, slicewise=True, show=False)
	mt.write_mhd_file(vision_path+"/"+uid+"_prediction.mhd", volume_predicted, volume_predicted.shape[::-1])
	
	nodule_clusters = mt.nodule_cluster(nodule_centers, 40, iterate=True)
	nodules = []
	for nc in range(len(nodule_clusters)):
		nodules.append(np.array(nodule_clusters[nc][0][0:3], dtype=int))
	volume_predicted = cv.view_coordinations(image, nodules, window_size=10, reverse=False, slicewise=True, show=False)
	mt.write_mhd_file(vision_path+"/"+uid+"_prediction_clustered.mhd", volume_predicted, volume_predicted.shape[::-1])

	print('Nodule coordinations:')
	if len(nodule_clusters)<=0:
		print('none')
	for nc in range(len(nodule_clusters)):
		#the output coordination order is [x,y,z], while the order for volume image should be [z,y,x]
		results.append([uid, (nodule_clusters[nc][0][2]*new_spacing[2])+origin[2], (nodule_clusters[nc][0][1]*new_spacing[1])+origin[1], (nodule_clusters[nc][0][0]*new_spacing[0])+origin[0], nodule_clusters[nc][0][3]])
		output_frame = pd.DataFrame(data=results, columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability'])
		output_frame.to_csv(result_file, index=False, float_format='%.4f')
		print('%d %d %d %f' %(nodule_clusters[nc][0][0], nodule_clusters[nc][0][1], nodule_clusters[nc][0][2], nodule_clusters[nc][0][3]))

#sess = ktb.get_session()
#sess.close()
#output_frame = pd.DataFrame(data=results, columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability'])
#output_frame.to_csv(result_file, index=False, float_format='%.4f')
print('Overall Detection Done')
