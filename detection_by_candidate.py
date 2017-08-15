#!/usr/bin/env python
# encoding: utf-8

import os
import shutil
import numpy as np
import pandas as pd
import SimpleITK as sitk
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
	segmask = lps.extend_mask(segmask)
	mbb = lps.mask_boundbox(segmask)
	#mbb.z_top = 195
	#mbb.z_bottom = 200
	#mbb.y_top = 0
	#mbb.y_bottom = image.shape[1] - 1
	#mbb.x_top = 0
	#mbb.x_bottom = image.shape[2] - 1
	image_bounded = image[mbb.z_top:mbb.z_bottom+1, mbb.y_top:mbb.y_bottom+1, mbb.x_top:mbb.x_bottom+1]
	#segimage = image.copy()
	#segimage[segmask==0] = 1024		#segment the lung at the image
	print('Lung Segmentation Done')

	#nodule_matrix, cindex = cd.candidate_detection(segimage)
	#cluster_labels = lc.seed_volume_cluster(nodule_matrix, cluster_size=30, result_vision=False)
	num_segments = int(image_bounded.shape[0] * image_bounded.shape[1] * image_bounded.shape[2] / 27)	#the volume of a 3mm nodule is 27 voxels
	print('cluster number:%d' %(num_segments))
	cluster_labels = 0 - np.ones(shape=image.shape, dtype=int)
	cluster_labels_bounded = lc.slic_segment(image_bounded, num_segments=num_segments)
	print('Clustering Done')
	cluster_labels[mbb.z_top:mbb.z_bottom+1, mbb.y_top:mbb.y_bottom+1, mbb.x_top:mbb.x_bottom+1] = cluster_labels_bounded
	cluster_labels[segmask==0] = -1
	cluster_labels_filtered = lc.cluster_filter(image, cluster_labels)

	#cluster_labels = lc.cluster_merge(segimage, cluster_labels)
	#print('Clusters Merged')
	segresult = lc.segment_vision(image, cluster_labels_filtered)
	np.save(vision_path+"/"+uid+"_segresult.npy", segresult)
	candidate_coords, candidate_labels = lc.cluster_centers(cluster_labels_filtered)
	volume_candidated = cv.view_coordinations(image, candidate_coords, window_size=10, reverse=False, slicewise=True, show=False)
	mt.write_mhd_file(vision_path+"/"+uid+"_candidate.mhd", volume_candidated, volume_candidated.shape[::-1])
	print('Candidate Done')