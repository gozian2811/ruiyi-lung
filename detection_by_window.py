#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd
import SimpleITK as sitk
from keras.models import load_model
from glob import glob
from tqdm import tqdm
from MITools import *
from CTViewer import view_CT


WINDOW_SIZE = 45
SLIDING_STEP = 5
ENVIRONMENT_FILE = "./constants.txt"
IMG_WIDTH, IMG_HEIGHT, NUM_VIEW, MAX_BOUND, MIN_BOUND, PIXEL_MEAN = read_environment(ENVIRONMENT_FILE)

test_path = "./TIANCHI_data/test"
net_file = "./models/tianchi-cnn-2D-v2/tianchi-cnn-2D-v2.h5"


net_model = load_model(net_file)

all_patients = glob(test_path + "/*.mhd")
for patient in enumerate(tqdm(all_patients)):
	patient = patient[1]
	uid = get_serie_uid(patient)
	print('Processing patient:%s' %(uid))
	full_image_info = sitk.ReadImage(patient)
        full_scan = sitk.GetArrayFromImage(full_image_info)
        origin = np.array(full_image_info.GetOrigin())[::-1]	#---获取“体素空间”中结节中心的坐标
        old_spacing = np.array(full_image_info.GetSpacing())[::-1]	#---该CT在“世界空间”中各个方向上相邻单位的体素的间距
	image, new_spacing = resample(full_scan, old_spacing)	#resample
	print('Resample Done')
	
	window_half = WINDOW_SIZE / 2
	nodule_centers = []
	window_mask = np.zeros((int((image.shape[0]-WINDOW_SIZE+1)/SLIDING_STEP), int((image.shape[1]-WINDOW_SIZE+1)/SLIDING_STEP), int((image.shape[2]-WINDOW_SIZE+1)/SLIDING_STEP)), dtype = float)
	for windowz in range(0, image.shape[0]-WINDOW_SIZE, SLIDING_STEP):
		for windowy in range(0, image.shape[1]-WINDOW_SIZE, SLIDING_STEP):
			for windowx in range(0, image.shape[2]-WINDOW_SIZE, SLIDING_STEP):
				local_region = image[windowz:windowz+WINDOW_SIZE, windowy:windowy+WINDOW_SIZE, windowx:windowx+WINDOW_SIZE]
				patchs = make_patchs(local_region)
				patchs = (patchs-MIN_BOUND) / (MAX_BOUND-MIN_BOUND) - PIXEL_MEAN
				patchs = patchs.reshape(patchs.shape[0], patchs.shape[1], patchs.shape[2], 1)
				prediction = net_model.predict(patchs[:][:][:], batch_size=9)
				isnodule = np.prod(prediction)
				notnodule = np.prod(1-prediction)
				window_mask[windowz/SLIDING_STEP, windowy/SLIDING_STEP, windowx/SLIDING_STEP] = isnodule / (isnodule + notnodule)
				if isnodule>notnodule:
					nodule_centers.append([windowz+window_half, windowy+window_half, windowx+window_half, isnodule / (isnodule + notnodule)])
	print('Prediction Done')
	view_CT(window_mask)
	'''
	translations = [[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]]
	while 1:
		hlz, hlx, hly = np.argmax(window_mask)
		prob = window_mask[hlz, hlx, hly]
		if prob<=0.5:
			break
		nodule_centers.append([hlz+window_half, hlx+window_half, hly+window_half, prob])
		cluster = []
		cluster.append([hlz, hlx, hly])
		window_mask[hlz, hlx, hly] = 0.0
		while cluster:
			seed = cluster.pop()
			for t in range(translations.shape[0]):
				neightbor = [seed[0]+translations[t][0], seed[1]+translations[t][1], seed[2]+translations[t][2]]
				if window_mask[neighbor[0], neighbor[1], neighbor[2]]>0.5:
					window_mask[neighbor[0], neighbor[1], neighbor[2]] = 0.0
					cluster.append([neighbor[0], neighbor[1], neighbor[2]])
	print('Clustering Done')
	'''
	
	results = []
	print('Nodule coordinations:')
	for nc in range(len(nodule_centers)):
		results.append([uid, (nodule_centers[nc][2]*new_spacing[2])+origin[2], (nodule_centers[nc][1]*new_spacing[1])+origin[1], (nodule_centers[nc][0]*new_spacing[0])+origin[0], nodule_centers[nc][3]])	#the order of coordinates is adjusted here
		print('%d %d %d %f' %(nodule_centers[nc][0], nodule_centers[nc][1], nodule_centers[nc][2], nodule_centers[nc][3]))

output_frame = pd.DataFrame(data=results, columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability'])
output_frame.to_csv('./result.csv', index=False)
print('Overall Detection Done')
