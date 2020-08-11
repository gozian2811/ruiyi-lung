#!/usr/bin/env python
# encoding: utf-8


"""
@version: python 2.7
@author: Sober.JChen
@license: Apache Licence 
@contact: jzcjedu@foxmail.com
@software: PyCharm
@file: crop_save_and_view_nodules_in_3d.py
@time: 2017/3/14 13:15
"""

# ToDo ---这个脚本运行时请先根据预定义建好文件夹，并将candidates.csv文件的class头改成nodule_class并存为candidates_class.csv，否则会报错。

# ======================================================================
# Program:   Diffusion Weighted MRI Reconstruction
# Link:      https://code.google.com/archive/p/diffusion-mri
# Module:    $RCSfile: mhd_utils.py,v $
# Language:  Python
# Author:    $Author: bjian $
# Date:      $Date: 2008/10/27 05:55:55 $
# Version:
#           $Revision: 1.1 by PJackson 2013/06/06 $
#               Modification: Adapted to 3D
#               Link: https://sites.google.com/site/pjmedphys/tutorials/medical-images-in-python
#
#           $Revision: 2   by RodenLuo 2017/03/12 $
#               Modication: Adapted to LUNA2016 data set for DSB2017
#               Link:https://www.kaggle.com/rodenluo/data-science-bowl-2017/crop-save-and-view-nodules-in-3d
#           $Revision: 3   by Sober.JChen 2017/03/15 $
#               Modication: Adapted to LUNA2016 data set for DSB2017
#               Link:
#-------write_meta_header,dump_raw_data,write_mhd_file------三个函数的由来
# ======================================================================

import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
import time
import math
import scipy
import random
import shutil
import skimage
from glob import glob
from collections import OrderedDict
from toolbox import basicconfig
from toolbox import BasicTools as bt
from toolbox import MITools as mt
from toolbox import SPHTools as st
from toolbox import LIDCTools as lt
from toolbox import CTViewer_Multiax as cvm
from toolbox import Lung_Cluster as lc
from toolbox import CandidateDetection as cd
from toolbox import Nodule_Detection as nd
from toolbox import Ipris as ip
from toolbox.CT_Pattern_Segmentation import segment_lung_mask_fast, extend_mask
try:
	from tqdm import tqdm # long waits are not fun
except:
	print('')
	tqdm = lambda x : x
# import traceback

BOX_SIZE = 64

class CTCrop(object):
	def __init__(self, all_patients_path, output_path='./nodule_cubes/', vision_path=None, volume_sample=True, multiview_sample=False, nodule_vision=False, path_clear=True):
		self.all_patients_path = all_patients_path
		self.output_path = output_path
		self.vision_path = vision_path
		#self.nodules_npy_path = output_path + "/npy/"
		self.nonnodule_npy_path = output_path + "/npy_non/"
		if path_clear and os.access(self.output_path, os.F_OK): shutil.rmtree(self.output_path)
		if not os.access(self.output_path, os.F_OK): os.makedirs(self.output_path)
		if self.vision_path is not None and not os.access(self.vision_path, os.F_OK): os.makedirs(self.vision_path)
		#if not os.access(self.nodules_npy_path, os.F_OK): os.mkdir(self.nodules_npy_path)		#训练用正样本路径
		if not os.access(self.nonnodule_npy_path, os.F_OK): os.mkdir(self.nonnodule_npy_path)	#训练用负样本路径
		if volume_sample:
			self.nodules_npy_path = output_path + "/npy/"
			if not os.access(self.nodules_npy_path, os.F_OK): os.mkdir(self.nodules_npy_path)
		if multiview_sample:
			self.nodules_multiview_path = output_path + "/multiview/"
			if not os.access(self.nodules_multiview_path, os.F_OK): os.mkdir(self.nodules_multiview_path)
		if nodule_vision:
			self.nodules_vision_path = output_path + "/vision/"
			if not os.access(self.nodules_vision_path, os.F_OK): os.mkdir(self.nodules_vision_path)

	def save_annotations_nodule(self, nodule_crop, store_name, mhd_store=False):
		np.save(os.path.join(self.nodules_npy_path, store_name + "_annotation.npy"), nodule_crop)
		if mhd_store:
			mt.write_mhd_file(self.all_annotations_mhd_path + store_name + "_annotation.mhd", nodule_crop, nodule_crop.shape)

	def save_multiview_nodule(self, view_list, store_name, triple_channels=False):
		for i in range(len(view_list)):
			viewimg = np.uint8(mt.medical_normalization(view_list[i], pixel_mean=0) * 255)
			if triple_channels:
				viewimg = np.stack((viewimg, viewimg, viewimg), axis=2)
			skimage.io.imsave(os.path.join(self.nodules_multiview_path, "%s_v%d.png" %(store_name, i+1)), viewimg)

	def save_nonnodule(self, nodule_crop, store_name, mhd_store=False):
		np.save(os.path.join(self.nonnodule_npy_path, store_name + "_nonannotation.npy"), nodule_crop)
		if mhd_store:
			mt.write_mhd_file(self.no_annotation_mhd_path + store_name + "_nonannotation.mhd", nodule_crop, nodule_crop.shape)

	def save_vision_nodule(self, nodule_crop, store_name):
		visimg = mt.volume2image(nodule_crop, BOX_SIZE)
		visimg = mt.medical_normalization(visimg, pixel_mean=0)
		skimage.io.imsave(os.path.join(self.nodules_vision_path, store_name + "_vision.png"), visimg)

	def get_filename(self, file_list, case):
		for f in file_list:
			if str(case) in f:
				return (f)

	def get_ct_constants(self):
		maxvalue = -2000
		minvalue = 2000
		for patient in enumerate(tqdm(self.ls_all_patients)):
			patient = patient[1]
			#print(patient)
			patient_uid = mt.get_serie_uid(patient)
			patient_nodules = self.df_annotations[self.df_annotations.file == patient]
			full_image_info = sitk.ReadImage(patient)
			full_scan = sitk.GetArrayFromImage(full_image_info)
			full_scan[full_scan<basicconfig.config['MIN_BOUND']] = basicconfig.config['MIN_BOUND']
			segimage, segmask, flag = cd.segment_lung_mask(full_scan)
			vmax = full_scan[segmask==1].max()
			vmin = full_scan[segmask==1].min()
			if maxvalue<vmax:
				maxvalue = vmax
				maxfile = patient
			if minvalue>vmin:
				minvalue = vmin
		print("maxvalue:%d minvalue:%d" %(maxvalue, minvalue))
		print("%s" %(maxfile))
		return maxvalue, minvalue, maxfile

class ChallengeCrop(CTCrop):
	def __init__(self, all_patients_path="./sample_patients/", file_ext='mhd', annotations_file=None, candidates_file=None, excludes_file=None, output_path="./nodule_cubes/", vision_path=None, nodule_vision=False, path_clear=True):
		super(ChallengeCrop, self).__init__(all_patients_path, output_path, vision_path, nodule_vision=nodule_vision, path_clear=path_clear)
		self.ls_all_patients = glob(self.all_patients_path + '*.'+file_ext)
		self.df_annotations = None
		self.df_candidates = None
		self.df_excludes = None
		if annotations_file is not None:
			self.df_annotations = pd.read_csv(annotations_file)
			self.df_annotations["file"] = self.df_annotations["seriesuid"].map(lambda file_name: self.get_filename(self.ls_all_patients, file_name))
			self.df_annotations = self.df_annotations.dropna()
		if candidates_file is not None:
			self.df_candidates = pd.read_csv(candidates_file)
			self.df_candidates["file"] = self.df_candidates["seriesuid"].map(lambda file_name: self.get_filename(self.ls_all_patients, file_name))
			self.df_candidates = self.df_candidates.dropna()	
		if excludes_file is not None:
			self.df_excludes = pd.read_csv(excludes_file)
			self.df_excludes["file"] = self.df_excludes["seriesuid"].map(lambda file_name: self.get_filename(self.ls_all_patients, file_name))
			self.df_excludes = self.df_excludes.dropna()	

class NpyCrop(ChallengeCrop):
	def __init__(self, all_patients_path="./sample_patients/", annotations_file=None, candidates_file=None, excludes_file=None, output_path="./nodule_cubes/", vision_path=None, nodule_vision=False, path_clear=True):
		super(NpyCrop, self).__init__(all_patients_path, 'npy', annotations_file, candidates_file, excludes_file, output_path, vision_path, nodule_vision=nodule_vision, path_clear=path_clear)
	
	def annotations_crop(self, start=0, overbound=True):
		if self.df_annotations is None:
			print('no annotation file provided')
			return

		#annotations = []
		#omitted = []
		for pi, patient in enumerate(tqdm(self.ls_all_patients[start:])):
			print(patient)
			# 检查这个病人有没有大于3mm的结节标注
			if patient not in self.df_annotations.file.values:
				print('Patient ' + patient + ' Not exist!')
				continue
			patient_uid = mt.get_mhd_uid(patient)
			#if str(patient_uid)!='407271':
			#	continue
			patient_nodules = self.df_annotations[self.df_annotations.file == patient]
			image = np.load(patient)
			print('resample done')
			#np.save("/data/fyl/datasets/Tianchi_Lung_Disease/train_resampled/%s.npy"%(patient_uid), image)
			#segmask = segment_lung_mask_fast(image)
			#segmask = extend_mask(segmask)
			annodicts = []
			center_coords = []
			diameters = []
			for index, nodule in patient_nodules.iterrows():
				annotation = {}
				annotation['index'] = index
				if hasattr(nodule, 'diameter_mm'):
					annotation['nodule_diameter'] = nodule.diameter_mm
				elif hasattr(nodule, 'diameterX') and  hasattr(nodule, 'diameterY') and hasattr(nodule, 'diameterZ'):
					annotation['nodule_diameter'] = np.rint([nodule.diameterZ, nodule.diameterY, nodule.diameterX]).astype(int)
				diameters.append(annotation['nodule_diameter'])
				if hasattr(nodule, 'label'):
					annotation['label'] = nodule.label
				v_center = np.array([nodule.coordZ, nodule.coordY, nodule.coordX], dtype=int)	#---获取“世界空间”中结节中心的坐标
				annotation['v_center'] = v_center
				annodicts.append(annotation)
				center_coords.append(v_center)
			#volume_regioned = cvm.view_coordinates(image, center_coords, window_size=diameters, reverse=False, slicewise=False, show=False)
			#np.save(self.vision_path+"/"+patient_uid+"_annotated.npy", volume_regioned)
			#---这一系列的if语句是根据“判断一个结节的癌性与否需要结合该结节周边位置的阴影和位置信息”而来，故每个结节都获取了比该结节尺寸略大的3D体素
			#pd.DataFrame(data=annotations, columns=['seriesuid', 'index', 'coordX', 'coordY', 'coordZ']).to_csv(self.output_path+'/annotation.csv', index=False)
			#pd.DataFrame(data=omitted, columns=['seriesuid', 'index', 'coordX', 'coordY', 'coordZ']).to_csv(self.output_path+'/omitted.csv', index=False)

			#get annotations nodule
			if overbound:
				box_size = 2 * BOX_SIZE
			else:
				box_size = BOX_SIZE
			for annotation in annodicts:
				index = annotation['index']
				v_center = annotation['v_center']
				img_crop = mt.local_crop(image, v_center, box_size)
				#img_crop[img_crop>basicconfig.config['MAX_BOUND']] = basicconfig.config['MAX_BOUND']
				img_crop[img_crop<basicconfig.config['MIN_BOUND']] = basicconfig.config['MIN_BOUND']  # ---设置窗宽，小于config.config['MIN_BOUND']的体素值设置为config.config['MIN_BOUND']
				outname = patient_uid + "_" + str(index)
				if 'label' in annotation.keys():
					outname += "_" + str(annotation['label'])
				else:
					outname += "_ob"
				self.save_annotations_nodule(img_crop, outname)
				if hasattr(self, 'nodules_vision_path'):
					self.save_vision_nodule(img_crop, outname)
			print("annotation sampling done")
		print("Done for all!")

	def candidates_crop(self, start=0, region_filter=False, save_positive=False, nodule_exclusion=False):
		if self.df_candidates is None:
			print('no candidate file provided')
			return

		for pi, patient in enumerate(tqdm(self.ls_all_patients[start:])):
			print(patient)
			# 检查这个病人有没有大于3mm的结节
			if patient not in self.df_candidates.file.values:
				print('Patient ' + patient + ' Not exist!')
				continue
			patient_uid = mt.get_mhd_uid(patient)
			image = np.load(patient)
			nodule_candidates = self.df_candidates[self.df_candidates.file == patient]

			candidates = []
			nonnodule_coords = []
			nodule_coords = []
			for index, nodule in nodule_candidates.iterrows():
				if hasattr(nodule, 'class'): nodule_class = nodule.get('class')
				else: nodule_class = 0
				v_center = np.array([nodule.coordZ, nodule.coordY, nodule.coordX], dtype=int)#---获取“世界空间”中结节中心的坐标
				candidates.append([index, nodule_class, v_center])
				if nodule_class==1:
					nodule_coords.append(v_center)
				else:
					nonnodule_coords.append(v_center)
			excludes = []
			if self.df_excludes is not None:
				suspect_excludes = self.df_excludes[self.df_excludes.file == patient]
				for index, nodule in suspect_excludes.iterrows():
					if hasattr(nodule, 'diameter_mm') and nodule.diameter_mm>0:
						nodule_diameter = nodule.diameter_mm
					elif hasattr(nodule, 'diameterX') and  hasattr(nodule, 'diameterY') and hasattr(nodule, 'diameterZ'):
						nodule_diameter = np.rint([nodule.diameterZ, nodule.diameterY, nodule.diameterX]).astype(int)
					else:
						nodule_diameter = 3.0
					#nodule_diameter = nodule.diameter_mm
					#if nodule_diameter < 0: nodule_diameter = 3.0
					v_center = np.array([nodule.coordZ, nodule.coordY, nodule.coordX], dtype=int)#---获取“世界空间”中结节中心的坐标
					exclude = {'index': index, 'nodule_diameter': nodule_diameter, 'v_center': v_center}
					excludes.append(exclude)
			#volume_regioned = cvm.view_coordinates(image, nonnodule_coords, window_size=56, reverse=False, slicewise=True, show=False)
			#np.save(self.vision_path+"/"+patient_uid+"_candidatenonnodule.npy", volume_regioned)
			#volume_regioned = cvm.view_coordinates(image, nodule_coords, window_size=10, reverse=False,
			#					 slicewise=False, show=False)
			#np.save(self.vision_path+"/"+patient_uid+"_candidatenodule.npy", volume_regioned)
			#---这一系列的if语句是根据“判断一个结节的癌性与否需要结合该结节周边位置的阴影和位置信息”而来，故每个结节都获取了比该结节尺寸略大的3D体素

			#get candidates nodule
			if region_filter:
				segmask = segment_lung_mask_fast(image)
			for index, nodule_class, candidate_center in candidates:
				invalid_loc = False
				if nodule_exclusion:
					for exclude in excludes:
						rpos = np.abs(exclude['v_center'] - candidate_center) - exclude['nodule_diameter'] / 2
						if rpos.max()<0:
							invalid_loc = True	#the negative sample is located within the range of positive lesions
							print("candidate of row %d excluded for annotation of row %d" %(index, exclude['index']))
							break
				if region_filter and not segmask[candidate_center[0]][candidate_center[1]][candidate_center[2]]:
					invalid_loc = True
					#print("candidate of row %d out of the lung region" %(index))
				if not invalid_loc:
					#start_time = time.time()
					img_crop = mt.local_crop(image, candidate_center, BOX_SIZE)
					#print("crop time: {}" .format(time.time()-start_time))
					img_crop[img_crop<basicconfig.config['MIN_BOUND']] = basicconfig.config['MIN_BOUND']	#---设置窗宽，小于-1000的体素值设置为-1000
					if nodule_class==0:
						self.save_nonnodule(img_crop, "%s_%d_cc"%(patient_uid, index))
					elif save_positive:
						self.save_annotations_nodule(img_crop, "%s_%d_ob"%(patient_uid, index))
			print("candidate sampling done")
		print('Done for all!')

class MhdCrop(ChallengeCrop):
	def __init__(self, all_patients_path="./sample_patients/", annotations_file=None, candidates_file=None, excludes_file=None, output_path="./nodule_cubes/", vision_path=None, nodule_vision=False, path_clear=True):
		super(MhdCrop, self).__init__(all_patients_path, 'mhd', annotations_file, candidates_file, excludes_file, output_path, vision_path, nodule_vision=nodule_vision, path_clear=path_clear)
		'''
		self.ls_all_patients = glob(self.all_patients_path + "*.mhd")
		self.df_annotations = None
		self.df_candidates = None
		self.df_excludes = None
		if annotations_file is not None:
			self.df_annotations = pd.read_csv(annotations_file)
			self.df_annotations["file"] = self.df_annotations["seriesuid"].map(lambda file_name: self.get_filename(self.ls_all_patients, file_name))
			self.df_annotations = self.df_annotations.dropna()
		if candidates_file is not None:
			self.df_candidates = pd.read_csv(candidates_file)
			self.df_candidates["file"] = self.df_candidates["seriesuid"].map(lambda file_name: self.get_filename(self.ls_all_patients, file_name))
			self.df_candidates = self.df_candidates.dropna()	
		if excludes_file is not None:
			self.df_excludes = pd.read_csv(excludes_file)
			self.df_excludes["file"] = self.df_excludes["seriesuid"].map(lambda file_name: self.get_filename(self.ls_all_patients, file_name))
			self.df_excludes = self.df_excludes.dropna()	
		'''

	def annotations_crop(self, start=0, target_spacing=[1,1,1], annosample='overbound', randsample=False, candsample=False):
		if self.df_annotations is None:
			print('no annotation file provided')
			return

		#annotations = []
		#omitted = []
		if candsample: candidates = []
		for patient in enumerate(tqdm(self.ls_all_patients[start:])):
			patient = patient[1]
			#while patient != './LUNA16/subset6\\1.3.6.1.4.1.14519.5.2.1.6279.6001.167237290696350215427953159586.mhd': continue
			print(patient)
			# 检查这个病人有没有大于3mm的结节标注
			if patient not in self.df_annotations.file.values:
				print('Patient ' + patient + ' not exist in annotation!')
				if not randsample and not candsample: continue
			patient_uid = mt.get_mhd_uid(patient)
			#if str(patient_uid)!='407271':
			#	continue
			patient_nodules = self.df_annotations[self.df_annotations.file == patient]
			full_image_info = sitk.ReadImage(patient)
			full_scan = sitk.GetArrayFromImage(full_image_info)
			origin = np.array(full_image_info.GetOrigin())[::-1]  #---获取“体素空间”中结节中心的坐标
			old_spacing = np.array(full_image_info.GetSpacing())[::-1]  #---该CT在“世界空间”中各个方向上相邻单位的体素的间距
			image, new_spacing = mt.resample(full_scan, old_spacing, np.array(target_spacing))#---重采样
			print('resample done')
			#np.save("/data/fyl/datasets/Tianchi_Lung_Disease/train_resampled/%s.npy"%(patient_uid), image)
			#segmask = segment_lung_mask_fast(image)
			#segmask = extend_mask(segmask)
			annodicts = []
			if self.vision_path is not None:
				center_coords = []
				diameters = []
			for index, nodule in patient_nodules.iterrows():
				annotation = {}
				annotation['index'] = index
				if hasattr(nodule, 'diameter_mm'):
					annotation['nodule_diameter'] = nodule.diameter_mm
				elif hasattr(nodule, 'diameterX') and  hasattr(nodule, 'diameterY') and hasattr(nodule, 'diameterZ'):
					annotation['nodule_diameter'] = np.rint([nodule.diameterZ/new_spacing[0], nodule.diameterY/new_spacing[1], nodule.diameterX/new_spacing[2]]).astype(int)
				if hasattr(nodule, 'label'):
					annotation['label'] = nodule.label
				nodule_center = np.array([nodule.coordZ, nodule.coordY, nodule.coordX])	#---获取“世界空间”中结节中心的坐标
				#v_center = np.rint(np.abs((nodule_center - origin) / new_spacing)).astype(int)	#映射到“体素空间”中的坐标
				v_center = np.int_(np.rint(mt.coord_conversion(nodule_center, origin, old_spacing, full_scan.shape, image.shape, dir_array=True)))
				annotation['v_center'] = v_center
				annodicts.append(annotation)
				if 'diameters' in dir():
					diameters.append(annotation['nodule_diameter'])
				if 'center_coords' in dir():
					center_coords.append(v_center)
				#annorow = [patient_uid, index+1, v_center[2], v_center[1], v_center[0]]
				#annotations.append(annorow)
				#if not segmask[v_center[0]][v_center[1]][v_center[2]]:
				#	print("annotation {}:{} out of the segmentated pulmonary area" .format(patient_uid, index+1))
				#	omitted.append(annorow)
			if self.vision_path is not None:
				volume_regioned = cvm.view_coordinates(image, center_coords, window_size=diameters, reverse=False, slicewise=False, show=False)
				np.save(self.vision_path+"/"+patient_uid+"_annotated.npy", volume_regioned)
			#---这一系列的if语句是根据“判断一个结节的癌性与否需要结合该结节周边位置的阴影和位置信息”而来，故每个结节都获取了比该结节尺寸略大的3D体素
			#pd.DataFrame(data=annotations, columns=['seriesuid', 'index', 'coordX', 'coordY', 'coordZ']).to_csv(self.output_path+'/annotation.csv', index=False)
			#pd.DataFrame(data=omitted, columns=['seriesuid', 'index', 'coordX', 'coordY', 'coordZ']).to_csv(self.output_path+'/omitted.csv', index=False)

			if target_spacing is not None:
				np.save(self.output_path+'/target_spacing.npy', np.array(target_spacing))

			#get annotations nodule
			if annosample=='overbound':
				for annotation in annodicts:
					index = annotation['index']
					v_center = annotation['v_center']
					img_crop = mt.local_crop(image, v_center, 2*BOX_SIZE)
					#img_crop = mt.box_sample(image, v_center, 2*BOX_SIZE)
					'''
					zyx_1 = v_center - BOX_SIZE  # 注意是: Z, Y, X
					zyx_2 = v_center + BOX_SIZE
					if mt.coord_overflow(zyx_1, image.shape) or mt.coord_overflow(zyx_2, image.shape):
						continue
					img_crop = image[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]  # ---截取立方体
					'''
					#img_crop[img_crop>basicconfig.config['MAX_BOUND']] = basicconfig.config['MAX_BOUND']
					img_crop[img_crop<basicconfig.config['MIN_BOUND']] = basicconfig.config['MIN_BOUND']  # ---设置窗宽，小于config.config['MIN_BOUND']的体素值设置为config.config['MIN_BOUND']
					outname = patient_uid + "_" + str(index)
					if 'label' in annotation.keys():
						outname += "_" + str(annotation['label'])
					else:
						outname += "_ob"
					self.save_annotations_nodule(img_crop, outname)
					if hasattr(self, 'nodules_vision_path'):
						self.save_vision_nodule(img_crop, outname)
				print("annotation sampling done")
			elif annosample=='mask':
				mask = np.zeros_like(image)
				for annotation in annodicts:
					v_center = annotation['v_center']
					nodule_diameter = annotation['nodule_diameter']
					label = annotation['label']
					diameter_half = np.int_(nodule_diameter/2)
					bb = v_center - diameter_half
					bt = v_center + nodule_diameter - diameter_half
					mask[bb[0]:bt[0], bb[1]:bt[1], bb[2]:bt[2]][image[bb[0]:bt[0], bb[1]:bt[1], bb[2]:bt[2]]>-600] = label
				outname = patient_uid + "_mk"
				self.save_annotations_nodule(mask, outname)
				print("annotation sampling done")
			elif annosample in ('simple', 'augment'):
				if annosample=='simple':
					scales = [1.0]
					translations = np.array([0,0,0])
				elif annosample=='augment':
					scales = [0.8,1.0,1.25]
					#translations = np.array([[0,0,0],[0,0,1],[0,0,-1],[0,1,0],[0,-1,0],[1,0,0],[-1,0,0]], dtype=float)
					translations = np.array([[0,0,0],[0,0,1],[0,0,-1], [0,1,0],[0,math.sqrt(0.5),math.sqrt(0.5)],[0,math.sqrt(0.5),-math.sqrt(0.5)], [0,-1,0],[0,-math.sqrt(0.5),math.sqrt(0.5)],[0,-math.sqrt(0.5),-math.sqrt(0.5)],
							    [1,0,0],[math.sqrt(0.5),0,math.sqrt(0.5)],[math.sqrt(0.5),0,-math.sqrt(0.5)], [math.sqrt(0.5),math.sqrt(0.5),0],[math.sqrt(0.3333),math.sqrt(0.3333),math.sqrt(0.3333)],[math.sqrt(0.3333),math.sqrt(0.3333),-math.sqrt(0.3333)], [math.sqrt(0.5),-math.sqrt(0.5),0],[math.sqrt(0.3333),-math.sqrt(0.3333),math.sqrt(0.3333)],[math.sqrt(0.3333),-math.sqrt(0.3333),-math.sqrt(0.3333)],
							    [-1,0,0],[-math.sqrt(0.5),0,math.sqrt(0.5)],[-math.sqrt(0.5),0,-math.sqrt(0.5)], [-math.sqrt(0.5),math.sqrt(0.5),0],[-math.sqrt(0.3333),math.sqrt(0.3333),math.sqrt(0.3333)],[-math.sqrt(0.3333),math.sqrt(0.3333),-math.sqrt(0.3333)], [-math.sqrt(0.5),-math.sqrt(0.5),0],[-math.sqrt(0.3333),-math.sqrt(0.3333),math.sqrt(0.3333)],[-math.sqrt(0.3333),-math.sqrt(0.3333),-math.sqrt(0.3333)]])

				num_translations = 3
				for annotation in annodicts:
					index = annotation['index']
					nodule_diameter = annotation['nodule_diameter']
					v_center = annotation['v_center']
					for s in range(len(scales)):
						rt = np.zeros(num_translations, dtype=int)
						rt[1:num_translations] = np.random.choice(range(1,len(translations)), num_translations-1, False)
						rt = np.sort(rt)
						for t in range(rt.size):
							scale = scales[s]
							box_size = int(np.ceil(BOX_SIZE*scale))
							window_size = int(box_size/2)
							translation = np.array(nodule_diameter/2*translations[rt[t]]/new_spacing, dtype=int)
							tnz = translation.nonzero()
							if tnz[0].size==0 and t!=0:
								continue
							zyx_1 = v_center + translation - window_size  # 注意是: Z, Y, X
							zyx_2 = v_center + translation + box_size - window_size
							if mt.coord_overflow(zyx_1, image.shape) or mt.coord_overflow(zyx_2, image.shape):
								continue
							nodule_box = np.zeros([BOX_SIZE, BOX_SIZE, BOX_SIZE], np.int16)  # ---nodule_box_size = 45
							img_crop = image[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]  # ---截取立方体
							img_crop[img_crop<basicconfig.config['MIN_BOUND']] = basicconfig.config['MIN_BOUND']  # ---设置窗宽，小于config.config['MIN_BOUND']的体素值设置为config.config['MIN_BOUND']
							if augmode!='augment' or scale==1.0:
								img_crop_rescaled = img_crop
							else:
								img_crop_rescaled, rescaled_spacing = mt.resample(img_crop, new_spacing, new_spacing*scale)
							try:
								padding_shape = (img_crop_rescaled.shape - np.array([BOX_SIZE, BOX_SIZE, BOX_SIZE])) / 2
								nodule_box = img_crop_rescaled[padding_shape[0]:padding_shape[0]+BOX_SIZE, padding_shape[1]:padding_shape[1]+BOX_SIZE, padding_shape[2]:padding_shape[2]+BOX_SIZE]  # ---将截取的立方体置于nodule_box
							except:
								# f = open("log.txt", 'a')
								# traceback.print_exc(file=f)
								# f.flush()
								# f.close()
								print("annotation error")
								continue
							#nodule_box[nodule_box == 0] = config.config['MIN_BOUND']  # ---将填充的0设置为-1000，可能有极少数的体素由0=>-1000，不过可以忽略不计
							self.save_annotations_nodule(nodule_box, patient_uid+"_"+str(index)+"_"+str(s*rt.size+t))
				print("annotation sampling done")

			window_half = int(BOX_SIZE/2)

			#get candidate annotation nodule
			if candsample:
				'''
				segimage, segmask, flag = cd.segment_lung_mask(image)
				if segimage is not None:
					#nodule_matrix, index = cd.candidate_detection(segimage,flag)
					#cluster_labels = lc.seed_mask_cluster(nodule_matrix, cluster_size=1000)
					cluster_labels = lc.seed_volume_cluster(image, segmask, eliminate_lower_size=-1)
					#lc.cluster_size_vision(cluster_labels)
					candidate_coords, _ = lc.cluster_centers(cluster_labels)
					#candidate_coords = lc.cluster_center_filter(image, candidate_coords)
				'''
				candidate_coords, _, _ = nd.slic_candidate(image, clsize=2e4, focus_area='lung')
				if candidate_coords is not None:
					#the coordination order is [z,y,x]
					print("candidate number:%d" %(len(candidate_coords)))
					#volume_regioned = cv.view_coordinates(image, candidate_coords, window_size=10, reverse=False, slicewise=True, show=False)
					#mt.write_mhd_file(self.vision_path+"/"+patient_uid+"_candidate.mhd", volume_regioned, volume_regioned.shape[::-1])
					for cc in range(len(candidate_coords)):
						candidate_center = candidate_coords[cc]
						invalid_loc = False
						#if mt.coord_overflow(candidate_center-window_half, image.shape) or mt.coord_overflow(candidate_center+BOX_SIZE-window_half, image.shape):
						#	invalid_loc = True
						#	continue
						for annotation in annodicts:
							rpos = np.abs(annotation['v_center'] - candidate_center) - annotation['nodule_diameter'] / 2
							if rpos.max()<0:
								invalid_loc = True	#the negative sample is located within the range of positive lesions
								break
							#if abs(rpos[0])<window_half and abs(rpos[1])<window_half and abs(rpos[2])<window_half:  #the negative sample is located in the positive location
							#	invalid_loc = True
							#	break
						if not invalid_loc:
							candidate = [patient_uid]
							candidate.extend(candidate_center[::-1])
							candidates.append(candidate)
							img_crop = mt.local_crop(image, candidate_center, BOX_SIZE)
							img_crop[img_crop<basicconfig.config['MIN_BOUND']] = basicconfig.config['MIN_BOUND']	#---设置窗宽，小于-1000的体素值设置为-1000
							'''
							zyx_1 = candidate_center - window_half
							zyx_2 = candidate_center + BOX_SIZE - window_half
							nodule_box = np.zeros([BOX_SIZE,BOX_SIZE,BOX_SIZE], np.int16)#---nodule_box_size = 45
							img_crop = image[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]#---截取立方体
							img_crop[img_crop<basicconfig.config['MIN_BOUND']] = basicconfig.config['MIN_BOUND']	#---设置窗宽，小于-1000的体素值设置为-1000
							if img_crop.shape[0]!=BOX_SIZE | img_crop.shape[1]!=BOX_SIZE | img_crop.shape[2]!=BOX_SIZE:
								print("error in resmapleing shape")
							try:
								nodule_box[0:BOX_SIZE, 0:BOX_SIZE, 0:BOX_SIZE] = img_crop  # ---将截取的立方体置于nodule_box
							except:
								print("random error")
								continue
							#nodule_box[nodule_box == 0] = config.config['MIN_BOUND']#---将填充的0设置为-1000，可能有极少数的体素由0=>-1000，不过可以忽略不计
							'''
							self.save_nonnodule(img_crop, "%s_%d_cc"%(patient_uid, cc))
					pd.DataFrame(data=candidates, columns=['seriesuid', 'coordX', 'coordY', 'coordZ']).to_csv(self.output_path+'/candidate.csv', index=False)
					print("candidate sampling done")

			#get random annotation nodule
			if randsample:
				if augmode=='overbound':
					augnum = 100
				elif augment:
					augnum = len(scales) * num_translations
				else:
					augnum = 1
				if augnum*len(annodicts)>len(candidate_coords):
					randnum = augnum*len(annodicts) - len(candidate_coords)
				else:
					randnum = len(candidate_coords)
				for rc in range(randnum):  #the random samples is one-to-one number of nodules
					#index, nodule_diameter, v_center = annodicts[rc]
					rand_center = np.array([0,0,0])  # 注意是: Z, Y, X
					invalid_loc = True
					candidate_overlap = True
					while invalid_loc:
						invalid_loc = False
						candidate_overlap = False
						for axis in range(rand_center.size):
							rand_center[axis] = np.random.randint(0, image.shape[axis])
						if mt.coord_overflow(rand_center-window_half, image.shape) or mt.coord_overflow(rand_center+BOX_SIZE-window_half, image.shape):
							invalid_loc = True
							continue
						if 'segmask' in dir() and not (segmask is None) and not segmask[rand_center[0], rand_center[1], rand_center[2]]:
							invalid_loc = True
							continue
						for annotation in annodicts:
							rpos = annotation['v_center'] - rand_center
							if abs(rpos[0])<window_half and abs(rpos[1])<window_half and abs(rpos[2])<window_half:  #the negative sample is located in the positive location
								invalid_loc = True
								break
						for candidate_coord in candidate_coords:
							rpos = candidate_coord - rand_center
							if abs(rpos[0])<window_half and abs(rpos[1])<window_half and abs(rpos[2])<window_half:  #the negative sample is located in the pre-extracted candidate locations
								candidate_overlap = True
								break
					if candidate_overlap:
						continue
                        
					zyx_1 = rand_center - window_half
					zyx_2 = rand_center + BOX_SIZE - window_half
					nodule_box = np.zeros([BOX_SIZE,BOX_SIZE,BOX_SIZE],np.int16)#---nodule_box_size = 45
					img_crop = image[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]#---截取立方体
					img_crop[img_crop<basicconfig.config['MIN_BOUND']] = basicconfig.config['MIN_BOUND']	#---设置窗宽，小于-1000的体素值设置为-1000
					if img_crop.shape[0]!=BOX_SIZE | img_crop.shape[1]!=BOX_SIZE | img_crop.shape[2]!=BOX_SIZE:
						print("error in resmapleing shape")
					try:
						nodule_box[0:BOX_SIZE, 0:BOX_SIZE, 0:BOX_SIZE] = img_crop  # ---将截取的立方体置于nodule_box
					except:
						# f = open("log.txt", 'a')
						# traceback.print_exc(file=f)
						# f.flush()
						# f.close()
						print("candidate error")
						continue
					#nodule_box[nodule_box == 0] = config.config['MIN_BOUND']#---将填充的0设置为-1000，可能有极少数的体素由0=>-1000，不过可以忽略不计
					self.save_nonnodule(nodule_box, patient_uid+"_rc_"+str(rc))
				print("random sampling done")

			print('Done for this patient!')
		print('Done for all!')

	def candidates_crop(self, target_spacing=[0.5,0.5,0.5], save_positive=False, nodule_exclusion=False):
		if self.df_candidates is None:
			print('no candidate file provided')
			return

		for patient in enumerate(tqdm(self.ls_all_patients[::-1])):
			patient = patient[1]
			#patient = './TIANCHI_data/val/LKDS-00002.mhd'
			#if patient != './datasets/LUNA16/subset6/1.3.6.1.4.1.14519.5.2.1.6279.6001.167237290696350215427953159586.mhd': continue
			print(patient)
			# 检查这个病人有没有大于3mm的结节
			if patient not in self.df_candidates.file.values:
				print('Patient ' + patient + ' Not exist!')
				continue
			patient_uid = mt.get_mhd_uid(patient)
			full_image_info = sitk.ReadImage(patient)
			full_scan = sitk.GetArrayFromImage(full_image_info)
			origin = np.array(full_image_info.GetOrigin())[::-1]  #---获取“体素空间”中结节中心的坐标
			old_spacing = np.array(full_image_info.GetSpacing())[::-1]  #---该CT在“世界空间”中各个方向上相邻单位的体素的间距
			image, new_spacing = mt.resample(full_scan, old_spacing, np.array(target_spacing))#---重采样
			print('resample done')
			
			annotations = []
			nonnodule_coords = []
			nodule_coords = []
			nodule_candidates = self.df_candidates[self.df_candidates.file == patient]
			for index, nodule in nodule_candidates.iterrows():
				if hasattr(nodule, 'class'): nodule_class = nodule.get('class')
				else: nodule_class = 0
				nodule_center = np.array([nodule.coordZ, nodule.coordY, nodule.coordX])#---获取“世界空间”中结节中心的坐标
				#v_center = np.rint(np.abs(nodule_center - origin) / new_spacing).astype(int)#映射到“体素空间”中的坐标
				v_center = np.int_(np.rint(mt.coord_conversion(nodule_center, origin, old_spacing, full_scan.shape, image.shape, dir_array=True)))
				annotations.append([index, nodule_class, v_center])
				if nodule_class==1:
					nodule_coords.append(v_center)
				else:
					nonnodule_coords.append(v_center)
			excludes = []
			if self.df_excludes is not None:
				suspect_excludes = self.df_excludes[self.df_excludes.file == patient]
				for index, nodule in suspect_excludes.iterrows():
					if hasattr(nodule, 'diameter_mm') and nodule.diameter_mm>0:
						nodule_diameter = nodule.diameter_mm
					else:
						nodule_diameter = 3.0
					#nodule_diameter = nodule.diameter_mm
					#if nodule_diameter < 0: nodule_diameter = 3.0
					nodule_center = np.array([nodule.coordZ, nodule.coordY, nodule.coordX])#---获取“世界空间”中结节中心的坐标
					#v_center = np.rint(np.abs((nodule_center - origin) / new_spacing)).astype(int)#映射到“体素空间”中的坐标
					v_center = np.int_(np.rint(mt.coord_conversion(nodule_center, origin, old_spacing, full_scan.shape, image.shape, dir_array=True)))
					excludes.append([index, nodule_diameter, v_center])
			#volume_regioned = cvm.view_coordinates(image, nonnodule_coords, window_size=56, reverse=False, slicewise=True, show=False)
			#np.save(self.vision_path+"/"+patient_uid+"_candidatenonnodule.npy", volume_regioned)
			#volume_regioned = cvm.view_coordinates(image, nodule_coords, window_size=10, reverse=False,
			#					 slicewise=False, show=False)
			#np.save(self.vision_path+"/"+patient_uid+"_candidatenodule.npy", volume_regioned)
			#---这一系列的if语句是根据“判断一个结节的癌性与否需要结合该结节周边位置的阴影和位置信息”而来，故每个结节都获取了比该结节尺寸略大的3D体素

			#get annotations nodule
			window_half = int(BOX_SIZE/2)
			num_translations = 1
			for index, nodule_class, v_center in annotations:
				invalid_loc = False
				if nodule_class==0:
					for s_ind, s_diameter, s_center in excludes:
						dist_thresh = s_diameter / 2.0
						rpos = s_center - v_center
						if abs(rpos[0]) <= dist_thresh and abs(rpos[1]) <= dist_thresh and abs(rpos[2]) <= dist_thresh:
							invalid_loc = True
							print('candidate line {} excluded' .format(index))
							break
					if not invalid_loc and nodule_exclusion:
						for nodule_coord in nodule_coords:
							rpos = nodule_coord - v_center
							if abs(rpos[0]) <= window_half and abs(rpos[1]) <= window_half and abs(rpos[2]) <= window_half:
								# the negative sample is located in the positive location
								invalid_loc = True
								break
				else:
					invalid_loc = True
				if not invalid_loc:
					zyx_1 = v_center - window_half  # 注意是: Z, Y, X
					zyx_2 = v_center + BOX_SIZE - window_half
					if mt.coord_overflow(zyx_1, image.shape) or mt.coord_overflow(zyx_2, image.shape):
						continue
					nodule_box = np.zeros([BOX_SIZE, BOX_SIZE, BOX_SIZE], np.int16)  # ---nodule_box_size = 45
					img_crop = image[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]  # ---截取立方体
					img_crop[img_crop<basicconfig.config['MIN_BOUND']] = basicconfig.config['MIN_BOUND']  # ---设置窗宽，小于config.config['MIN_BOUND']的体素值设置为config.config['MIN_BOUND']
					try:
						nodule_box = img_crop[0:BOX_SIZE, 0:BOX_SIZE, 0:BOX_SIZE]  # ---将截取的立方体置于nodule_box
					except:
						print("candidate error")
						continue
					if nodule_class==0:
						self.save_nonnodule(nodule_box, patient_uid+"_"+str(index)+"_cc")
					elif save_positive:
						self.save_annotations_nodule(nodule_box, patient_uid+"_"+str(index)+"_ob")
			print('Done for this patient!')
		print('Done for all!')

class LIDCCrop(CTCrop):
	def __init__(self, all_patients_path="./DOI/", output_path="./lidc_cubes_64_overbound/", vision_path=None, volume_sample=True, multiview_sample=False, slicewise_sample=False, statistics=False, nodule_vision=False, path_clear=True, ipris=False):
		super(LIDCCrop, self).__init__(all_patients_path, output_path, vision_path, volume_sample, multiview_sample, nodule_vision, path_clear)
		if slicewise_sample:
			self.nodules_slicewise_path = output_path + "/slicewise/"
			if not os.access(self.nodules_slicewise_path, os.F_OK): os.mkdir(self.nodules_slicewise_path)
		if statistics:
			self.nodules_statistics_file = self.output_path + "/statistics.csv"
		if ipris:
			self.nodules_ipris_path = output_path + "/ipris/"
		self.ls_all_patients = lt.retrieve_scan(self.all_patients_path)
		self.error_patients = []

	def save_slicewise_nodule(self, volume, mask, store_name, image_size=None, texture_crop=False, triple_channels=False):
		box_center, box_shape = mt.mask_box(mask, square=True, overbound=True)
		box_shape[0] = min(box_shape[0], 3)
		nodule_crop = mt.local_crop(volume, box_center, box_shape)
		mask_crop = mt.local_crop(mask, box_center, box_shape)
		nodule_crop[nodule_crop>basicconfig.config['MAX_BOUND']] = basicconfig.config['MAX_BOUND']
		nodule_crop[nodule_crop<basicconfig.config['MIN_BOUND']] = basicconfig.config['MIN_BOUND']
		for i in range(len(nodule_crop)):
			#slice = mt.medical_normalization(volume[i], pixel_mean=0)
			imgslice = nodule_crop[i]
			maskslice = mask_crop[i]
			threshslice = maskslice * (basicconfig.config['MAX_BOUND'] - basicconfig.config['MIN_BOUND']) + basicconfig.config['MIN_BOUND']
			shapeslice = np.maximum(imgslice, threshslice)
			textureslice = np.minimum(imgslice, threshslice)
			if image_size is not None:
				resize = np.array((image_size, image_size)) / np.array(nodule_crop.shape, dtype=float)[1:]
				imgslice = scipy.ndimage.interpolation.zoom(imgslice, resize, mode='nearest')
				shapeslice = scipy.ndimage.interpolation.zoom(shapeslice, resize, mode='nearest')
				if texture_crop and np.min(textureslice.shape)>=32:
					#If the ROI is larger than 16*16, a 16*16 patch that contains the maximum nodule voxels is extracted.
					mask_coords = np.array(maskslice.nonzero())
					if mask_coords.shape[1]>0:
						slice_center = np.int_(mask_coords.mean(axis=1)+0.5)
						#slice_center = np.int_(np.array(textureslice.shape)/2)
						size_half = np.concatenate((slice_center, np.array(textureslice.shape)-slice_center, np.array([16]))).min()
						textureslice = textureslice[slice_center[0]-size_half:slice_center[0]+size_half, slice_center[1]-size_half:slice_center[1]+size_half]
						resize = np.array((image_size, image_size)) / np.array(textureslice.shape, dtype=float)
				textureslice = scipy.ndimage.interpolation.zoom(textureslice, resize, mode='nearest')
			imgslice = np.uint8(mt.medical_normalization(imgslice, pixel_mean=0) * 255)
			shapeslice = np.uint8(mt.medical_normalization(shapeslice, pixel_mean=0) * 255)
			textureslice = np.uint8(mt.medical_normalization(textureslice, pixel_mean=0) * 255)
			if triple_channels:
				imgslice = np.stack((imgslice, imgslice, imgslice), axis=2)
				shapeslice = np.stack((shapeslice, shapeslice, shapeslice), axis=2)
				textureslice = np.stack((textureslice, textureslice, textureslice), axis=2)
			skimage.io.imsave(os.path.join(self.nodules_slicewise_path, "%s_oa_s%d.png" %(store_name, i+1)), imgslice)
			skimage.io.imsave(os.path.join(self.nodules_slicewise_path, "%s_hs_s%d.png" %(store_name, i+1)), shapeslice)
			skimage.io.imsave(os.path.join(self.nodules_slicewise_path, "%s_hvv_s%d.png" %(store_name, i+1)), textureslice)
		
	def mask_sample(self, start = 0):
		self.mask_path = self.output_path + "/masks"
		if not os.access(self.mask_path, os.F_OK): os.mkdir(self.mask_path)
		for patient in enumerate(tqdm(self.ls_all_patients[start:])):
			patient = patient[1]
			print(patient)
			if self.error_patients.count(patient)>0:
				print('Data error, ignored!')
				continue
			series_id = patient.split('/')[-1]
			nodules, nonnodules = lt.parseXML(patient)
			transed_nodule_list, image, spacing = lt.coord_trans(nodules, patient)
			filled_nodule_list = lt.fill_hole(transed_nodule_list, image.shape[0], image.shape[1], image.shape[2])
			#if expert_wise:
			#	nodule_list = filled_nodule_list
			#else:
			#	filled_nodule_list = lt.fuse_nodules(filled_nodule_list)
			#	nodule_list = lt.calc_union_freq(filled_nodule_list)
			mask = np.zeros_like(image)
			for nodule_info in filled_nodule_list:
				mask[nodule_info['mask']!=0] = 1
			np.save(self.mask_path + '/' + series_id, mask)
		
	def pathology_crop(self, target_spacing=None, expert_wise=False, start=0):
		#target_spacing = np.array(target_spacing)
		if hasattr(self, 'nodules_ipris_path'):
			os.mkdir(self.nodules_ipris_path)

		#characterstats = {}
		#num_nonnodules = 0
		num_annotations_pn = {}
		if hasattr(self, 'nodules_statistics_file'):
			statlabels = ['patient_id', 'nodule_id', 'subtlety', 'internalStructure', 'calcification', 'sphericity', 'margin', 'lobulation', 'spiculation', 'texture', 'malignancy', 'diameter']
			statistics = []
		patient_counts = {}
		#noduleanno_count = 0
		for patient in enumerate(tqdm(self.ls_all_patients[start:])):
			patient = patient[1]
			#patient = "/data/fyl/datasets/DOI/LIDC-IDRI-0078/1.3.6.1.4.1.14519.5.2.1.6279.6001.339170810277323131167631068432/1.3.6.1.4.1.14519.5.2.1.6279.6001.303494235102183795724852353824"
			#patient= "/data/fyl/datasets/DOI/LIDC-IDRI-0141/1.3.6.1.4.1.14519.5.2.1.6279.6001.160881090830720390023167668360/1.3.6.1.4.1.14519.5.2.1.6279.6001.267957701183569638795986183786"
			#patient = "/data/fyl/datasets/DOI/LIDC-IDRI-0267/1.3.6.1.4.1.14519.5.2.1.6279.6001.324984193116544130562864675468/1.3.6.1.4.1.14519.5.2.1.6279.6001.245181799370098278918756923992"
			#patient = "/data/fyl/datasets/DOI/LIDC-IDRI-0857/1.3.6.1.4.1.14519.5.2.1.6279.6001.215559453287121684893831546044/1.3.6.1.4.1.14519.5.2.1.6279.6001.401389720232123950202941034290"
			print(patient)
			if self.error_patients.count(patient)>0:
				print('Data error, ignored!')
				continue
			patient_id = patient.split('/')[-3]
			if patient_id not in patient_counts.keys():
				patient_counts[patient_id] = 0
			else:
				patient_counts[patient_id] += 1
				patient_id += '-' + str(patient_counts[patient_id]+1)
			nodules, nonnodules = lt.parseXML(patient)
			'''
			num_nonnodules += len(nonnodules)
			for nodule in nodules:
				for charkey in nodule['characteristics'].keys():
					if charkey not in characterstats.keys():
						characterstats[charkey] = nodule['characteristics'][charkey]
					else:
						characterstats[charkey] = max(characterstats[charkey], nodule['characteristics'][charkey])
			'''
			#time_step = time.time()
			transed_nodule_list, image, spacing = lt.coord_trans(nodules, patient)
			#transed_nodule_list = lt.duplicate_nodules(transed_nodule_list)
			#print('nodule translation time:{}' .format(time.time()-time_step))
			filled_nodule_list = lt.fill_hole(transed_nodule_list, image.shape[0], image.shape[1], image.shape[2])
			#print('nodule fill time:{}' .format(time.time()-time_step))
			if expert_wise:
				nodule_center_list = lt.nodule_centers(filled_nodule_list)
			else:
				filled_nodule_list = lt.fuse_nodules(filled_nodule_list)
				#print('nodule fuse time:{}' .format(time.time()-time_step))
				union_nodule_list = lt.calc_union_freq(filled_nodule_list)
				nodule_center_list = lt.nodule_centers(union_nodule_list)
				
				for union_nodule in union_nodule_list:
					numanno = union_nodule['radiologist']
					if numanno>4:
						print("more than 4 annotations on a nodule")
					if str(numanno) not in num_annotations_pn.keys():
						num_annotations_pn[str(numanno)] = 1
					else:
						num_annotations_pn[str(numanno)] += 1
				
			#print('nodule center time:{}' .format(time.time()-time_step))

			#box_shape = np.array([BOX_SIZE, BOX_SIZE, BOX_SIZE])
			if target_spacing is not None:
				#box_shape = np.int_(np.ceil(box_shape / spacing * target_spacing))
				np.save(self.output_path+'/target_spacing.npy', np.array(target_spacing))

			#image_padded = np.pad(image, ((box_shape[0], box_shape[0]), (box_shape[1], box_shape[1]), (box_shape[2], box_shape[2])), 'constant', constant_values = ((basicconfig.config['MIN_BOUND'], basicconfig.config['MIN_BOUND']), (basicconfig.config['MIN_BOUND'], basicconfig.config['MIN_BOUND']), (basicconfig.config['MIN_BOUND'], basicconfig.config['MIN_BOUND'])))
			for nodule_center in nodule_center_list:
				nodule_id = nodule_center['nodule_id']
				'''
				v_center = np.array(nodule_center['coord'])
				zyx_1 = v_center
				zyx_2 = v_center + 2 * box_shape
				img_crop = image_padded[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1],
					   zyx_1[2]:zyx_2[2]]  # ---截取立方体
				#img_crop[img_crop < -1000] = -1000  # ---设置窗宽，小于-1000的体素值设置为-1000
				if target_spacing is not None:
					img_crop, _ = mt.resample(img_crop, spacing, np.array(target_spacing))
				try:
					nodule_box = img_crop[int(img_crop.shape[0]/2-BOX_SIZE):int(img_crop.shape[0]/2+BOX_SIZE), int(img_crop.shape[1]/2-BOX_SIZE):int(img_crop.shape[1]/2+BOX_SIZE), int(img_crop.shape[2]/2-BOX_SIZE):int(img_crop.shape[2]/2+BOX_SIZE)]  # ---将截取的立方体置于nodule_box
				except:
					print("annotation error")
					continue
				# nodule_box[nodule_box == 0] = config.config['MIN_BOUND']  # ---将填充的0设置为-1000，可能有极少数的体素由0=>-1000，不过可以忽略不计
				'''
				outname = patient_id + "_" + str(nodule_id) + "_" + str(nodule_center['characteristics']['malignancy'])
				if expert_wise: outname += str(random.random()*0.5)[1:4]
				if target_spacing is None:
					outname += "_" + str(spacing)
					resize = None
				else:
					resize = spacing/np.float_(target_spacing)
				nodule_box = mt.box_sample(image, nodule_center['coord'], 2*BOX_SIZE, resize)
				if hasattr(self, 'nodules_npy_path'):
					self.save_annotations_nodule(nodule_box, outname)
				if hasattr(self, 'nodules_multiview_path'):
					if int(float(nodule_center['characteristics']['malignancy'])+0.5)>3:
						splitnum = 7
					elif int(float(nodule_center['characteristics']['malignancy'])+0.5)<3:
						splitnum = 5
					if 'splitnum' in dir():
						view_list = mt.spherical_sample(image, nodule_center['coord'], spacing, splitnum, 50, 224)
						self.save_multiview_nodule(view_list, outname, triple_channels=False)
				if hasattr(self, 'nodules_slicewise_path'):
					self.save_slicewise_nodule(image, nodule_center['mask'], outname, image_size=360, texture_crop=False, triple_channels=False)
					'''
					nodule_data = mt.medical_normalization(nodule_box, pixel_mean=0)
					nodule_mask = mt.box_sample(nodule_center['mask'], nodule_center['coord'], 2*BOX_SIZE, spacing/target_spacing.astype(float))
					nodule_shape = nodule_data * nodule_mask
					nodule_texture = 1 - (1 - nodule_data) * (1 - nodule_mask)
					nodule_box_cropped = mt.mask_crop(nodule_data, nodule_mask)
					nodule_shape_cropped = mt.mask_crop(nodule_shape, nodule_mask)
					nodule_texture_cropped = mt.mask_crop(nodule_texture, nodule_mask)
					self.save_slicewise_nodule(nodule_box_cropped, outname+'_oa', image_size=200, triple_channels=True)
					self.save_slicewise_nodule(nodule_shape_cropped, outname+'_hs', image_size=200, triple_channels=True)
					self.save_slicewise_nodule(nodule_texture_cropped, outname+'_hvv', image_size=200, triple_channels=True)
					'''
				if hasattr(self, 'nodules_statistics_file'):
					box_center, box_shape = mt.mask_box(nodule_center['mask'], square=True, overbound=False)
					diameter = max((box_shape[0]+1)*spacing[0], box_shape[-1]*spacing[-1])
					statistic = [patient_id, nodule_id]
					for label in statlabels[2:-1]:
						statistic.append(nodule_center['characteristics'][label])
					statistic.append(diameter)
					statistics.append(statistic)
					csv_frame = pd.DataFrame(data=statistics, columns=statlabels)
					csv_frame.to_csv(self.nodules_statistics_file, index=False)
				if hasattr(self, 'nodules_vision_path'):
					self.save_vision_nodule(nodule_box, outname)
				if hasattr(self, 'nodules_ipris_path'):
					for union_nodule in union_nodule_list:
						if union_nodule['nodule_id'] == nodule_center['nodule_id']:
							ipris_feature_list = []
							for mi in range(len(union_nodule['masks'])):
								mask = union_nodule['masks'][mi]
								iprisfeature = ip.ipris_feature(image, mask)
								ipris_feature_list.append(iprisfeature)
								#mask_padded = np.pad(mask, ((box_shape[0], box_shape[0]), (box_shape[1], box_shape[1]), (box_shape[2], box_shape[2])), #'minimum')
								#mask_crop = mask_padded[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]
								#if target_spacing is not None:
								#	mask_crop, _ = mt.resample(mask_crop, spacing, target_spacing)
								#mask_box = mask_crop[int(mask_crop.shape[0]/2-BOX_SIZE):int(mask_crop.shape[0]/2+BOX_SIZE), #int(mask_crop.shape[1]/2-BOX_SIZE):int(mask_crop.shape[1]/2+BOX_SIZE), #int(mask_crop.shape[2]/2-BOX_SIZE):int(mask_crop.shape[2]/2+BOX_SIZE)]
								#maskoutname = outname + "_m" + str(mi+1)
								#np.save(os.path.join(self.nodules_mask_path, maskoutname + "_mask.npy"), mask_box)
							np.save(os.path.join(self.nodules_ipris_path, outname + "_ipris.npy"), np.array(ipris_feature_list))
							break
				
		#print(noduleanno_count)
		print(num_annotations_pn)

class SPHCrop(CTCrop):
	def __init__(self, all_patients_path="./SPH_data/", annotations_file="./csv_files/annotations.csv", diameters_file=None, output_path="./nodule_cubes/", vision_path=None, volume_sample=True, multiview_sample=False, slicewise_sample=False, nodule_vision=False, path_clear=True, version=2):
		super(SPHCrop, self).__init__(all_patients_path, output_path, vision_path, volume_sample, multiview_sample, nodule_vision, path_clear)
		if slicewise_sample:
			if diameters_file!=None:
				self.df_diameters = pd.read_excel(diameters_file)
				self.nodules_slicewise_path = output_path + "/slicewise/"
				if not os.access(self.nodules_slicewise_path, os.F_OK): os.mkdir(self.nodules_slicewise_path)
			else:
				print("no diameter information for slices sampling.")
		#self.nodules_npy_path = output_path + "npy/"
		#self.nonnodule_npy_path = output_path + "npy_non/"
		self.df_annotations = pd.read_excel(annotations_file)
		self.excludes = ["/data/fyl/datasets/SPH_data/2015/20180719/800351034 519"]
		self.version = version
		self.diagnosises = ["AAH", "AIS", "MIA"]
		if version==1: self.annotation_columns = ["anno1", "anno2", "anno3", "anno4", "anno5", "anno6", "anno7", "anno8", "anno9", "anno10", "anno11", "anno12"]
		elif version==2: self.annotation_columns = ["X1", "Y1", "Z1", "X2", "Y2", "Z2", "X3", "Y3", "Z3", "X4", "Y4", "Z4", "X5", "Y5", "Z5", "X6", "Y6", "Z6", "X7", "Y7", "Z7", "X8", "Y8", "Z8", "X9", "Y9", "Z9", "X10", "Y10", "Z10", "X11", "Y11", "Z11"]
		#elif version==2: self.annotation_columns = [["X1", "Y1", "Z1"], ["X2", "Y2", "Z2"], ["X3", "Y3", "Z3"], ["X4", "Y4", "Z4"], ["X5", "Y5", "Z5"], ["X6", "Y6", "Z6"], ["X7", "Y7", "Z7"], ["X8", "Y8", "Z8"], ["X9", "Y9", "Z9"], ["X10", "Y10", "Z10"]]
		self.ls_all_patients = []
		time_packages = bt.get_dirs(self.all_patients_path)
		for package in time_packages:
			self.ls_all_patients.extend(bt.get_dirs(package))
		#self.ls_all_patients = ["/data/fyl/datasets/SPH_data/2016/20170714/k01395919", "/data/fyl/datasets/SPH_data/2015/20180713/K01395919 169"]
		'''
		annocompcolumns = ["serie_id", "order", "diagnosis"]
		annocompcolumns.extend(self.annotation_columns)
		annotations_compensated = []
		for pi, patient in enumerate(tqdm(self.ls_all_patients)):
			patient_infos = patient.split('/')[-1].split(' ')
			patient_uid = patient_infos[0]
			serie_order = int(patient_infos[1])
			if serie_order=='199' or serie_order=='331':
				print(serie_order)
			#_, _, patient_uid = mt.read_sph_scan(patient)
			for annoidx in np.where(self.df_annotations.order==serie_order)[0]:
				annocomp = []
				annocomp.append(patient_uid)
				annocomp.append(serie_order)
				annocomp.append(self.df_annotations.diagnosis[annoidx])
				for annocol in self.annotation_columns:
					annocomp.append(self.df_annotations[annocol][annoidx])
				annotations_compensated.append(annocomp)
			pd.DataFrame(data=annotations_compensated, columns=annocompcolumns).to_excel("MIAannotations2014.xlsx", index=False)
		exit()
		'''
		self.serie_ids = self.df_annotations.serie_id.tolist()
		for sidx in range(len(self.serie_ids)):
			if type(self.serie_ids[sidx])!=str:
				self.serie_ids[sidx] = str(self.serie_ids[sidx])
			self.serie_ids[sidx] = self.serie_ids[sidx].upper()
			
	def save_slicewise_nodule(self, volume, center, slice_size, store_name, image_size=None, overbound=True, triple_channels=False):
		nodule_crop = mt.local_crop(volume, center, (3, slice_size*(1+int(overbound)), slice_size*(1+int(overbound))))	#double the slice size for overbound
		#nodule_crop[nodule_crop>basicconfig.config['MAX_BOUND']] = basicconfig.config['MAX_BOUND']
		#nodule_crop[nodule_crop<basicconfig.config['MIN_BOUND']] = basicconfig.config['MIN_BOUND']
		for i in range(len(nodule_crop)):
			imgslice = nodule_crop[i]
			if image_size is not None:
				resize = np.array((image_size, image_size)) / np.array(nodule_crop.shape, dtype=float)[1:]
				imgslice = scipy.ndimage.interpolation.zoom(imgslice, resize, mode='nearest')
			imgslice = np.uint8(mt.medical_normalization(imgslice, pixel_mean=0) * 255)
			if triple_channels:
				imgslice = np.stack((imgslice, imgslice, imgslice), axis=2)
			skimage.io.imsave(os.path.join(self.nodules_slicewise_path, "%s_oa_s%d.png" %(store_name, i+1)), imgslice)


	def get_annotation(self, patient_uid, exceptomit=True):
		annotations = []
		exception = ""
		if self.version==1:
			#in version 1 the information including numbers are all stored as the type of string.
			rowidcs = np.where(np.array(self.serie_ids)==patient_uid)[0]
			for rowidx in rowidcs:
				#rowidx = self.serie_ids.index(patient_uid)
				if 'diagnosis' in self.df_annotations.keys(): diagnosis = self.df_annotations['diagnosis'][rowidx]
				for annocol in self.annotation_columns:
					if annocol in self.df_annotations.keys():
						annostr = self.df_annotations.get(annocol)[rowidx]
						'''
						if type(annostr)==str:
							if annostr.find('*')>=0:
								continue
							coordbegin = -1
							coordend = -1
							for ci in range(len(annostr)):
								if coordbegin<0:
									if annostr[ci]>='0' and annostr[ci]<='9':
										coordbegin = ci
								elif (annostr[ci]<'0' or annostr[ci]>'9') and annostr[ci]!=' ':
									coordend = ci
									break
							if coordbegin>=0:
								# annotation = np.array(annostr.split('（')[0].split(' '), dtype=int)
								if coordend<0:
									coordend = len(annostr)
								coordstr = annostr[coordbegin:coordend]
								annotation = np.array(coordstr.split(' '), dtype=int)
								annotations.append([annocol, annotation])  # the index order is [x,y,z]
						'''
						annotation = st.annostr2coord(annostr, exceptomit)
						if annotation is not None:
							annotations.append([annocol, annotation])  # the index order is [x,y,z]
		elif self.version==1.1:
			#in version 1.1 each row in the annotation file indicates a coordinate, where an identical serie_uid may correspond with multiple rows.
			siarray = np.array(self.serie_ids)
			rowidcs = np.where(siarray==patient_uid)[0]
			for rowidx in rowidcs:
				coordstr = self.df_annotations.get('coord_XYZ')[rowidx]
				coordinate = st.annostr2coord(coordstr, exceptomit)
				diagnosis = self.df_annotations['diagnosis'][rowidx]
				annotations.append([rowidx+1, diagnosis, coordinate])
		elif self.version==2:
			#in version 2 the information with numbers are all stored as the type of integer or float.
			rowidcs = np.where(np.array(self.serie_ids)==patient_uid)[0]
			annoid = 0
			for rowidx in rowidcs:
				#possibly multiple rows correspond to a single uid of a patient
				#rowidx = self.serie_ids.index(patient_uid)
				diagnosis = self.df_annotations['diagnosis'][rowidx]
				diaglist = diagnosis.split('+')
				if '*' in diagnosis:
					diagtranslist = []
					for diag in diaglist:
						if '*' in diag:
							multdiag = diag.split('*')
							if len(multdiag)==2:
								diagtranslist.extend([multdiag[0] for i in range(int(multdiag[1]))])
							else:
								print("exceptional diagnosis multiplication: row{} {}" .format(rowidx, diag))
						else:
							diagtranslist.append(diag)
					diaglist = diagtranslist
				coords = []
				coordtuples = []
				extradiagnosises = []
				for annocol in self.annotation_columns:
					if annocol in self.df_annotations.keys():
						coord = self.df_annotations.get(annocol)[rowidx]
						if not (type(coord)!=str and math.isnan(coord)):
							coords.append(coord)
				for cidx in range(len(coords)):
					if type(coords[cidx])!=str:
						if cidx%3==0:
							coordtuples.append([])
						coordtuples[-1].append(coords[cidx])	#possible to get incomplete xyz coordinates.
					else:
						extradiagnosises.append(coords[cidx])
				if len(extradiagnosises)>0 and '~' in extradiagnosises[0]:
					#strongly accept this annotation
					for aidx in range(len(coordtuples)):
						annotations.append([annoid+1, diaglist[aidx], np.array(coordtuples[aidx])])
						annoid += 1
				else:
					if len(extradiagnosises)>0: diaglist = extradiagnosises
					if not exceptomit:
						for aidx in range(len(coordtuples)):
							annotations.append([annoid+1, np.array(coordtuples[aidx])])
							annoid += 1
					elif '#' in diaglist:
						#we mark '#' for each row as a ignorance of this serie of annotation
						exception += "annotation eliminated: row{} " .format(rowidx)
						print(exception)
					elif len(coordtuples)==0:
						exception += "no coordinates for this patient found: row{} " .format(rowidx)
						print(exception)
					elif len(coordtuples[-1])!=3:
						exception += "coordinates mess: row{} " .format(rowidx)
						print(exception)
					elif len(coordtuples)!=len(diaglist):
						exception += "incorrepsondence between diagnosis and coordinates: row{} " .format(rowidx)
						print(exception)
					elif len(extradiagnosises)==0 and len(diaglist)>1:
						exception += "more than 1 annotation in diagnosis column, ignore them temporarilty: row{} " .format(rowidx)
						print(exception)
					else:
						for aidx in range(len(coordtuples)):
							if '#' in diaglist[aidx]:
								#the information text along with '#' in a cell is regarded as a useless note
								exception += "annotation eliminated: row{} " .format(rowidx)
								print(exception)
								break
							else:
								annotations.append([annoid+1, diaglist[aidx], np.array(coordtuples[aidx])])
								annoid += 1
				
		return annotations, exception
		
	def annotations_statistics(self, fast=False, start=0):
		noanno_patient_uids = OrderedDict()
		nocoord_patients = []
		patient_uids = OrderedDict()
		anno_uids = OrderedDict()
		#sdict1 = bt.dictionary_load('sph2016_cubes_64_overbound/nodstats.log')
		#sdict2 = bt.dictionary_load('sph2016_cubes_64_overbound2/nodstats.log')
		#for key in sdict2.keys():
		#	if sdict1[key] != sdict2[key]: print(key)
		#slisttemp = bt.filelist_load('sph2016_cubes_64_overbound2/nodstats.log')[:-1]
		#for st in range(len(slisttemp)):
		#	slisttemp[st] = slisttemp[st][:-2]
		#set1 = set(slisttemp)
		#set2 = set(self.serie_ids)
		#self.ls_all_patients.index('/data/fyl/datasets/SPH_data/2015/20180719/800351034 519')
		for pi, patient in enumerate(tqdm(self.ls_all_patients[start:])):
			#print(patient)
			if fast:
				patient_uid = mt.read_sph_id(patient)
			else:
				_, _, patient_uid = mt.read_sph_scan(patient)
			patient_uid = patient_uid.upper()
			if self.serie_ids.count(patient_uid)==0:
				if patient_uid[0]=='0' and self.serie_ids.count(patient_uid[1:])>0:
					patient_uid = patient_uid[1:]
				else:
					#print('no annotation for this patient found')
					if patient_uid not in noanno_patient_uids.keys():
						noanno_patient_uids[patient_uid] = 1
					else:
						noanno_patient_uids[patient_uid] += 1
					continue
			#if patient_uid in ('800326548', '800351034'):
			#	print(pi, patient)
			if patient_uid not in patient_uids.keys():
				patient_uids[patient_uid] = 1
			else:
				patient_uids[patient_uid] += 1
			patient_nodules, _ = self.get_annotation(patient_uid, exceptomit=True)
			if len(patient_nodules)>0:
				if patient_uid not in anno_uids:
					anno_uids[patient_uid] = len(patient_nodules)
				elif anno_uids[patient_uid] != len(patient_nodules):
					print(patient_uid + ' annotation number exception')
					anno_uids[patient_uid] = max(anno_uids[patient_uid], len(patient_nodules))
			elif patient_uid not in nocoord_patients:
				nocoord_patients.append(patient_uid)
		bt.filelist_store(nocoord_patients, self.output_path + '/nocoordstats.log')
		
		nonstats = open(self.output_path + '/noannostats.log', 'w')
		count = 0
		for uid in noanno_patient_uids.keys():
			nonstats.write("{}:{}\n" .format(uid, noanno_patient_uids[uid]))
			count += noanno_patient_uids[uid]
		nonstats.write("total count:{}\n" .format(count))
		nonstats.close()
		stats = open(self.output_path + '/datastats.log', 'w')
		count = 0
		for uid in patient_uids.keys():
			stats.write("{}:{}\n" .format(uid, patient_uids[uid]))
			count += patient_uids[uid]
		stats.write("total count:{}\n" .format(count))
		stats.close()
		annstats = open(self.output_path + '/nodstats.log', 'w')
		count = 0
		for uid in anno_uids.keys():
			annstats.write("{}:{}\n" .format(uid, anno_uids[uid]))
			count += anno_uids[uid]
		annstats.write("total count:{}\n" .format(count))
		annstats.close()
	
	def annotations_crop(self, target_spacing=None, overbound=True, candsample=False, start=0):
		'''
		if os.access(self.output_path, os.F_OK):
			shutil.rmtree(self.output_path)
		os.makedirs(self.output_path)
		os.mkdir(self.nodules_npy_path)
		os.mkdir(self.nonnodule_npy_path)
		'''

		exceptions = []
		for patient in enumerate(tqdm(self.ls_all_patients[start:])):
			patient = patient[1]
			#patient = "/data/fyl/datasets/SPH_data/2015/20180713/K01395919 169"
			print(patient)
			if patient in self.excludes:
				print('data excluded')
				continue
			image, full_image_info, patient_uid = mt.read_sph_scan(patient)
			patient_uid = patient_uid.upper()
			#if full_scan.min()<basicconfig.config['MIN_BOUND']:
			#	errorlog = open("results/error.log", "w")
			#	errorlog.write("Hu unit incorrect:%s\n" %(patient))
			#	errorlog.close()
			if self.serie_ids.count(patient_uid)==0:
				if patient_uid[0]=='0' and self.serie_ids.count(patient_uid[1:])>0:
					patient_uid = patient_uid[1:]
				else:
					print('no annotation for this patient found')
					continue
			#serie_index = self.serie_ids.index(patient_uid)
			patient_nodules, exception = self.get_annotation(patient_uid)
			if len(patient_nodules)==0:
				exceptions.append(patient_uid + ' ' + exception)
				bt.filelist_store(exceptions, self.output_path + '/exceptions.log')
				continue
			
			if overbound:
				#box_shape = np.int_([2*BOX_SIZE, 2*BOX_SIZE, 2*BOX_SIZE])
				box_size = 2 * BOX_SIZE
			else:
				box_size = BOX_SIZE
				#box_shape = np.int_([BOX_SIZE, BOX_SIZE, BOX_SIZE])
			old_spacing = np.array(full_image_info.GetSpacing())[::-1]  #---该CT在“世界空间”中各个方向上相邻单位的体素的间距
			#if target_spacing is not None:
			#	box_shape = np.int_(np.ceil(box_shape / old_spacing * target_spacing))
			#	np.save(self.output_path+'/target_spacing.npy', np.array(target_spacing))

			#box_half = np.int_(box_shape/2)
			#image_padded = np.pad(image, ((box_half[0], box_half[0]), (box_half[1], box_half[1]), (box_half[2], box_half[2])), 'constant', constant_values = ((basicconfig.config['MIN_BOUND'], basicconfig.config['MIN_BOUND']), (basicconfig.config['MIN_BOUND'], basicconfig.config['MIN_BOUND']), (basicconfig.config['MIN_BOUND'], basicconfig.config['MIN_BOUND'])))

			for annoid, diag, nodule in patient_nodules:		
				v_center = nodule[::-1].astype(int)
				nodule_box = mt.box_sample(image, v_center, box_size, old_spacing/np.float_(target_spacing))
				outname = patient_uid + "_" + str(annoid) + "_" + diag
				if target_spacing is None:
					outname += "_" + str(old_spacing)
				if diag in self.diagnosises:
					if hasattr(self, 'nodules_npy_path'):
						self.save_annotations_nodule(nodule_box, outname)
					if hasattr(self, 'nodules_multiview_path'):
						splitnum = 6	
						if 'splitnum' in dir():
							view_list = mt.spherical_sample(image, v_center, old_spacing, splitnum, 50, 224)
							self.save_multiview_nodule(view_list, outname, triple_channels=False)
					if hasattr(self, 'nodules_slicewise_path'):
						diamids = np.logical_and(np.logical_and(self.df_diameters.patient_id==patient_uid, self.df_diameters.nodule_id==annoid), self.df_diameters.diagnosis==diag).nonzero()[0]
						if len(diamids)==1:
							slice_size = int(math.ceil(self.df_diameters.diameter[diamids[0]] / old_spacing[1:].min()))
							self.save_slicewise_nodule(image, v_center, slice_size, outname, image_size=360, overbound=overbound, triple_channels=False)
							'''
							example_crop = mt.local_crop(image, v_center, (3, 49, 49))	#double the slice size for overbound
							resize = 4
							slsb = (24 - int(slice_size/2)) * resize
							slst = (24 - int(slice_size/2) + slice_size) * resize
							exmpslice = example_crop[1]
							exmpslice = scipy.ndimage.interpolation.zoom(exmpslice, resize, mode='nearest')
							exmpslice = np.uint8(mt.medical_normalization(exmpslice, pixel_mean=0) * 255)
							exmpslice = np.stack((exmpslice, exmpslice, exmpslice), axis=2)
							exmpslice[slsb-1:slsb+2,slsb-1:slst+2,0] = 255
							exmpslice[slst-1:slst+2,slsb-1:slst+2,0] = 255
							exmpslice[slsb-1:slst+2,slsb-1:slsb+2,0] = 255
							exmpslice[slsb-1:slst+2,slst-1:slst+2,0] = 255
							skimage.io.imsave(os.path.join(self.nonnodule_npy_path, "%s_ep.png" %(outname)), exmpslice)
							'''
						elif len(diamids)>1:
							print("more than 1 diameters for a correponding nodule")
					if hasattr(self, 'nodules_vision_path'):
						self.save_vision_nodule(nodule_box, outname)
				else:
					self.save_nonnodule(nodule_box, outname)
			print("annotation sampling done")

			#get candidate annotation nodule
			candidate_coords = []
			if candsample:
				segimage, segmask, flag = cd.segment_lung_mask(image)
				if segimage is not None:
					nodule_matrix, index = cd.candidate_detection(segimage,flag)
					cluster_labels = lc.seed_mask_cluster(nodule_matrix, cluster_size=1000)
					#cluster_labels = lc.seed_volume_cluster(image, segmask, eliminate_lower_size=-1)
					#segresult = lc.segment_color_vision(image, cluster_labels)
					#cv.view_CT(segresult)
					#lc.cluster_size_vision(cluster_labels)
					candidate_coords, _ = lc.cluster_centers(cluster_labels)
					#candidate_coords = lc.cluster_center_filter(image, candidate_coords)
				#the coordination order is [z,y,x]
				print("candidate number:%d" %(len(candidate_coords)))
				#volume_regioned = cv.view_coordinates(image, candidate_coords, window_size=10, reverse=False, slicewise=True, show=False)
				#mt.write_mhd_file(self.vision_path+"/"+patient_uid+"_candidate.mhd", volume_regioned, volume_regioned.shape[::-1])
				for cc in range(len(candidate_coords)):
					candidate_center = candidate_coords[cc]
					invalid_loc = False
					if mt.coord_overflow(candidate_center-window_half, image.shape) or mt.coord_overflow(candidate_center+BOX_SIZE-window_half, image.shape):
						invalid_loc = True
						continue
					for index_search, v_center_search in v_centers:
						rpos = v_center_search - candidate_center
						if abs(rpos[0])<window_half and abs(rpos[1])<window_half and abs(rpos[2])<window_half:  #the negative sample is located in the positive location
							invalid_loc = True
							break
					if not invalid_loc:
						zyx_1 = candidate_center - window_half
						zyx_2 = candidate_center + BOX_SIZE - window_half
						nodule_box = np.zeros([BOX_SIZE,BOX_SIZE,BOX_SIZE], np.int16)	#---nodule_box_size = 45
						img_crop = image[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]	#---截取立方体
						img_crop[img_crop<basicconfig.config['MIN_BOUND']] = basicconfig.config['MIN_BOUND']	#---设置窗宽，小于-1000的体素值设置为-1000
						if img_crop.shape[0]!=BOX_SIZE | img_crop.shape[1]!=BOX_SIZE | img_crop.shape[2]!=BOX_SIZE:
							print("error in resmapleing shape")
						try:
							nodule_box[0:BOX_SIZE, 0:BOX_SIZE, 0:BOX_SIZE] = img_crop  # ---将截取的立方体置于nodule_box
						except:
							print("random error")
							continue
						#nodule_box[nodule_box == 0] = config.config['MIN_BOUND']#---将填充的0设置为-1000，可能有极少数的体素由0=>-1000，不过可以忽略不计
						self.save_nonnodule(nodule_box, patient_uid+"_cc_"+str(cc))
				print("candidate sampling done")
			print('Done for this patient!')
		print('Done for all!')

if __name__ == '__main__':
	#subsets = ['subset0', 'subset1', 'subset2', 'subset3', 'subset4', 'subset5', 'subset6', 'subset7', 'subset8', 'subset9']
	#subsets = ['subset6']
	#for subset in subsets:
		#nodule_cube_subset = MhdCrop("/data/fyl/datasets/LUNA16/"+subset+"/", annotations_file="/data/fyl/datasets/LUNA16/csvfiles/annotations_corrected.csv", candidates_file="/data/fyl/datasets/LUNA16/csvfiles/candidates.csv", excludes_file="/data/fyl/datasets/LUNA16/csvfiles/annotations_excluded_corrected.csv", output_path="tensorflow_project/luna_cubes_64_overbound/", path_clear=False)
		#nodule_cube_subset.annotations_crop(target_spacing=[1, 1, 1], annosample='overbound', candsample=False, randsample=False)
		#nodule_cube_subset.candidates_crop(target_spacing=[1, 1, 1], nodule_exclusion=True)
	#mc = MhdCrop("/data/fyl/datasets/LUNA16/subset5/", annotations_file="/data/fyl/datasets/LUNA16/csvfiles/annotations_corrected.csv", candidates_file="/data/fyl/datasets/LUNA16/csvfiles/candidates.csv", excludes_file="/data/fyl/datasets/LUNA16/csvfiles/annotations_excluded_corrected.csv", output_path="/data/fyl/data_samples/luna_cubes_64_overbound/subset5/", path_clear=False)
	#mc.annotations_crop(target_spacing=[1, 1, 1], annosample='overbound', candsample=False, randsample=False)
	#mc.candidates_crop(target_spacing=[1, 1, 1], nodule_exclusion=True)

	#nc_train = NodulesCropMhd("./", "./TIANCHI_data/train/", "./TIANCHI_data/csv_files/train/annotations.csv", "./nodule_cubes/train/", "./detection_vision/train/")
	#nc_train.annotations_crop(overbound=True, candsample=False, randsample=False)
	#nc_val = NodulesCropMhd("./", "./TIANCHI_data/val/", "./TIANCHI_data/csv_files/val/annotations.csv", "./nodule_cubes/val/", "./detection_vision/val")
	#nc_val.annotations_crop(overbound=True, candsample=False, randsample=False)
	#nc_train.candidates_crop(start=0, region_filter=True, save_positive=False, nodule_exclusion=True)
	
	#nc_train = MhdCrop("/data/fyl/datasets/Tianchi_Lung_Disease/train/", "/data/fyl/datasets/Tianchi_Lung_Disease/chestCT_round1_annotation.csv", output_path="/data/fyl/data_samples/tianchild_cubes_overbound2/", vision_path="/data/fyl/datasets/Tianchi_Lung_Disease/detection_vision2/", nodule_vision=False, path_clear=False)
	#nc_train.candidates_crop(start=0, region_filter=True, save_positive=False, nodule_exclusion=True)
	#nc_train.annotations_crop(start=800, target_spacing=(1, 1, 1), annosample='overbound', candsample=False, randsample=False)
	#nc_train = NpyCrop("/data/fyl/datasets/Tianchi_Lung_Disease/train_resampled/", candidates_file="/data/fyl/datasets/Tianchi_Lung_Disease/candidate.csv", excludes_file="/data/fyl/datasets/Tianchi_Lung_Disease/annotation.csv", output_path="/data/fyl/data_samples/tianchild_cubes_overbound2/", vision_path="/data/fyl/datasets/Tianchi_Lung_Disease/detection_vision/", nodule_vision=False, path_clear=True)
	#nc_train.candidates_crop(start=0, region_filter=True, save_positive=False, nodule_exclusion=True)

	lc = LIDCCrop("/data/fyl/datasets/DOI/", "/data/fyl/data_samples/lidc_3slices", volume_sample=False, multiview_sample=False, slicewise_sample=True, statistics=False, nodule_vision=False, ipris=False, path_clear=False)
	lc.pathology_crop(target_spacing=None, expert_wise=False, start=0)

	'''
	root_path = "/data/fyl/datasets/SPH_data/"
	data_paths = ["2014/", "2015/", "2016/"]
	annotation_files = ["annotations/MIAannotations2014.xlsx", "annotations/MIAannotations2015.xlsx", "annotations/annotations2016.xlsx"]
	output_paths = ["/data/fyl/data_samples/sph_overbound", "/data/fyl/data_samples/sph_overbound", "/data/fyl/data_samples/sph_overbound"] 
	versions = [2, 2, 2]
	starts = [0, 0, 0]
	for d in range(1, 2):
		sc = SPHCrop(root_path+data_paths[d], root_path+annotation_files[d], "/data/fyl/datasets/SPH_data/annotations/statistics.xlsx",  output_path=output_paths[d], version=versions[d], volume_sample=False, multiview_sample=False, slicewise_sample=True, nodule_vision=False, path_clear=False)
		sc.annotations_crop(target_spacing=None, start=starts[d])
		#sc.annotations_statistics(fast=True, start=0)
	'''
	#sc = SPHCrop("/data/fyl/datasets/SPH_data/2016/", "/data/fyl/datasets/SPH_data/annotations/annotations2016.xlsx", diameters_file="/data/fyl/datasets/SPH_data/annotations/statistics.xlsx", output_path="data_samples/sph2016_overbound/", version=2, volume_sample=False, multiview_sample=False, slicewise_sample=True, nodule_vision=False, path_clear=True)
	#sc.annotations_crop(target_spacing=(0.5, 0.5, 0.5), overbound=True, candsample=False, start=-5)
	#sc.annotations_statistics(fast=True, start=0)
