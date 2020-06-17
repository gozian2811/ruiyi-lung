import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import functools as ft
import os
import shutil
import glob
import math
import random
import time
from config import config
from toolbox import MITools as mt
from toolbox import TensorflowTools as tft

try:
	from tqdm import tqdm  # long waits are not fun
except:
	print('tqdm not installed')
	tqdm = lambda x: x

#constants = bt.read_constants("./constants2.txt")
#MAX_BOUND = float(constants["MAX_BOUND"])
#MIN_BOUND = float(constants["MIN_BOUND"])
#PIXEL_MEAN = float(constants["PIXEL_MEAN"])
MAX_BOUND = float(config['MAX_BOUND'])
MIN_BOUND = float(config['MIN_BOUND'])
PIXEL_MEAN = float(config['PIXEL_MEAN'])
NORM_CROP = config["NORM_CROP"]

REGION_SIZES = (20, 30, 40)
BATCH_SIZE = 20
VALIDATION_RATE = 0
NEGATIVE_VALIDATION = True
FUSION_MODE = 'late'

SCALE_AUGMENTATION = False
TRANSLATION_AUGMENTATION = False
ROTATION_AUGMENTATION = False
FLIP_AUGMENTATION = False
CENTERING = True
AUGMENTATION = SCALE_AUGMENTATION or TRANSLATION_AUGMENTATION or ROTATION_AUGMENTATION or FLIP_AUGMENTATION

aug_proportion = (1+2*int(SCALE_AUGMENTATION)) * (1+26*int(TRANSLATION_AUGMENTATION)*(1+2*int(CENTERING))) * (1+9*int(ROTATION_AUGMENTATION)+3*int(FLIP_AUGMENTATION))

print("batch size:{}" .format(BATCH_SIZE))
print("region sizes:{}" .format(REGION_SIZES))
print("max bound:{}" .format(MAX_BOUND))
print("min bound:{}" .format(MIN_BOUND))
print("pixel mean:{}" .format(PIXEL_MEAN))
region_halfs = [int(REGION_SIZES[0]/2), int(REGION_SIZES[1]/2), int(REGION_SIZES[2]/2)]

load_path = "models_tensorflow"

net_init_names = ["luna_slh_3D_bndo_flbias_l5_20_aug_stage3", "luna_slh_3D_bndo_flbias_l5_30_aug2_stage2", "luna_slh_3D_bndo_flbias_l6_40_aug_stage2"]
if 'net_init_names' in dir():
	net_init_paths = [load_path + "/" + net_init_names[0], load_path + "/" + net_init_names[1], load_path + "/" + net_init_names[2]]
	net_init_files = [net_init_paths[0]+"/epoch20/epoch20", net_init_paths[1]+"/epoch10/epoch10", net_init_paths[2]+"/epoch25/epoch25", load_path+"/luna_slh_3D_fusion2/epoch6/epoch6"]
	bn_files = ["models_tensorflow/luna_slh_3D_bndo_flbias_l5_20_aug_stage3/batch_normalization_statistic.npy",
		   "models_tensorflow/luna_slh_3D_bndo_flbias_l5_30_aug2_stage2/batch_normalization_statistic.npy",
		   "models_tensorflow/luna_slh_3D_bndo_flbias_l6_40_aug_stage2/batch_normalization_statistic_25.npy"]
pfilepath = 'luna_cubes_56_overbound/subset9/npy'
nfilepath = 'luna_cubes_56_overbound/subset9/npy_non'
#pfilelist_path = net_init_path + "/pfilelist.log"
#nfilelist_path = net_init_path + "/nfilelist.log"
vision_path = "detection_vision"

if 'pfilelist_path' in dir():
	print("read pfilelist from: %s" %(pfilelist_path))
	pfilelist_file = open(pfilelist_path, "r")
	pfiles = pfilelist_file.readlines()
	for pfi in range(len(pfiles)):
		pfiles[pfi] = pfiles[pfi][:-1]
	pfilelist_file.close()
else:
	pdir = pfilepath + "/*.npy"
	pfiles = glob.glob(pdir)
if 'nfilelist_path' in dir():
	print("read nfilelist from: %s" % (nfilelist_path))
	nfilelist_file = open(nfilelist_path, "r")
	nfiles = nfilelist_file.readlines()
	for nfi in range(len(nfiles)):
		nfiles[nfi] = nfiles[nfi][:-1]
	nfilelist_file.close()
else:
	ndir = nfilepath + "/*.npy"
	nfiles = glob.glob(ndir)

num_positive = len(pfiles)
num_negative = len(nfiles)

positive_val_num = int(num_positive * VALIDATION_RATE)
positive_train_num = num_positive - positive_val_num
negative_val_num = int(num_negative * VALIDATION_RATE)
negative_train_num = num_negative - negative_val_num
tpfiles = pfiles[:positive_train_num]
vpfiles = pfiles[positive_train_num:num_positive]
positive_train_indices = [i for i in range(positive_train_num)]
positive_val_indices = [i for i in range(positive_val_num)]
tnfiles = nfiles[:negative_train_num]
vnfiles = nfiles[negative_train_num:num_negative]
negative_train_indices = [i for i in range(negative_train_num)]
negative_val_indices = [i for i in range(negative_val_num)]

#random.shuffle(positive_train_indices)
#random.shuffle(positive_val_indices)

#net construct
volume_inputs = [tf.placeholder(tf.float32, [None, REGION_SIZES[0], REGION_SIZES[0], REGION_SIZES[0]]),
		 tf.placeholder(tf.float32, [None, REGION_SIZES[1], REGION_SIZES[1], REGION_SIZES[1]]),
		 tf.placeholder(tf.float32, [None, REGION_SIZES[2], REGION_SIZES[2], REGION_SIZES[2]])]
volumes_reshape = [tf.reshape(volume_inputs[0], [-1, REGION_SIZES[0], REGION_SIZES[0], REGION_SIZES[0], 1]),
		   tf.reshape(volume_inputs[1], [-1, REGION_SIZES[1], REGION_SIZES[1], REGION_SIZES[1], 1]),
		   tf.reshape(volume_inputs[2], [-1, REGION_SIZES[2], REGION_SIZES[2], REGION_SIZES[2], 1])]
real_label = tf.placeholder(tf.float32, [None, 2])
#r_bn1, b_bn1, w_conv1, w_conv2, out_conv1, out_bn1, hidden_conv1, hidden_conv2, hidden_conv3, out_fc1, softmax_out, softmax_out = tft.volume_bnnet2_l6_56(volume_reshape)
if "bn_files" in dir():
	bn_params = [np.load(bn_files[0]), np.load(bn_files[1]), np.load(bn_files[2])]
else:
	bn_params = [None, None, None]
net_outs0, variables0, _ = tft.volume_bndo_flbias_l5_20(volumes_reshape[0], False, dropout_rate=0.0, batch_normalization_statistic=False, bn_params=bn_params[0])
net_outs1, variables1, _ = tft.volume_bndo_flbias_l5_30(volumes_reshape[1], False, dropout_rate=0.0, batch_normalization_statistic=False, bn_params=bn_params[1])
net_outs2, variables2, _ = tft.volume_bndo_flbias_l6_40(volumes_reshape[2], False, dropout_rate=0.0, batch_normalization_statistic=False, bn_params=bn_params[2])
if FUSION_MODE == 'late':
	features = [net_outs0['flattened_out'], net_outs1['flattened_out'], net_outs2['fc1_out']]
	_, softmax_out, variables_fusion = tft.late_fusion(features, True)
elif FUSION_MODE == 'committe':
	predictions = [net_outs0['sm_out'], net_outs1['sm_out'], net_outs2['sm_out']]
	softmax_out = tft.committe_fusion(predictions)
correct_prediction = tf.equal(tf.argmax(softmax_out, 1), tf.argmax(real_label, 1))
batch_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

if not AUGMENTATION:
	correct_output = open(vision_path + "/correct_predictions_"+FUSION_MODE+"_subset9.log", "w")
	incorrect_output = open(vision_path + "/incorrect_predictions_"+FUSION_MODE+"_subset9.log", "w")

extract_volumes = [ft.partial(mt.extract_volumes, volume_shape=np.int_([REGION_SIZES[0], REGION_SIZES[0], REGION_SIZES[0]]), centering=CENTERING, scale_augment=SCALE_AUGMENTATION, translation_augment=TRANSLATION_AUGMENTATION, rotation_augment=ROTATION_AUGMENTATION, flip_augment=FLIP_AUGMENTATION),
		   ft.partial(mt.extract_volumes, volume_shape=np.int_([REGION_SIZES[1], REGION_SIZES[1], REGION_SIZES[1]]), centering=CENTERING, scale_augment=SCALE_AUGMENTATION, translation_augment=TRANSLATION_AUGMENTATION, rotation_augment=ROTATION_AUGMENTATION, flip_augment=FLIP_AUGMENTATION),
		   ft.partial(mt.extract_volumes, volume_shape=np.int_([REGION_SIZES[2], REGION_SIZES[2], REGION_SIZES[2]]), centering=CENTERING, scale_augment=SCALE_AUGMENTATION, translation_augment=TRANSLATION_AUGMENTATION, rotation_augment=ROTATION_AUGMENTATION, flip_augment=FLIP_AUGMENTATION)]
medical_normalization = ft.partial(mt.medical_normalization, max_bound=MAX_BOUND, min_bound=MIN_BOUND, pixel_mean=PIXEL_MEAN, crop=NORM_CROP, input_copy=False)
saver0 = tf.train.Saver(variables0)
saver1 = tf.train.Saver(variables1)
saver2 = tf.train.Saver(variables2)
if FUSION_MODE == 'late':
	saver_fusion = tf.train.Saver(variables_fusion)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
	#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
	print("loading net from files:{}" .format(net_init_files))
	saver0.restore(sess, net_init_files[0])
	saver1.restore(sess, net_init_files[1])
	saver2.restore(sess, net_init_files[2])
	if FUSION_MODE == 'late':
		saver_fusion.restore(sess, net_init_files[3])
	
	ptd_num = 0
	ptd_accuracy = 0.0
	positive_batch_size = int(BATCH_SIZE/aug_proportion)
	if not AUGMENTATION:
		correct_output.write("positive training:\n")
		incorrect_output.write("positive training:\n")
	for pbi in tqdm(range(0, positive_train_num, positive_batch_size)):
		posbatchend = min(pbi+positive_batch_size, positive_train_num)
		for pti in range(pbi, posbatchend):
			data_index = positive_train_indices[pti]
			pfile = tpfiles[data_index]
			positive_data = np.load(pfile)
			nodule_diameter = 0
			if "positive_batches" not in dir():
				#positive_batch = mt.extract_volumes(positive_data, centering=CENTERING, nodule_diameter=nodule_diameter, scale_augment=SCALE_AUGMENTATION, translation_augment=TRANSLATION_AUGMENTATION, rotation_augment=ROTATION_AUGMENTATION, flip_augment=FLIP_AUGMENTATION)
				positive_batches = [extract_volumes[0](positive_data, nodule_diameter=nodule_diameter), extract_volumes[1](positive_data, nodule_diameter=nodule_diameter), extract_volumes[2](positive_data, nodule_diameter=nodule_diameter)]
			else:
				#positive_batch = np.concatenate((positive_batch, mt.extract_volumes(positive_data, centering=CENTERING, nodule_diameter=nodule_diameter, scale_augment=SCALE_AUGMENTATION, translation_augment=TRANSLATION_AUGMENTATION, rotation_augment=ROTATION_AUGMENTATION, flip_augment=FLIP_AUGMENTATION)), axis=0)
				positive_batches = [np.concatenate((positive_batches[0], extract_volumes[0](positive_data, nodule_diameter=nodule_diameter)), axis=0),
						    np.concatenate((positive_batches[1], extract_volumes[1](positive_data, nodule_diameter=nodule_diameter)), axis=0),
						    np.concatenate((positive_batches[2], extract_volumes[2](positive_data, nodule_diameter=nodule_diameter)), axis=0)]
			del positive_data
		positive_batches = [medical_normalization(positive_batches[0]), medical_normalization(positive_batches[1]), medical_normalization(positive_batches[2])]
		positive_label = np.zeros(shape=(positive_batches[0].shape[0], 2), dtype=float)
		positive_label[:,0] = 1
		predictions, accuracies = sess.run([softmax_out, correct_prediction], {volume_inputs[0]: positive_batches[0], volume_inputs[1]: positive_batches[1], volume_inputs[2]: positive_batches[2], real_label: positive_label})
		if not AUGMENTATION:
			for a in range(len(accuracies)):
				data_index = positive_train_indices[pbi+a]
				pfile = tpfiles[data_index]
				if accuracies[a]:
					correct_output.write(pfile+" {}\n" .format(predictions[a]))
				else:
					incorrect_output.write(pfile+" {}\n" .format(predictions[a]))
		for di in range(len(accuracies)):
			ptd_num += 1
			ptd_accuracy += accuracies[di]
		
		del positive_batches

	if ptd_num > 0:
		ptd_accuracy /= ptd_num
	print("positive training accuracy:%f" %(ptd_accuracy))
	
	pvd_num = 0
	pvd_accuracy = 0.0
	if not AUGMENTATION:
		correct_output.write("\npositive validation:\n")
		incorrect_output.write("\npositive validation:\n")
	for pbi in tqdm(range(0, positive_val_num, positive_batch_size)):
		posbatchend = min(pbi+positive_batch_size, positive_val_num)
		for pvi in range(pbi, posbatchend):
			data_index = positive_val_indices[pvi]
			pfile = vpfiles[data_index]
			positive_data = np.load(pfile)
			nodule_diameter = 0
			if "positive_batches" not in dir():
				#positive_batch = mt.extract_volumes(positive_data, centering=CENTERING, nodule_diameter=nodule_diameter, scale_augment=SCALE_AUGMENTATION, translation_augment=TRANSLATION_AUGMENTATION, rotation_augment=ROTATION_AUGMENTATION, flip_augment=FLIP_AUGMENTATION)
				positive_batches = [extract_volumes[0](positive_data, nodule_diameter=nodule_diameter), extract_volumes[1](positive_data, nodule_diameter=nodule_diameter), extract_volumes[2](positive_data, nodule_diameter=nodule_diameter)]
			else:
				#positive_batch = np.concatenate((positive_batch, mt.extract_volumes(positive_data, centering=CENTERING, nodule_diameter=nodule_diameter, scale_augment=SCALE_AUGMENTATION, translation_augment=TRANSLATION_AUGMENTATION, rotation_augment=ROTATION_AUGMENTATION, flip_augment=FLIP_AUGMENTATION)), axis=0)
				positive_batches = [np.concatenate((positive_batches[0], extract_volumes[0](positive_data, nodule_diameter=nodule_diameter)), axis=0),
						    np.concatenate((positive_batches[1], extract_volumes[1](positive_data, nodule_diameter=nodule_diameter)), axis=0),
						    np.concatenate((positive_batches[2], extract_volumes[2](positive_data, nodule_diameter=nodule_diameter)), axis=0)]
		positive_batches = [medical_normalization(positive_batches[0]), medical_normalization(positive_batches[1]), medical_normalization(positive_batches[2])]
		positive_label = np.zeros(shape=(positive_batches[0].shape[0], 2), dtype=float)
		positive_label[:,0] = 1
		predictions, accuracies = sess.run([softmax_out, correct_prediction], {volume_inputs[0]: positive_batches[0], volume_inputs[1]: positive_batches[1], volume_inputs[2]: positive_batches[2], real_label: positive_label})
		if not AUGMENTATION:
			for a in range(len(accuracies)):
				data_index = positive_val_indices[pbi+a]
				pfile = vpfiles[data_index]
				if accuracies[a]:
					correct_output.write(pfile+" {}\n" .format(predictions[a]))
				else:
					incorrect_output.write(pfile+" {}\n" .format(predictions[a]))
		for di in range(len(accuracies)):
			pvd_num += 1
			pvd_accuracy += accuracies[di]
		
		del positive_batches

	if pvd_num > 0:
		pvd_accuracy /= pvd_num
	print("positive validation accuracy:%f" %(pvd_accuracy))
	
	if NEGATIVE_VALIDATION:
		ntd_num = 0
		ntd_accuracy = 0.0	
		if not AUGMENTATION:
			correct_output.write("\nnegative training:\n")
			incorrect_output.write("\nnegative training:\n")
		for nbi in tqdm(range(0, negative_train_num, BATCH_SIZE)):
			negbatchend = min(nbi+BATCH_SIZE, negative_train_num)
			negative_batches = [np.empty((negbatchend-nbi, REGION_SIZES[0], REGION_SIZES[0], REGION_SIZES[0]), dtype=float),
					    np.empty((negbatchend-nbi, REGION_SIZES[1], REGION_SIZES[1], REGION_SIZES[1]), dtype=float),
					    np.empty((negbatchend-nbi, REGION_SIZES[2], REGION_SIZES[2], REGION_SIZES[2]), dtype=float)]
			for nti in range(nbi, negbatchend):
				data_index = negative_train_indices[nti]
				nfile = tnfiles[data_index]
				negative_data = np.load(nfile)
				ndcenter = np.int_([negative_data.shape[0]/2, negative_data.shape[1]/2, negative_data.shape[2]/2])
				negative_batches[0][nti-nbi] = negative_data[ndcenter[0]-region_halfs[0]:ndcenter[0]+REGION_SIZES[0]-region_halfs[0], ndcenter[1]-region_halfs[0]:ndcenter[1]+REGION_SIZES[0]-region_halfs[0], ndcenter[2]-region_halfs[0]:ndcenter[2]+REGION_SIZES[0]-region_halfs[0]]
				negative_batches[1][nti-nbi] = negative_data[ndcenter[0]-region_halfs[1]:ndcenter[0]+REGION_SIZES[1]-region_halfs[1], ndcenter[1]-region_halfs[1]:ndcenter[1]+REGION_SIZES[1]-region_halfs[1], ndcenter[2]-region_halfs[1]:ndcenter[2]+REGION_SIZES[1]-region_halfs[1]]
				negative_batches[2][nti-nbi] = negative_data[ndcenter[0]-region_halfs[2]:ndcenter[0]+REGION_SIZES[2]-region_halfs[2], ndcenter[1]-region_halfs[2]:ndcenter[1]+REGION_SIZES[2]-region_halfs[2], ndcenter[2]-region_halfs[2]:ndcenter[2]+REGION_SIZES[2]-region_halfs[2]]
			negative_batches = [medical_normalization(negative_batches[0]), medical_normalization(negative_batches[1]), medical_normalization(negative_batches[2])]
			negative_label = np.zeros(shape=(negbatchend-nbi, 2), dtype=float)
			negative_label[:,1] = 1
			predictions, accuracies = sess.run([softmax_out, correct_prediction], {volume_inputs[0]: negative_batches[0], volume_inputs[1]: negative_batches[1], volume_inputs[2]: negative_batches[2], real_label: negative_label})
			if not AUGMENTATION:
				for a in range(len(accuracies)):
					data_index = negative_train_indices[nbi+a]
					nfile = tnfiles[data_index]
					if accuracies[a]:
						correct_output.write(nfile+" {}\n" .format(predictions[a]))
					else:
						incorrect_output.write(nfile+" {}\n" .format(predictions[a]))
			for di in range(len(accuracies)):
				ntd_num += 1
				ntd_accuracy += accuracies[di]
			
			del negative_batches

		if ntd_num > 0:
			ntd_accuracy /= ntd_num
		print("negative training accuracy:%f" %(ntd_accuracy))
		
		nvd_num = 0
		nvd_accuracy = 0.0
		if not AUGMENTATION:
			correct_output.write("\nnegative validation:\n")
			incorrect_output.write("\nnegative validation:\n")
		for nbi in tqdm(range(0, negative_val_num, BATCH_SIZE)):
			negbatchend = min(nbi+BATCH_SIZE, negative_val_num)
			negative_batches = [np.empty((negbatchend-nbi, REGION_SIZES[0], REGION_SIZES[0], REGION_SIZES[0]), dtype=float),
					    np.empty((negbatchend-nbi, REGION_SIZES[1], REGION_SIZES[1], REGION_SIZES[1]), dtype=float),
					    np.empty((negbatchend-nbi, REGION_SIZES[2], REGION_SIZES[2], REGION_SIZES[2]), dtype=float)]
			for nvi in range(nbi, negbatchend):
				data_index = negative_val_indices[nvi]
				nfile = vnfiles[data_index]
				negative_data = np.load(nfile)
				ndcenter = np.int_([negative_data.shape[0]/2, negative_data.shape[1]/2, negative_data.shape[2]/2])
				negative_batches[0][nvi-nbi] = negative_data[ndcenter[0]-region_halfs[0]:ndcenter[0]+REGION_SIZES[0]-region_halfs[0], ndcenter[1]-region_halfs[0]:ndcenter[1]+REGION_SIZES[0]-region_halfs[0], ndcenter[2]-region_halfs[0]:ndcenter[2]+REGION_SIZES[0]-region_halfs[0]]
				negative_batches[1][nvi-nbi] = negative_data[ndcenter[0]-region_halfs[1]:ndcenter[0]+REGION_SIZES[1]-region_halfs[1], ndcenter[1]-region_halfs[1]:ndcenter[1]+REGION_SIZES[1]-region_halfs[1], ndcenter[2]-region_halfs[1]:ndcenter[2]+REGION_SIZES[1]-region_halfs[1]]
				negative_batches[2][nvi-nbi] = negative_data[ndcenter[0]-region_halfs[2]:ndcenter[0]+REGION_SIZES[2]-region_halfs[2], ndcenter[1]-region_halfs[2]:ndcenter[1]+REGION_SIZES[2]-region_halfs[2], ndcenter[2]-region_halfs[2]:ndcenter[2]+REGION_SIZES[2]-region_halfs[2]]
			negative_batches = [medical_normalization(negative_batches[0]), medical_normalization(negative_batches[1]), medical_normalization(negative_batches[2])]
			negative_label = np.zeros(shape=(negbatchend-nbi, 2), dtype=float)
			negative_label[:,1] = 1
			predictions, accuracies = sess.run([softmax_out, correct_prediction], {volume_inputs[0]: negative_batches[0], volume_inputs[1]: negative_batches[1], volume_inputs[2]: negative_batches[2], real_label: negative_label})
			if not AUGMENTATION:
				for a in range(len(accuracies)):
					data_index = negative_train_indices[nbi+a]
					nfile = vnfiles[data_index]
					if accuracies[a]:
						correct_output.write(nfile+" {}\n" .format(predictions[a]))
					else:
						incorrect_output.write(nfile+" {}\n" .format(predictions[a]))
			for di in range(len(accuracies)):
				nvd_num += 1
				nvd_accuracy += accuracies[di]
			
			del negative_batches

		if nvd_num > 0:
			nvd_accuracy /= nvd_num
		print("negative validation accuracy:%f" %(nvd_accuracy))

	#print("positive_train_acc:%f positive_val_acc:%f" %(ptd_accuracy, pvd_accuracy))
	#print("positive_train_acc:%f negative_train_acc:%f\npositive_val_acc:%f negative_val_acc:%f" %(ptd_accuracy, ntd_accuracy, pvd_accuracy, nvd_accuracy))

if not AUGMENTATION:
	correct_output.close()
	incorrect_output.close()