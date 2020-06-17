import tensorflow as tf
import numpy as np
import functools as ft
import os
import shutil
import glob
import math
import random
import time
#from tensorflow.python import debug as tf_debug
from config import config
from toolbox import BasicTools as bt
from toolbox import MITools as mt
from toolbox import TensorflowTools as tft

try:
	from tqdm import tqdm  #long waits are not fun
except:
	print('tqdm not installed')
	tqdm = lambda x: x

MAX_BOUND = float(config["MAX_BOUND"])
MIN_BOUND = float(config["MIN_BOUND"])
PIXEL_MEAN = float(config["PIXEL_MEAN"])
NORM_CROP = config["NORM_CROP"]
LOSS_BALANCING = False
FOCAL_LOSS = True

#NUM_EPOCH = 200
SNAPSHOT_EPOCH = 5
#DECAY_EPOCH = 0
#INITIAL_LEARNING_RATE = 0.001
#DECAY_LEARNING_RATE = 1.0

ALL_NEGATIVE = True
REGION_SIZE = 30
BATCH_SIZE = 100
VALIDATION_RATE = 0

SCALE_AUGMENTATION = False
TRANSLATION_AUGMENTATION = False
ROTATION_AUGMENTATION = True
FLIP_AUGMENTATION = True
CENTERING = True
AUGMENTATION = SCALE_AUGMENTATION or TRANSLATION_AUGMENTATION or ROTATION_AUGMENTATION or FLIP_AUGMENTATION

print("batch size:{}" .format(BATCH_SIZE))
print("region size:{}" .format(REGION_SIZE))
print("max bound:{}" .format(MAX_BOUND))
print("min bound:{}" .format(MIN_BOUND))
print("pixel mean:{}" .format(PIXEL_MEAN))
print("normalization crop:{}" .format(NORM_CROP))
print("loss balancing:{}" .format(LOSS_BALANCING))
print("focal loss:{}" .format(FOCAL_LOSS))
region_half = int(REGION_SIZE/2)

store_path = "models_tensorflow"
load_path = "models_tensorflow"
net_store_name = "luna_3D_bndo_flbias_l5_30_aug_stage3"
net_store_path = store_path + "/" + net_store_name
tensorboard_path = net_store_path + "/tensorboard/"
net_init_name = "luna_3D_bndo_flbias_l5_30_aug_stage2"
#pfilelist_path = load_path + "/luna_slh_3D_bndo_flbias_l6_40_aug_init/pfilelist.log"
#nfilelist_path = load_path + "/luna_slh_3D_bndo_flbias_l6_40_aug_init/nfilelist.log"
if 'net_init_name' in dir():
	net_init_path = load_path + "/" + net_init_name
	net_init_file = net_init_path + "/epoch1/epoch1"
	pfilelist_path = net_init_path + "/pfilelist.log"
	nfilelist_path = net_init_path + "/nfilelist.log"
else:
	pfilelist_path = None
	nfilelist_path = None

if os.access(net_store_path, os.F_OK):
	shutil.rmtree(net_store_path)
os.makedirs(net_store_path)
if os.access(tensorboard_path, os.F_OK):
	shutil.rmtree(tensorboard_path)

#data arrangement
luna_dir="luna_cubes_56_overbound_excorr"
#data_dir2 = "tianchi_cubes_56_overbound"
slh_dir = "slh_cubes_56_overbound"
luna_trainsets = ["subset0", "subset1", "subset2", "subset3", "subset4", "subset5", "subset6", "subset7", "subset8"]
luna_valsets = ["subset9"]
tianchi_trainsets = ["train", "val"]
#train_sets = ["subset9"]

'''
if "pfilelist_path" in dir() and os.path.exists(pfilelist_path):
	print("read pfilelist from: %s" %(pfilelist_path))
	pfiles = bt.filelist_load(pfilelist_path)
	#pfilelist_file = open(pfilelist_path, "r")
	#pfiles = pfilelist_file.readlines()
	#for pfi in range(len(pfiles)):
	#	pfiles[pfi] = pfiles[pfi][:-1]
	#pfilelist_file.close()
else:
	pfiles = []
	for set in train_sets:
		train_dir = os.path.join(data_dir1, set)
		pdir = os.path.join(train_dir,"npy","*.npy")
		pfiles.extend(glob.glob(pdir))
		#ndir = os.path.join(train_dir,"npy_non","*.npy")
		#nfiles.extend(glob.glob(ndir))
	if "data_dir2" in dir():
		for set in train_sets2:
			train_dir2 = os.path.join(data_dir2, set)
			pdir = os.path.join(train_dir2,"npy","*.npy")
			pfiles.extend(glob.glob(pdir))
	if "data_dir3" in dir():
		pfiles.extend(glob.glob(os.path.join(data_dir3,"npy","*.npy")))
	random.shuffle(pfiles)
bt.filelist_store(pfiles, net_store_path + "/pfilelist.log")
#pfile_storage = open(net_store_path + "/pfilelist.log", "w")
#for pfile in pfiles:
#	pfile_storage.write("%s\n" %(pfile))
#pfile_storage.close()
if "nfilelist_path" in dir() and os.path.exists(nfilelist_path):
	print("read nfilelist from: %s" % (nfilelist_path))
	nfiles = bt.filelist_load(nfilelist_path)
	#nfilelist_file = open(nfilelist_path, "r")
	#nfiles = nfilelist_file.readlines()
	#for nfi in range(len(nfiles)):
	#	nfiles[nfi] = nfiles[nfi][:-1]
	#nfilelist_file.close()
else:
	nfiles = []
	for set in train_sets:
		train_dir = os.path.join(data_dir1, set)
		ndir = os.path.join(train_dir, "npy_non", "*.npy")
		nfiles.extend(glob.glob(ndir))
	random.shuffle(nfiles)
bt.filelist_store(nfiles, net_store_path + "/nfilelist.log")
#nfile_storage = open(net_store_path + "/nfilelist.log", "w")
#for nfile in nfiles:
#	nfile_storage.write("%s\n" %(nfile))
#nfile_storage.close()

num_positive = len(pfiles)
num_negative = len(nfiles)
#num_positive = 20
#num_negative = 800

#num_negative = 1000 * num_positive
if num_positive==0:
	print("no positive training file found")
	exit()

positive_val_num = int(num_positive * VALIDATION_RATE)
#positive_val_num = 1
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
#negative_importances = 1000*np.ones(shape=[num_negative], dtype=float)
'''
files_lists = bt.filelist_training(pfilelist_path, nfilelist_path, luna_dir=luna_dir, luna_trainsets = luna_trainsets, luna_valsets = luna_valsets, slh_dir = None, valrate = VALIDATION_RATE, list_store_path = net_store_path)
tpfiles = files_lists['tpfiles']
vpfiles = files_lists['vpfiles']
tnfiles = files_lists['tnfiles']
vnfiles = files_lists['vnfiles']
positive_train_num = len(tpfiles)
positive_val_num = len(vpfiles)
negative_train_num = len(tnfiles)
negative_val_num = len(vnfiles)
#positive_train_num = 1
#negative_train_num = 1
positive_train_indices = [i for i in range(positive_train_num)]
positive_val_indices = [i for i in range(positive_val_num)]
negative_train_indices = [i for i in range(negative_train_num)]
negative_val_indices = [i for i in range(negative_val_num)]

#net construct
volume_input = tf.placeholder(tf.float32, [None, REGION_SIZE, REGION_SIZE, REGION_SIZE])
volume_reshape = tf.reshape(volume_input, [-1, REGION_SIZE, REGION_SIZE, REGION_SIZE, 1])
real_label = tf.placeholder(tf.float32, [None, 2])
#bn_params = np.load(net_init_path + "/batch_normalization_statistic.npy")
aug_proportion = (1+2*int(SCALE_AUGMENTATION)) * (1+26*int(TRANSLATION_AUGMENTATION)*(1+2*int(CENTERING))) * (1+9*int(ROTATION_AUGMENTATION)+3*int(FLIP_AUGMENTATION))
if ALL_NEGATIVE:
	np_proportion = negative_train_num / float(positive_train_num)
else:
	np_proportion = aug_proportion
num_positive_aug = positive_train_num * aug_proportion
positive_confidence = num_positive_aug / float(num_positive_aug + positive_train_num * np_proportion)
net_outs, _, bn_params = tft.volume_bndo_flbias_l5_30(volume_reshape, True, positive_confidence, dropout_rate=0.3, batch_normalization_statistic=True, bn_params=None)
loss = tf.nn.softmax_cross_entropy_with_logits(logits = net_outs['last_out'], labels = real_label)
if type(LOSS_BALANCING)==float:
	balancing_factor = LOSS_BALANCING
else:
	balancing_factor = positive_confidence
balancing = real_label[:,0] + tf.pow(tf.constant(-1, dtype=tf.float32), real_label[:,0]) * tf.constant(balancing_factor, tf.float32)
modulation = tf.pow(real_label[:,0]+tf.pow(tf.constant(-1, dtype=tf.float32), real_label[:,0])*net_outs['sm_out'][:,0], tf.constant(2, dtype=tf.float32))
if LOSS_BALANCING:
	loss = balancing * loss
if FOCAL_LOSS:
	loss = modulation * loss
batch_loss = tf.reduce_mean(loss)
correct_prediction = tf.equal(tf.argmax(net_outs['sm_out'], 1), tf.argmax(real_label, 1))
batch_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_loss_summary = tf.summary.scalar('train loss', batch_loss)
val_loss_summary = tf.summary.scalar('val loss', batch_loss)
train_acc_summary = tf.summary.scalar('train acc', batch_accuracy)
val_acc_summary = tf.summary.scalar('val acc', batch_accuracy)
train_merge = tf.summary.merge([train_loss_summary, train_acc_summary])
val_merge = tf.summary.merge([val_loss_summary, val_acc_summary])
'''
trains = []
epochs = []
learning_rate = INITIAL_LEARNING_RATE
for ti in range(0, NUM_EPOCH, DECAY_EPOCH):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	trains.append(optimizer.minimize(batch_loss))
	epochs.append(min(NUM_EPOCH-ti, DECAY_EPOCH))
	learning_rate *= DECAY_LEARNING_RATE
'''
epochs = [0, 50, 0]
learning_rates = [0.01, 0.005, 0.0001]
trains = []
for stage in range(len(learning_rates)):
	optimizer = tf.train.GradientDescentOptimizer(learning_rates[stage])
	train = optimizer.minimize(batch_loss)
	trains.append(train)
'''
optimizer1 = tf.train.GradientDescentOptimizer(0.01)
train1 = optimizer1.minimize(batch_loss)
optimizer2 = tf.train.GradientDescentOptimizer(0.001)
train2 = optimizer2.minimize(batch_loss)
optimizer3 = tf.train.GradientDescentOptimizer(0.0001)
train3 = optimizer3.minimize(batch_loss)
trains = [train1, train2, train3]	#training stages of different learning rates
'''

extract_volumes = ft.partial(mt.extract_volumes, volume_shape=np.int_([REGION_SIZE, REGION_SIZE, REGION_SIZE]), centering=CENTERING, scale_augment=SCALE_AUGMENTATION, translation_augment=TRANSLATION_AUGMENTATION, rotation_augment=ROTATION_AUGMENTATION, flip_augment=FLIP_AUGMENTATION)
medical_normalization = ft.partial(mt.medical_normalization, max_bound=MAX_BOUND, min_bound=MIN_BOUND, pixel_mean=PIXEL_MEAN, crop=NORM_CROP, input_copy=False)
saver = tf.train.Saver(max_to_keep=None)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
	#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
	if 'net_init_file' not in dir() or net_init_file is None:
		print("net randomly initialized")
		sess.run(tf.global_variables_initializer())
	else:
		print("loading net from file:{}" .format(net_init_file))
		saver.restore(sess, net_init_file)
	tensorboard_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
	best_accuracy = 0.0
	best_epoch = 0
	overall_epoch = 0
	training_messages = []
	for train_stage in range(len(trains)):
		train = trains[train_stage]
		for epoch in range(epochs[train_stage]):
			print('epoch:%d/%d' %(overall_epoch+1, epochs[train_stage]))
			batch_num = 0
			bn_stats = np.zeros(shape=np.array(bn_params).shape, dtype=float)

			# the number of negative samples is in default larger than the number of positive samples
			ptd_num = 0
			ptd_loss = 0.0
			ptd_accuracy = 0.0
			ntd_num = 0
			ntd_loss = 0.0
			ntd_accuracy = 0.0
			random.shuffle(positive_train_indices)
			random.shuffle(negative_train_indices)
			#positive_batch_size = 1
			positive_batch_size = (1-int(ALL_NEGATIVE))*(int(BATCH_SIZE/2/aug_proportion)-1) + 1	#if augmentatiion implemented then set batch size to 1 or half of BATCH_SIZE
			#negative_batch_size = int(positive_batch_size * np_proportion)
			for pbi in tqdm(range(0, positive_train_num, positive_batch_size)):
				#if AUGMENTATION:
				#	print("training process:%d/%d" %(pbi, positive_train_num))
				posbatchend = min(pbi+positive_batch_size, positive_train_num)
				for pti in range(pbi, posbatchend):
					data_index = positive_train_indices[pti]
					pfile = tpfiles[data_index]
					#pfile = 'luna_cubes_56_overbound/subset5/npy/1.3.6.1.4.1.14519.5.2.1.6279.6001.112740418331256326754121315800_34_ob_annotations.npy'
					if pfile.split('/')[0].find("luna")>=0:
						patient_uid, nodule_diameter = mt.get_annotation_informations(pfile, "luna_cubes_56_overbound/luna_annotations.csv")
					elif pfile.split('/')[0].find("tianchi")>=0:
						patient_uid, nodule_diameter = mt.get_annotation_informations(pfile, "tianchi_cubes_56_overbound/tianchi_annotations.csv")
					else:
						patient_uid = mt.get_volume_informations(pfile)[0]
						nodule_diameter = 0
					positive_data = np.load(pfile)
					if "positive_batch" not in dir():
						#positive_batch = mt.extract_volumes(positive_data, centering=CENTERING, nodule_diameter=nodule_diameter, scale_augment=SCALE_AUGMENTATION, translation_augment=TRANSLATION_AUGMENTATION, rotation_augment=ROTATION_AUGMENTATION, flip_augment=FLIP_AUGMENTATION)
						positive_batch = extract_volumes(positive_data, nodule_diameter=nodule_diameter)
					else:
						#positive_batch = np.concatenate((positive_batch, mt.extract_volumes(positive_data, centering=CENTERING, nodule_diameter=nodule_diameter, scale_augment=SCALE_AUGMENTATION, translation_augment=TRANSLATION_AUGMENTATION, rotation_augment=ROTATION_AUGMENTATION, flip_augment=FLIP_AUGMENTATION)), axis=0)
						positive_batch = np.concatenate((positive_batch, extract_volumes(positive_data, nodule_diameter=nodule_diameter)), axis=0)
					del positive_data

				negative_batch = []
				if posbatchend==positive_train_num and ALL_NEGATIVE:
					negbatchend = negative_train_num
				else:
					negbatchend = min(int((pbi+positive_batch_size)*np_proportion), negative_train_num)
				for nti in range(int(pbi*np_proportion), negbatchend):
					data_index = negative_train_indices[nti]
					nfile = tnfiles[data_index]
					negative_data = np.load(nfile)
					ndcenter = np.int_([negative_data.shape[0]/2, negative_data.shape[1]/2, negative_data.shape[2]/2])
					negative_data = negative_data[ndcenter[0]-region_half:ndcenter[0]+REGION_SIZE-region_half, ndcenter[1]-region_half:ndcenter[1]+REGION_SIZE-region_half, ndcenter[2]-region_half:ndcenter[2]+REGION_SIZE-region_half]
					#negative_data = np.reshape(negative_data, (1, negative_data.shape[0], negative_data.shape[1], negative_data.shape[2]))
					negative_batch.append(negative_data)
					#if "negative_batch" not in dir():
					#	negative_batch = negative_data
					#else:
					#	negative_batch = np.concatenate((negative_batch, negative_data), axis=0)
					del negative_data
			
				train_data = np.zeros(shape=(positive_batch.shape[0]+len(negative_batch), positive_batch.shape[1], positive_batch.shape[2], positive_batch.shape[3]), dtype=float)
				train_label = np.zeros(shape=(positive_batch.shape[0]+len(negative_batch), 2), dtype=float)
				batch_random = np.random.permutation(positive_batch.shape[0]+len(negative_batch))
				#batch_random = np.arange(positive_batch.shape[0]+len(negative_batch))
				for bi in range(batch_random.size):
					batch_index = batch_random[bi]
					if batch_index<positive_batch.shape[0]:
						#train_data[bi] = (positive_batch[batch_index] - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
						train_data[bi] = medical_normalization(positive_batch[batch_index])
						train_label[bi][0] = 1
					else:
						#train_data[bi] = (negative_batch[batch_index-positive_batch.shape[0]] - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
						train_data[bi] = medical_normalization(negative_batch[batch_index-positive_batch.shape[0]])
						train_label[bi][1] = 1
				for bi in range(0, train_data.shape[0], BATCH_SIZE):
					batch_size = min(BATCH_SIZE, train_data.shape[0]-bi)
					data_batch = train_data[bi:bi+batch_size]
					label_batch = train_label[bi:bi+batch_size]
					_, losses, accuracies, bn_stat, train_summary = sess.run([train, loss, correct_prediction, bn_params, train_merge], {volume_input: data_batch, real_label: label_batch})
					#parameters2 = sess.run(variables, {volume_input: data_batch, real_label: label_batch})
					tensorboard_writer.add_summary(train_summary, epoch)
					for di in range(batch_size):
						#print("batch:%d/%d train_loss:%f train_acc:%f label:%d" % (bi+di, train_data.shape[0], losses[di], accuracies[di], label_batch[di][0]))
						if label_batch[di][0]>label_batch[di][1]:
							ptd_num += 1
							ptd_loss += losses[di]
							ptd_accuracy += accuracies[di]
						else:
							ntd_num += 1
							ntd_loss += losses[di]
							ntd_accuracy += accuracies[di]
					if batch_size > 1:
						batch_num += 1
						bn_stat = np.array(bn_stat)
						bn_stat[:,1] = bn_stat[:,1] * batch_size / (batch_size-1)	#unbiased variance estimate
						bn_stats += bn_stat

					#print("train_loss:%f train_acc:%f" % (train_loss/train_num, train_accuracy/train_num))
				if 'negative_batch' in dir():
					del negative_batch
				del positive_batch
				del train_data

			if ptd_num > 0:
				ptd_loss /= ptd_num
				ptd_accuracy /= ptd_num
			if ntd_num > 0:
				ntd_loss /= ntd_num
				ntd_accuracy /= ntd_num
			#tensorboard_writer.add_summary(summary, epoch)

			pvd_num = 0
			pvd_loss = 0.0
			pvd_accuracy = 0.0
			nvd_num = 0
			nvd_loss = 0.0
			nvd_accuracy = 0.0
			random.shuffle(positive_val_indices)
			random.shuffle(negative_val_indices)
			#positive_batch_size = (1 - int(AUGMENTATION)) * (int(BATCH_SIZE / 2) - 1) + 1  # if augmentatiion implemented then set batch size to 1 or half of BATCH_SIZE
			#negative_batch_size = int(math.ceil(positive_batch_size * np_proportion))
			for pbi in tqdm(range(0, positive_val_num, positive_batch_size)):
				#if AUGMENTATION:
				#	print("validation process:%d/%d" % (pbi, positive_val_num))
				posbatchend = min(pbi+positive_batch_size, positive_val_num)
				for pvi in range(pbi, posbatchend):
					data_index = positive_val_indices[pvi]
					pfile = vpfiles[data_index]
					if pfile.split('/')[0].find("luna") >= 0:
						patient_uid, nodule_diameter = mt.get_annotation_informations(pfile,
													      "luna_cubes_56_overbound/luna_annotations.csv")
					elif pfile.split('/')[0].find("tianchi") >= 0:
						patient_uid, nodule_diameter = mt.get_annotation_informations(pfile,
													      "tianchi_cubes_56_overbound/tianchi_annotations.csv")
					else:
						patient_uid = mt.get_volume_informations(pfile)[0]
						nodule_diameter = 0
					positive_data = np.load(pfile)
					if "positive_batch" not in dir():
						positive_batch = extract_volumes(positive_data, nodule_diameter=nodule_diameter)
					else:
						positive_batch = np.concatenate((positive_batch, extract_volumes(positive_data, nodule_diameter=nodule_diameter)), axis=0)
					del positive_data
					
				negative_batch = []
				if posbatchend==positive_val_num and ALL_NEGATIVE:
					negbatchend = negative_val_num
				else:
					negbatchend = min(int((pbi+positive_batch_size)*np_proportion), negative_val_num)
				for nvi in range(int(pbi*np_proportion), negbatchend):
					data_index = negative_val_indices[nvi]
					nfile = vnfiles[data_index]
					negative_data = np.load(nfile)
					ndcenter = np.int_([negative_data.shape[0]/2, negative_data.shape[1]/2, negative_data.shape[2]/2])
					negative_data = negative_data[ndcenter[0]-region_half:ndcenter[0]+REGION_SIZE-region_half, ndcenter[1]-region_half:ndcenter[1]+REGION_SIZE-region_half, ndcenter[2]-region_half:ndcenter[2]+REGION_SIZE-region_half]
					#negative_data = np.reshape(negative_data, (1, negative_data.shape[0], negative_data.shape[1], negative_data.shape[2]))
					negative_batch.append(negative_data)
					#if "negative_batch" not in dir():
					#	negative_batch = negative_data
					#else:
					#	negative_batch = np.concatenate((negative_batch, negative_data), axis=0)
					del negative_data

				val_data = np.zeros(shape=(
				positive_batch.shape[0] + len(negative_batch), positive_batch.shape[1],
				positive_batch.shape[2], positive_batch.shape[3]), dtype=float)
				val_label = np.zeros(shape=(positive_batch.shape[0] + len(negative_batch), 2),
						       dtype=float)
				batch_random = np.random.permutation(positive_batch.shape[0] + len(negative_batch))
				#batch_random = np.arange(positive_batch.shape[0]+len(negative_batch))
				for bi in range(batch_random.size):
					batch_index = batch_random[bi]
					if batch_index < positive_batch.shape[0]:
						#val_data[bi] = (positive_batch[batch_index] - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
						val_data[bi] = medical_normalization(positive_batch[batch_index])
						val_label[bi][0] = 1
					else:
						#val_data[bi] = (negative_batch[batch_index - positive_batch.shape[0]] - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
						val_data[bi] = medical_normalization(negative_batch[batch_index - positive_batch.shape[0]])
						val_label[bi][1] = 1
				for bi in range(0, val_data.shape[0], BATCH_SIZE):
					batch_size = min(BATCH_SIZE, val_data.shape[0] - bi)
					data_batch = val_data[bi:bi + batch_size]
					label_batch = val_label[bi:bi + batch_size]
					# rb1, bb1, wc1, wc2, oc1, ob1, hc1, hc2, hc3, of1, of2, so = sess.run([r_bn1, b_bn1, w_conv1, w_conv2, out_conv1, out_bn1, hidden_conv1, hidden_conv2, hidden_conv3, out_fc1, out_fc2, softmax_out], {volume_input: data_batch, real_label: label_batch})
					losses, accuracies, bn_stat, val_summary = sess.run(
						[loss, correct_prediction, bn_params, val_merge],
						{volume_input: data_batch, real_label: label_batch})
					tensorboard_writer.add_summary(val_summary, epoch)
					for di in range(batch_size):
						# print("batch:%d/%d val_loss:%f val_acc:%f label:%d" % (bi+di, val_data.shape[0], losses[di], accuracies[di], label_batch[di][0]))
						if label_batch[di][0]>label_batch[di][1]:
							pvd_num += 1
							pvd_loss += losses[di]
							pvd_accuracy += accuracies[di]
						else:
							nvd_num += 1
							nvd_loss += losses[di]
							nvd_accuracy += accuracies[di]
					if batch_size > 1:
						batch_num += 1
						bn_stat = np.array(bn_stat)
						bn_stat[:,1] = bn_stat[:,1] * batch_size / (batch_size-1)	#unbiased variance estimate
						bn_stats += bn_stat
					# print("val_loss:%f val_acc:%f" % (val_loss/val_num, val_accuracy/val_num))
				if 'negative_batch' in dir():
					del negative_batch
				del positive_batch
				del val_data

			if batch_num > 0:
				bn_stats /= batch_num
			if pvd_num > 0:
				pvd_loss /= pvd_num
				pvd_accuracy /= pvd_num
			if nvd_num > 0:
				nvd_loss /= nvd_num
				nvd_accuracy /= nvd_num

			#tensorboard_writer.add_summary(summary, epoch)
			print("positive_train_loss:%f positive_train_acc:%f negative_train_loss:%f negative_train_acc:%f\npositive_val_loss:%f positive_val_acc:%f negative_val_loss:%f negative_val_acc:%f" % (
				ptd_loss, ptd_accuracy, ntd_loss,
				ntd_accuracy, pvd_loss, pvd_accuracy, nvd_loss,
				nvd_accuracy))
			if (pvd_accuracy+nvd_accuracy)/2>=best_accuracy:
				best_accuracy = (pvd_accuracy+nvd_accuracy)/2
				best_epoch = overall_epoch
				#best_dir = net_store_path+"/"+net_store_name+"_best"
				best_dir = net_store_path+"/"+"best"
				if os.access(best_dir, os.F_OK):
					shutil.rmtree(best_dir)
				os.mkdir(best_dir)
				#saver.save(sess, best_dir+"/"+net_store_name+"_best")
				saver.save(sess, best_dir+"/best")
				np.save(best_dir+"/batch_normalization_statistic.npy", bn_stats)
			training_messages.append(
				[overall_epoch + 1, ptd_loss, ptd_accuracy, ntd_loss,
				 ntd_accuracy, pvd_loss, pvd_accuracy, nvd_loss,
				 nvd_accuracy])
			# write training process to a logger file
			logger = open(net_store_path + "/" + net_store_name + ".log", 'w')
			logger.write("validation rate:{}\n" .format(VALIDATION_RATE))
			logger.write("batch size:{}\n" .format(BATCH_SIZE))
			logger.write("region size:{}\n" .format(REGION_SIZE))
			logger.write("max bound:{}\n" .format(MAX_BOUND))
			logger.write("min bound:{}\n" .format(MIN_BOUND))
			logger.write("pixel mean:{}\n" .format(PIXEL_MEAN))
			logger.write("normalization crop:{}\n" .format(NORM_CROP))
			logger.write("loss balancing:{}\n" .format(LOSS_BALANCING))
			logger.write("focal loss:{}" .format(FOCAL_LOSS))
			logger.write("scale augmentation:{}\n" .format(SCALE_AUGMENTATION))
			logger.write("translation augmentation:{}\n" .format(TRANSLATION_AUGMENTATION))
			logger.write("rotation augmentation:{}\n" .format(ROTATION_AUGMENTATION))
			logger.write("flip augmentation:{}\n" .format(FLIP_AUGMENTATION))
			logger.write("centering augmentation:{}\n" .format(CENTERING))
			# logger.write("initial_learning_rate:%f decay_rate:%f decay_epoch:%d\n" %(INITIAL_LEARNING_RATE, DECAY_LEARNING_RATE, DECAY_EPOCH))
			logger.write("learning_rates:{}\n" .format(learning_rates))
			logger.write("epochs:%d %d %d\n" % (epochs[0], epochs[1], epochs[2]))
			logger.write(
				"epoch pos_train_loss pos_train_acc neg_train_loss neg_train_acc pos_val_loss pos_val_acc neg_val_loss neg_val_acc\n")
			for tm in range(len(training_messages)):
				logger.write("%d %f %f %f %f %f %f %f %f\n" % (
				training_messages[tm][0], training_messages[tm][1], training_messages[tm][2],
				training_messages[tm][3], training_messages[tm][4], training_messages[tm][5],
				training_messages[tm][6], training_messages[tm][7], training_messages[tm][8]))
			logger.write("best epoch:%d" %(best_epoch+1))
			logger.close()
			if SNAPSHOT_EPOCH>0 and (overall_epoch+1)%SNAPSHOT_EPOCH==0:
				#snapshot_dir = net_store_path+"/"+net_store_name+"_epoch"+str(overall_epoch+1)
				snapshot_dir = net_store_path+"/"+"epoch"+str(overall_epoch+1)
				if os.access(snapshot_dir, os.F_OK):
					shutil.rmtree(snapshot_dir)
				os.mkdir(snapshot_dir)
				saver.save(sess, snapshot_dir+"/"+"epoch"+str(overall_epoch+1))
				np.save(snapshot_dir+"/batch_normalization_statistic.npy", bn_stats)
				print("Snapshot saved in: %s" %(snapshot_dir))
			overall_epoch += 1
	saver.save(sess, net_store_path + "/" + net_store_name)
	np.save(net_store_path + "/batch_normalization_statistic.npy", bn_stats)
sess.close()
print("Overall training done!")
print("The network is saved as:%s" %(net_store_name))