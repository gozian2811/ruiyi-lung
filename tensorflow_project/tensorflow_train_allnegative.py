from tensorflow.python import debug as tf_debug
import tensorflow as tf
import numpy as np
import os
import sys
import shutil
import glob
import math
import random
sys.path.append("/home/fyl/programs/lung_project")
from toolbox import BasicTools as bt
from toolbox import MITools as mt
from toolbox import Data_Augmentation as da
from toolbox import TensorflowTools as tft

constants = bt.read_constants("./constants2.txt")
os.environ['CUDA_VISIBLE_DEVICES'] = constants["GPU_DEVICE"]
REGION_SIZE = constants["REGION_SIZE"]
MAX_BOUND = float(constants["MAX_BOUND"])
MIN_BOUND = float(constants["MIN_BOUND"])
PIXEL_MEAN = float(constants["PIXEL_MEAN"])

#NUM_EPOCH = 200
SNAPSHOT_EPOCH = 20
#DECAY_EPOCH = 0
#INITIAL_LEARNING_RATE = 0.001
#DECAY_LEARNING_RATE = 1.0

BATCH_SIZE = 30
VALIDATION_RATE = 0.2

AUGMENTATION = False

store_path = "models_tensorflow"
net_store_name = "luna-tianchi-slh_3D_l3454-512-2_bn_fl"
#net_init_file = "models_tensorflow/luna-tianchi-slh_3D_l3454-512-2_bn_aug/init/luna-tianchi-slh_3D_l3454-512-2_bn_best"
data_dir1 = "luna_cubes_64_overbound"
#data_dir2 = "tianchi_cubes_56_overbound"
#data_dir3 = "slh_cubes_56_overbound"
annotation = "/data/fyl/datasets/LUNA16/csvfiles/annotations.csv"
annotation2 = "tianchi_cubes_56_overbound/tianchi_annotations.csv"
net_store_path = store_path + "/" + net_store_name
tensorboard_path = net_store_path + "/tensorboard/"
#pfilelist_path = "models_tensorflow/luna_3D_l3454-512-2_bn5/pfilelist.log"

if not os.access(net_store_path, os.F_OK):
	os.makedirs(net_store_path)
if os.access(tensorboard_path, os.F_OK):
	shutil.rmtree(tensorboard_path)

#data arrangement
#train_sets = ["subset0", "subset1", "subset2", "subset3", "subset4", "subset5", "subset6", "subset7", "subset8"]
#train_sets2 = ["train", "val"]
train_sets = [""]

if "pfilelist_path" in dir() and os.path.exists(pfilelist_path):
	print("read pfilelist from: %s" %(pfilelist_path))
	pfilelist_file = open(pfilelist_path, "r")
	pfiles = pfilelist_file.readlines()
	for pfi in range(len(pfiles)):
		pfiles[pfi] = pfiles[pfi][:-1]
	nfiles = []
	for set in train_sets:
		train_dir = os.path.join(data_dir1, set)
		ndir = os.path.join(train_dir,"npy_non","*.npy")
		nfiles.extend(glob.glob(ndir))
else:
	pfiles = []
	nfiles = []
	for set in train_sets:
		train_dir = os.path.join(data_dir1, set)
		pdir = os.path.join(train_dir,"npy","*.npy")
		pfiles.extend(glob.glob(pdir))
		ndir = os.path.join(train_dir,"npy_non","*.npy")
		nfiles.extend(glob.glob(ndir))
	if "data_dir2" in dir():
		for set in train_sets2:
			train_dir2 = os.path.join(data_dir2, set)
			pdir = os.path.join(train_dir2,"npy","*.npy")
			pfiles.extend(glob.glob(pdir))
	if "data_dir3" in dir():
		pfiles.extend(glob.glob(os.path.join(data_dir3,"npy","*.npy")))
	
num_positive = len(pfiles)
num_negative = len(nfiles)
num_total = num_positive + num_negative
#num_positive = 5
#num_negative = 20
#num_negative = 1000 * num_positive
if num_total==0:
	print("no training file found")
	exit()
fileindices = [i for i in range(num_total)]
random.shuffle(fileindices)
dfile_storage = open(net_store_path + "/datafilelist.log", "w")
for fi in fileindices:
	findex = fileindices[fi]
	if findex < num_positive:
		dfile = pfiles[findex]
	else:
		dfile = nfiles[findex-num_positive]
	dfile_storage.write("%s\n" %(dfile))
dfile_storage.close()

val_num = int(num_total * VALIDATION_RATE)
#positive_val_num = 1
train_num = num_total - val_num
train_indices = fileindices[:train_num]
val_indices = fileindices[train_num:]

#net construct
volume_input = tf.placeholder(tf.float32, [None, REGION_SIZE, REGION_SIZE, REGION_SIZE])
volume_reshape = tf.reshape(volume_input, [-1, REGION_SIZE, REGION_SIZE, REGION_SIZE, 1])
real_label = tf.placeholder(tf.float32, [None, 2])
#r_bn1, b_bn1, w_conv1, w_conv2, out_conv1, out_bn1, hidden_conv1, hidden_conv2, hidden_conv3, out_fc1, out_fc2, softmax_out = tft.volume_bnnet2_l6_56(volume_reshape)
out_fc2, softmax_out = tft.volume_bnnet_flbias_l6_56(volume_reshape)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = out_fc2, labels = real_label)
#pm = tf.pow(tf.constant(-1, dtype=tf.float32), real_label[:,0])
#prediction_pm = tf.multiply(pm, softmax_out[:,0])
#addition = tf.add(real_label[:,0], prediction_pm)
#modulation = tf.pow(addition, tf.constant(2, dtype=tf.float32))
modulation = tf.pow(tf.add(real_label[:,0], tf.multiply(tf.pow(tf.constant(-1, dtype=tf.float32), real_label[:,0]), softmax_out[:,0])), tf.constant(2, dtype=tf.float32))
focal_loss = tf.multiply(modulation, cross_entropy)
batch_loss = tf.reduce_mean(focal_loss)
tf.summary.scalar('loss', batch_loss)
correct_prediction = tf.equal(tf.argmax(softmax_out, 1), tf.argmax(real_label, 1))
batch_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('acc', batch_accuracy)
merged = tf.summary.merge_all()
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
epochs = [10, 200, 0]
learning_rates = [0.01, 0.001, 0.0001]
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

saver = tf.train.Saver(max_to_keep=None)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
	#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
	#sess.add_tensor_filter("debug_tensor_watch", tf_debug.add_debug_tensor_watch)
	if 'net_init_file' not in dir() or net_init_file is None:
		sess.run(tf.global_variables_initializer())
	else:
		saver.restore(sess, net_init_file)
	train_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
	#best_accuracy = 0.0
	#best_epoch = 0
	overall_epoch = 0
	training_messages = []
	for train_stage in range(len(trains)):
		train = trains[train_stage]
		for epoch in range(epochs[train_stage]):
			print('epoch:%d' %(overall_epoch+1))
			random.shuffle(train_indices)
			#random.shuffle(val_indices)

			# the number of negative samples is in default larger than the number of positive samples
			#train_data_num = 0
			#train_loss = 0.0
			#train_accuracy = 0.0
			positive_train_num = 0
			positive_train_loss = 0.0
			positive_train_accuracy = 0.0
			negative_train_num = 0
			negative_train_loss = 0.0
			negative_train_accuracy = 0.0
			import_batch_size = 900
			for pbi in range(0, train_num, import_batch_size):
				print("training process:%d/%d" %(pbi, train_num))
				for pti in range(pbi, min(pbi+import_batch_size, train_num)):
					data_index = train_indices[pti]
					if data_index < num_positive:
						pfile = pfiles[data_index]
						isnodule = True
					else:
						pfile = nfiles[data_index - num_positive]
						isnodule = False
					data_volume = np.load(pfile)
					if isnodule:
						if pfile.split('/')[0].find("luna")>=0:
							patient_uid, nodule_diameter = mt.get_luna_annotation_informations(pfile, annotation)
						elif pfile.split('/')[0].find("tianchi")>=0:
							patient_uid, nodule_diameter = mt.get_luna_annotation_informations(pfile, annotation2)
						else:
							patient_uid = mt.get_volume_informations(pfile)[0]
							nodule_diameter = 0
						data_volume = da.extract_volumes(data_volume, np.int_([REGION_SIZE, REGION_SIZE, REGION_SIZE]), nodule_diameter=nodule_diameter, scale_augment=AUGMENTATION, translation_augment=AUGMENTATION, rotation_augment=AUGMENTATION)
					else:
						data_volume = mt.local_crop(data_volume, np.rint(np.array(data_volume.shape)/2).astype(int), REGION_SIZE)
						data_volume = data_volume.reshape((1, data_volume.shape[0], data_volume.shape[1], data_volume.shape[2]))
					data_label = np.zeros(shape=(data_volume.shape[0], 2), dtype=float)
					data_label[:,1-int(isnodule)] = 1
					if "data_volumes" not in dir():
						data_volumes = data_volume
						data_labels = data_label
					else:
						data_volumes = np.concatenate((data_volumes, data_volume), axis=0)
						data_labels = np.concatenate((data_labels, data_label), axis=0)

				train_data = np.zeros(shape=data_volumes.shape, dtype=float)
				train_label = np.zeros(shape=data_labels.shape, dtype=float)
				batch_random = np.random.permutation(data_volumes.shape[0])
				for bi in range(batch_random.size):
					batch_index = batch_random[bi]
					train_data[bi] = (data_volumes[batch_index] - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
					train_label[bi] = data_labels[batch_index]
				#train_data_num += train_data.shape[0]
				for bi in range(0, train_data.shape[0], BATCH_SIZE):
					batch_size = min(BATCH_SIZE, train_data.shape[0]-bi)
					data_batch = train_data[bi:bi+batch_size]
					label_batch = train_label[bi:bi+batch_size]
					#rb1, bb1, wc1, wc2, oc1, ob1, hc1, hc2, hc3, of1, of2, so = sess.run([r_bn1, b_bn1, w_conv1, w_conv2, out_conv1, out_bn1, hidden_conv1, hidden_conv2, hidden_conv3, out_fc1, out_fc2, softmax_out], {volume_input: data_batch, real_label: label_batch})
					#s, pp, ad, ml = sess.run([pm, prediction_pm, addition, modulation], {volume_input:data_batch, real_label:label_batch})
					_, losses, accuracies, summary = sess.run([train, focal_loss, correct_prediction, merged], {volume_input: data_batch, real_label: label_batch})
					for di in range(data_batch.shape[0]):
						#print("batch:%d/%d train_loss:%f train_acc:%f label:%d" % (bi+di, train_data.shape[0], losses[di], accuracies[di], label_batch[di][0]))
						if label_batch[di][0]>label_batch[di][1]:
							positive_train_num += 1
							positive_train_loss += losses[di]
							positive_train_accuracy += accuracies[di]
						else:
							negative_train_num += 1
							negative_train_loss += losses[di]
							negative_train_accuracy += accuracies[di]
					#print("train_loss:%f train_acc:%f" % (train_loss/train_data_num, train_accuracy/train_data_num))
				del data_volumes
				del data_labels
				del train_data
				del train_label

			if positive_train_num > 0:
				positive_train_loss /= positive_train_num
				positive_train_accuracy /= positive_train_num
			if negative_train_num > 0:
				negative_train_loss /= negative_train_num
				negative_train_accuracy /= negative_train_num
			#train_writer.add_summary(summary, epoch)

			#val_data_num = 0
			#val_loss = 0.0
			#val_accuracy = 0.0
			positive_val_num = 0
			positive_val_loss = 0.0
			positive_val_accuracy = 0.0
			negative_val_num = 0
			negative_val_loss = 0.0
			negative_val_accuracy = 0.0
			for pbi in range(0, val_num, import_batch_size):
				print("validation process:%d/%d" %(pbi, val_num))
				for pti in range(pbi, min(pbi+import_batch_size, val_num)):
					data_index = val_indices[pti]
					if data_index < num_positive:
						pfile = pfiles[data_index]
						isnodule = True
					else:
						pfile = nfiles[data_index - num_positive]
						isnodule = False
					data_volume = np.load(pfile)
					if isnodule:
						if pfile.split('/')[0].find("luna")>=0:
							patient_uid, nodule_diameter = mt.get_annotation_informations(pfile, annotation)
						elif pfile.split('/')[0].find("tianchi")>=0:
							patient_uid, nodule_diameter = mt.get_annotation_informations(pfile, annotation2)
						else:
							patient_uid = mt.get_volume_informations(pfile)[0]
							nodule_diameter = 0
						data_volume = da.extract_volumes(data_volume, np.int_([REGION_SIZE, REGION_SIZE, REGION_SIZE]), nodule_diameter=nodule_diameter, scale_augment=AUGMENTATION, translation_augment=AUGMENTATION, rotation_augment=AUGMENTATION)
					else:
						data_volume = mt.local_crop(data_volume, np.rint(np.array(data_volume.shape)/2).astype(int), REGION_SIZE)
						data_volume = data_volume.reshape((1, data_volume.shape[0], data_volume.shape[1], data_volume.shape[2]))
					data_label = np.zeros(shape=(data_volume.shape[0], 2), dtype=float)
					data_label[:,1-int(isnodule)] = 1
					if "data_volumes" not in dir():
						data_volumes = data_volume
						data_labels = data_label
					else:
						data_volumes = np.concatenate((data_volumes, data_volume), axis=0)
						data_labels = np.concatenate((data_labels, data_label), axis=0)

				val_data = np.zeros(shape=data_volumes.shape, dtype=float)
				val_label = np.zeros(shape=data_labels.shape, dtype=float)
				batch_random = np.random.permutation(data_volumes.shape[0])
				for bi in range(batch_random.size):
					batch_index = batch_random[bi]
					val_data[bi] = (data_volumes[batch_index] - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
					val_label[bi] = data_labels[batch_index]
				#val_data = (val_data - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
				#val_data_num += val_data.shape[0]
				for bi in range(0, val_data.shape[0], BATCH_SIZE):
					batch_size = min(BATCH_SIZE, val_data.shape[0]-bi)
					data_batch = val_data[bi:bi+batch_size]
					label_batch = val_label[bi:bi+batch_size]
					#rb1, bb1, wc1, wc2, oc1, ob1, hc1, hc2, hc3, of1, of2, so = sess.run([r_bn1, b_bn1, w_conv1, w_conv2, out_conv1, out_bn1, hidden_conv1, hidden_conv2, hidden_conv3, out_fc1, out_fc2, softmax_out], {volume_input: data_batch, real_label: label_batch})
					#s, pp, ad, ml = sess.run([pm, prediction_pm, addition, modulation], {volume_input:data_batch, real_label:label_batch})
					losses, accuracies, summary = sess.run([focal_loss, correct_prediction, merged], {volume_input: data_batch, real_label: label_batch})
					for di in range(data_batch.shape[0]):
						#print("batch:%d/%d val_loss:%f val_acc:%f label:%d" % (bi+di, val_data.shape[0], losses[di], accuracies[di], label_batch[di][0]))
						if label_batch[di][0]>label_batch[di][1]:
							positive_val_num += 1
							positive_val_loss += losses[di]
							positive_val_accuracy += accuracies[di]
						else:
							negative_val_num += 1
							negative_val_loss += losses[di]
							negative_val_accuracy += accuracies[di]
					#print("val_loss:%f val_acc:%f" % (val_loss/val_data_num, val_accuracy/val_data_num))
				del data_volumes
				del data_labels
				del val_data
				del val_label

			if positive_val_num > 0:
				positive_val_loss /= positive_val_num
				positive_val_accuracy /= positive_val_num
			if negative_val_num > 0:
				negative_val_loss /= negative_val_num
				negative_val_accuracy /= negative_val_num
			train_writer.add_summary(summary, epoch)
			print("positive_train_loss:%f positive_train_acc:%f negative_train_loss:%f negative_train_acc:%f\npositive_val_loss:%f positive_val_acc:%f negative_val_loss:%f negative_val_acc:%f" % (positive_train_loss, positive_train_accuracy, negative_train_loss, negative_train_accuracy, positive_val_loss, positive_val_accuracy, negative_val_loss, negative_val_accuracy))
			#if val_accuracy>=best_accuracy:
			#	best_accuracy = val_accuracy
			#	best_epoch = overall_epoch
			#	saver.save(sess, net_store_path+"/"+net_store_name+"_best")
			training_messages.append([overall_epoch+1, positive_train_loss, positive_train_accuracy, negative_train_loss, negative_train_accuracy, positive_val_loss, positive_val_accuracy, negative_val_loss, negative_val_accuracy])
			#write training process to a logger file
			logger = open(net_store_path + "/" + net_store_name + ".log", 'w')
			logger.write("random validation\n")
			#logger.write("initial_learning_rate:%f decay_rate:%f decay_epoch:%d\n" %(INITIAL_LEARNING_RATE, DECAY_LEARNING_RATE, DECAY_EPOCH))
			logger.write("learning_rates:%f %f %f\n" %(learning_rates[0], learning_rates[1], learning_rates[2]))
			logger.write("epochs:%d %d %d\n" %(epochs[0], epochs[1], epochs[2]))
			logger.write("epoch pos_train_loss pos_train_acc neg_train_loss neg_train_acc pos_val_loss pos_val_acc neg_val_loss neg_val_acc\n")
			for tm in range(len(training_messages)):
				logger.write("%d %f %f %f %f %f %f %f %f\n" %(training_messages[tm][0], training_messages[tm][1], training_messages[tm][2], training_messages[tm][3], training_messages[tm][4], training_messages[tm][5], training_messages[tm][6], training_messages[tm][7], training_messages[tm][8]))
			#logger.write("best epoch:%d" %(best_epoch+1))
			logger.close()
			if SNAPSHOT_EPOCH>0 and (overall_epoch+1)%SNAPSHOT_EPOCH==0:
				snapshot_dir = net_store_path+"/"+net_store_name+"_epoch"+str(overall_epoch+1)
				if os.access(snapshot_dir, os.F_OK):
					shutil.rmtree(snapshot_dir)
				os.mkdir(snapshot_dir)
				saver.save(sess, snapshot_dir+"/"+net_store_name+"_epoch"+str(overall_epoch+1))
			overall_epoch += 1
	saver.save(sess, net_store_path + "/" + net_store_name)
sess.close()
print("Overall training done!")
print("The network is saved as:%s" %(net_store_name))
