import tensorflow as tf
import numpy as np
import functools as ft
import os
import sys
import shutil
import glob
import math
import random
import time
#from tensorflow.python import debug as tf_debug
from config import config
sys.path.append('/home/fyl/programs/lung_project')
from toolbox import BasicTools as bt
from toolbox import MITools as mt
from toolbox import LIDCTools as lt
from toolbox import Data_Augmentation as da
#from toolbox import TensorflowTools as tft
from toolbox import TensorflowMultiCrop as tmc
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
SNAPSHOT_EPOCH = 1
#DECAY_EPOCH = 0
#INITIAL_LEARNING_RATE = 0.001
#DECAY_LEARNING_RATE = 1.0

REGION_SIZE = 64
DATA_BATCH_SIZE = 20
TRAIN_BATCH_SIZE = 24

SCALE_AUGMENTATION = False
TRANSLATION_AUGMENTATION = True
ROTATION_AUGMENTATION = True
FLIP_AUGMENTATION = True
AUGMENTATION = SCALE_AUGMENTATION or TRANSLATION_AUGMENTATION or ROTATION_AUGMENTATION or FLIP_AUGMENTATION

print("data batch size:{}" .format(DATA_BATCH_SIZE))
print("train batch size:{}" .format(TRAIN_BATCH_SIZE))
print("region size:{}" .format(REGION_SIZE))
print("max bound:{}" .format(MAX_BOUND))
print("min bound:{}" .format(MIN_BOUND))
print("pixel mean:{}" .format(PIXEL_MEAN))
print("normalization crop:{}" .format(NORM_CROP))
print("loss balancing:{}" .format(LOSS_BALANCING))
print("focal loss:{}" .format(FOCAL_LOSS))

store_path = "models_tensorflow"
load_path = "models_tensorflow"
net_store_name = "lidc_3D_64_multi_crop_net_32-32-32_32-2_aug"
net_store_path = store_path + "/" + net_store_name
tensorboard_path = net_store_path + "/tensorboard/"
#net_init_name = "lidc_3D_multi_crop_net_64_aug3"

if os.access(net_store_path, os.F_OK):
	shutil.rmtree(net_store_path)
os.makedirs(net_store_path)
if os.access(tensorboard_path, os.F_OK):
	shutil.rmtree(tensorboard_path)

#data arrangement
if 'net_init_name' in dir():
	net_init_path = load_path + "/" + net_init_name
	net_init_file = net_init_path + "/epoch12/epoch12"
	filelist_train_path = net_init_path + "/filelist_train.log"
	filelist_val_path = net_init_path + "/filelist_val.log"
	filelist_test_path = net_init_path + "/filelist_test.log"
	train_files = bt.filelist_load(filelist_train_path)
	val_files = bt.filelist_load(filelist_val_path)
	test_files = bt.filelist_load(filelist_test_path)
else:
	lidc_dir="../data_samples/lidc_cubes_64_overbound_ipris"
	files_lists, _ = lt.filelist_training(lidc_dir, shuffle=True, cross_fold=5, test_fold=5)
	train_files = files_lists['train']
	val_files = files_lists['val']
	test_files = files_lists['test']
bt.filelist_store(train_files, net_store_path + '/' + 'filelist_train.log')
bt.filelist_store(val_files, net_store_path + '/' + 'filelist_val.log')
bt.filelist_store(test_files, net_store_path + '/' + 'filelist_test.log')

train_num = len(train_files)
val_num = len(val_files)
#train_num = 32
#val_num = 8
num_positive_train, num_negative_train, _ = lt.category_statistics(train_files)
num_positive_val, num_negative_val, _ = lt.category_statistics(val_files)

train_indices = [i for i in range(train_num)]
val_indices = [i for i in range(val_num)]

#net construct
volume_input = tf.placeholder(tf.float32, [None, REGION_SIZE, REGION_SIZE, REGION_SIZE])
volume_reshape = tf.reshape(volume_input, [-1, REGION_SIZE, REGION_SIZE, REGION_SIZE, 1])
real_label = tf.placeholder(tf.float32, [None, 2])
positive_rate = (num_positive_train + num_positive_val) / float(num_positive_train + num_positive_val + num_negative_train + num_negative_val)
#net_outs, _, bn_params = tft.volume_bndo_flbias_l5_30(volume_reshape, True, positive_confidence, dropout_rate=0.3, batch_normalization_statistic=True, bn_params=None)
net_outs, _ = tmc.multi_crop_net(volume_reshape, channels=[[32,32,32],[32,2]], poolings=[1,1,2], weight_decay_coefficiency=0.0005)
loss = tf.nn.softmax_cross_entropy_with_logits(logits = net_outs['last_out'], labels = real_label)
if type(LOSS_BALANCING)==float:
	balancing_factor = LOSS_BALANCING
else:
	balancing_factor = positive_rate
balancing = real_label[:,0] + tf.pow(tf.constant(-1, dtype=tf.float32), real_label[:,0]) * tf.constant(balancing_factor, tf.float32)
modulation = tf.pow(real_label[:,0]+tf.pow(tf.constant(-1, dtype=tf.float32), real_label[:,0])*net_outs['sm_out'][:,0], tf.constant(2, dtype=tf.float32))
if LOSS_BALANCING:
	loss = balancing * loss
if FOCAL_LOSS:
	loss = modulation * loss
batch_loss = tf.reduce_mean(loss, name='batch_loss')
tf.add_to_collection('losses', batch_loss)
total_loss = tf.add_n(tf.get_collection('losses'))
correct_prediction = tf.equal(tf.argmax(net_outs['sm_out'], 1), tf.argmax(real_label, 1))
batch_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_nloss_summary = tf.summary.scalar('train naked loss', batch_loss)
train_loss_summary = tf.summary.scalar('train loss', total_loss)
val_loss_summary = tf.summary.scalar('val loss', total_loss)
train_acc_summary = tf.summary.scalar('train acc', batch_accuracy)
val_acc_summary = tf.summary.scalar('val acc', batch_accuracy)
train_merge = tf.summary.merge([train_nloss_summary, train_loss_summary, train_acc_summary])
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
epochs = [50, 0, 0]
learning_rates = [0.01, 0.001, 0.0001]
trains = []
for stage in range(len(learning_rates)):
	optimizer = tf.train.GradientDescentOptimizer(learning_rates[stage])
	train = optimizer.minimize(total_loss)
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

#preparation
if SCALE_AUGMENTATION:
	scales = [0.8, 1.0, 1.25]
else:
	scales = [1.0]
if TRANSLATION_AUGMENTATION:
	transrange = (-6, 6)
	transnum = 10
else:
	transrange = None
	transnum = 0
aug_proportion = len(scales) * transnum * (1+9*int(ROTATION_AUGMENTATION)+3*int(FLIP_AUGMENTATION))
volume_shape = np.int_([REGION_SIZE, REGION_SIZE, REGION_SIZE])
scale_translation_augment = ft.partial(da.scale_translation_augment, volume_shape=volume_shape, scales=scales, transrange=transrange)
rotation_flip_augment = ft.partial(da.rotation_flip_augment, rotation_augment=ROTATION_AUGMENTATION, flip_augment=FLIP_AUGMENTATION)
medical_normalization = ft.partial(mt.medical_normalization, max_bound=MAX_BOUND, min_bound=MIN_BOUND, pixel_mean=PIXEL_MEAN, crop=NORM_CROP, input_copy=False)

#training process
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

			ptd_num = 0
			ptd_loss = 0.0
			ptd_accuracy = 0.0
			ntd_num = 0
			ntd_loss = 0.0
			ntd_accuracy = 0.0
			random.shuffle(train_indices)
			for bi in tqdm(range(0, train_num, DATA_BATCH_SIZE)):
				#if AUGMENTATION:
				#	print("training process:%d/%d" %(pbi, positive_train_num))
				batchend = min(bi+DATA_BATCH_SIZE, train_num)
				train_data = []
				train_label = []
				for ti in range(bi, batchend):
					data_index = train_indices[ti]
					file = train_files[data_index]
					malignancy = lt.sample_malignancy_label(file, True)
					if malignancy==1:
						label = [1.0, 0]
						#data_transnum = int(2*transnum*(1-positive_rate)+0.5)
					elif malignancy==0:
						label = [0, 1.0]
						#data_transnum = int(2*transnum*positive_rate+0.5)
					else:
						label = [0.5, 0.5]
					data_transnum = transnum
					data = np.load(file)
					data_augmented = rotation_flip_augment(scale_translation_augment(data, transnum=data_transnum))
					data_labels = [label for i in range(len(data_augmented))]
					train_data.extend(data_augmented)
					train_label.extend(data_labels)
					del data
				
				randbind = np.random.permutation(len(train_data))
				#randbind = np.arange(len(train_data))
				for nbi in range(0, len(train_data), TRAIN_BATCH_SIZE):
					batch_size = min(TRAIN_BATCH_SIZE, len(train_data)-nbi)
					data_batch = np.empty((batch_size, volume_shape[0], volume_shape[1], volume_shape[2]), dtype=float)
					label_batch = np.empty((batch_size, 2), dtype=float)
					for i in range(batch_size):
						data_batch[i] = medical_normalization(train_data[randbind[nbi+i]])
						label_batch[i] = train_label[randbind[nbi+i]]
					_, losses, accuracies, train_summary = sess.run([train, loss, correct_prediction, train_merge], {volume_input: data_batch, real_label: label_batch})
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
					del data_batch
					#print("train_loss:%f train_acc:%f" % (train_loss/train_num, train_accuracy/train_num))
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
			random.shuffle(val_indices)
			for bi in tqdm(range(0, val_num, DATA_BATCH_SIZE)):
				batchend = min(bi+DATA_BATCH_SIZE, val_num)
				val_data = []
				val_label = []
				for vi in range(bi, batchend):
					data_index = val_indices[vi]
					file = val_files[data_index]
					malignancy = lt.sample_malignancy_label(file, True)
					if malignancy==1:
						label = [1.0, 0]
						data_transnum = int(2*transnum*positive_rate+0.5)
					elif malignancy==0:
						label = [0, 1.0]
						data_transnum = int(2*transnum*(1-positive_rate)+0.5)
					else:
						label = [0.5, 0.5]
					data = np.load(file)
					data_augmented = rotation_flip_augment(scale_translation_augment(data, transnum=data_transnum))
					data_labels = [label for i in range(len(data_augmented))]
					val_data.extend(data_augmented)
					val_label.extend(data_labels)
					del data
				
				randbind = np.random.permutation(len(val_data))
				#randbind = np.arange(len(val_data))
				for nbi in range(0, len(val_data), TRAIN_BATCH_SIZE):
					batch_size = min(TRAIN_BATCH_SIZE, len(val_data)-nbi)
					data_batch = np.empty((batch_size, volume_shape[0], volume_shape[1], volume_shape[2]), dtype=float)
					label_batch = np.empty((batch_size, 2), dtype=float)
					for i in range(batch_size):
						data_batch[i] = medical_normalization(val_data[randbind[nbi+i]])
						label_batch[i] = val_label[randbind[nbi+i]]
					losses, accuracies, val_summary = sess.run([loss, correct_prediction, val_merge], {volume_input: data_batch, real_label: label_batch})
					tensorboard_writer.add_summary(val_summary, epoch)
					for di in range(batch_size):
						#print("batch:%d/%d train_loss:%f train_acc:%f label:%d" % (bi+di, train_data.shape[0], losses[di], accuracies[di], label_batch[di][0]))
						if label_batch[di][0]>label_batch[di][1]:
							pvd_num += 1
							pvd_loss += losses[di]
							pvd_accuracy += accuracies[di]
						else:
							nvd_num += 1
							nvd_loss += losses[di]
							nvd_accuracy += accuracies[di]
					del data_batch
					#print("train_loss:%f train_acc:%f" % (train_loss/train_num, train_accuracy/train_num))
				del val_data

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
			training_messages.append(
				[overall_epoch + 1, ptd_loss, ptd_accuracy, ntd_loss,
				 ntd_accuracy, pvd_loss, pvd_accuracy, nvd_loss,
				 nvd_accuracy])
			# write training process to a logger file
			logger = open(net_store_path + "/" + net_store_name + ".log", 'w')
			logger.write("data batch size:{}\n" .format(DATA_BATCH_SIZE))
			logger.write("train batch size:{}\n" .format(TRAIN_BATCH_SIZE))
			logger.write("region size:{}\n" .format(REGION_SIZE))
			logger.write("max bound:{}\n" .format(MAX_BOUND))
			logger.write("min bound:{}\n" .format(MIN_BOUND))
			logger.write("pixel mean:{}\n" .format(PIXEL_MEAN))
			logger.write("normalization crop:{}\n" .format(NORM_CROP))
			logger.write("loss balancing:{}\n" .format(LOSS_BALANCING))
			logger.write("focal loss:{}\n" .format(FOCAL_LOSS))
			logger.write("scale augmentation:{}\n" .format(SCALE_AUGMENTATION))
			logger.write("translation augmentation:{}\n" .format(TRANSLATION_AUGMENTATION))
			logger.write("rotation augmentation:{}\n" .format(ROTATION_AUGMENTATION))
			logger.write("flip augmentation:{}\n" .format(FLIP_AUGMENTATION))
			logger.write("translation range:{}\n" .format(transrange))
			logger.write("translation number:{}\n" .format(transnum))
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
				print("Snapshot saved in: %s" %(snapshot_dir))
			overall_epoch += 1
	saver.save(sess, net_store_path + "/" + net_store_name)
sess.close()
print("Overall training done!")
print("The network is saved as:%s" %(net_store_name))