import tensorflow as tf
import numpy as np
import os
import shutil
import glob
import math
import random
from toolbox import MITools as mt
from toolbox import TensorflowTools as tft

constants = mt.read_constants("./constants2.txt")
REGION_SIZE = constants["REGION_SIZE"]
MAX_BOUND = float(constants["MAX_BOUND"])
MIN_BOUND = float(constants["MIN_BOUND"])
PIXEL_MEAN = float(constants["PIXEL_MEAN"])

#NUM_EPOCH = 200
SNAPSHOT_EPOCH = 10
#DECAY_EPOCH = 0
#INITIAL_LEARNING_RATE = 0.001
#DECAY_LEARNING_RATE = 1.0

BATCH_SIZE = 30
#VALIDATION_RATE = 0.2

AUGMENTATION = True
ALLNEGATIVE = True

store_path = "models_tensorflow"
net_store_name = "luna_slh_3D_l3454-512-2_bn_aug_test"
#net_init_file = "models_tensorflow/luna_3D_l3454-512-2_bn3_init/luna_3D_l3454-512-2_bn3"
data_dir1 = "luna_cubes_56_overbound"
data_dir2 = "slh_cubes_56_overbound"
net_store_path = store_path + "/" + net_store_name
tensorboard_path = net_store_path+"/tensorboard/"

if not os.access(net_store_path, os.F_OK):
	os.makedirs(net_store_path)
if os.access(tensorboard_path, os.F_OK):
	shutil.rmtree(tensorboard_path)

#data arrangement
train_sets = ["subset0", "subset1", "subset2", "subset3", "subset4", "subset5", "subset6", "subset7", "subset8"]
val_sets = ["subset9"]
tpfiles = []
vpfiles = []
nfiles = []
for set in train_sets:
	train_dir = os.path.join(data_dir1, set)
	pdir = os.path.join(train_dir,"npy","*.npy")
	tpfiles.extend(glob.glob(pdir))
	ndir = os.path.join(train_dir,"npy_non","*.npy")
	nfiles.extend(glob.glob(ndir))
if "data_dir2" in dir():
	tpfiles.extend(glob.glob(os.path.join(data_dir2,"npy","*.npy")))
for set in val_sets:
	val_dir = os.path.join(data_dir1, set)
	pdir = os.path.join(val_dir, "npy", "*.npy")
	vpfiles.extend(glob.glob(pdir))
	ndir = os.path.join(val_dir, "npy_non", "*.npy")
	nfiles.extend(glob.glob(ndir))
positive_train_num = len(tpfiles)
positive_val_num = len(vpfiles)
num_negative = len(nfiles)
#num_positive = 5
#num_negative = 20
random.shuffle(tpfiles)
#random.shuffle(nfiles)
#num_negative = 1000 * num_positive

if positive_train_num==0:
	print("no positive training file found")
	exit()
positive_train_indices = [i for i in range(positive_train_num)]
positive_val_indices = [i for i in range(positive_val_num)]
#random.shuffle(positive_indices)
negative_importances = 1000*np.ones(shape=[num_negative], dtype=float)

#net construct
volume_input = tf.placeholder(tf.float32, [None, REGION_SIZE, REGION_SIZE, REGION_SIZE])
volume_reshape = tf.reshape(volume_input, [-1, REGION_SIZE, REGION_SIZE, REGION_SIZE, 1])
real_label = tf.placeholder(tf.float32, [None, 2])
#r_bn1, b_bn1, w_conv1, w_conv2, out_conv1, out_bn1, hidden_conv1, hidden_conv2, hidden_conv3, out_fc1, out_fc2, softmax_out = tft.volume_bnnet2_l6_56(volume_reshape)
out_fc2, softmax_out = tft.volume_bnnet2_l6_56(volume_reshape)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = out_fc2, labels = real_label)
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
epochs = [20, 0, 0]
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
	if 'net_init_file' not in dir() or net_init_file is None:
		sess.run(tf.global_variables_initializer())
	else:
		saver.restore(sess, net_init_file)
	train_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
	best_accuracy = 0.0
	best_epoch = 0
	overall_epoch = 0
	np_proportion = num_negative / (positive_train_num + positive_val_num)
	training_messages = []
	for train_stage in range(len(trains)):
		train = trains[train_stage]
		for epoch in range(epochs[train_stage]):
			#overall_epoch = train_stage*DECAY_EPOCH+epoch
			print('epoch:%d' %(overall_epoch+1))
			random.shuffle(positive_train_indices)
			#random.shuffle(positive_val_indices)
			negative_indices = [i for i in range(num_negative)]
			if negative_importances.nonzero()[0].size>0:
				negative_probabilities = negative_importances / negative_importances.sum()
			else:
				negative_probabilities[:] = negative_importances

			# the number of negative samples is in default larger than the number of positive samples
			train_num = 0
			train_loss = 0.0
			train_accuracy = 0.0
			positive_batch_size = (1-int(AUGMENTATION))*(int(BATCH_SIZE/2)-1) + 1	#if augmentatiion implemented then set batch size to 1 or half of BATCH_SIZE
			for pbi in range(0, positive_train_num, positive_batch_size):
				if AUGMENTATION:
					print("training process:%d/%d %s" %(pbi, positive_train_num, tpfiles[positive_train_indices[pbi]]))
				for pti in range(pbi, min(pbi+positive_batch_size, positive_train_num)):
					data_index = positive_train_indices[pti]
					pfile = tpfiles[data_index]
					if pfile.split('/')[0].find("luna")>=0:
						patient_uid, nodule_diameter = mt.get_annotation_informations(pfile, "luna_cubes_56_overbound/luna_annotations.csv")
					elif pfile.split('/')[0].find("tianchi")>=0:
						patient_uid, nodule_diameter = mt.get_annotation_informations(pfile, "tianchi_cubes_56_overbound/tianchi_annotations.csv")
					else:
						patient_uid = mt.get_volume_informations(pfile)[0]
						nodule_diameter = 0
					positive_data = np.load(pfile)
					if "positive_batch" not in dir():
						positive_batch = mt.extract_volumes(positive_data, nodule_diameter=nodule_diameter, scale_augment=AUGMENTATION, translation_augment=AUGMENTATION, rotation_augment=AUGMENTATION)
					else:
						positive_batch = np.concatenate((positive_batch, mt.extract_volumes(positive_data, nodule_diameter=nodule_diameter, scale_augment=AUGMENTATION, translation_augment=AUGMENTATION, rotation_augment=AUGMENTATION)), axis=0)
				#negative_batch_size = min(positive_batch.shape[0], negative_probabilities.nonzero()[0].size)
				negative_batch_size = min(int(math.ceil(positive_batch_size*np_proportion)), negative_probabilities.nonzero()[0].size)
				if negative_batch_size > 0:
					negative_batch = np.zeros(shape=[negative_batch_size, positive_batch.shape[1], positive_batch.shape[2], positive_batch.shape[3]], dtype=positive_batch.dtype)
					negative_candidate = np.random.choice(negative_indices, size=negative_batch_size, replace=False, p=negative_probabilities)
					for ni in range(negative_candidate.size):
						negative_batch[ni] = np.load(nfiles[negative_candidate[ni]])
						negative_probabilities[negative_candidate[ni]] = 0
					if negative_probabilities.sum() > 0:
						negative_probabilities /= negative_probabilities.sum()
			
				train_data = np.zeros(shape=(positive_batch.shape[0]+negative_batch_size, positive_batch.shape[1], positive_batch.shape[2], positive_batch.shape[3]), dtype=float)
				train_label = np.zeros(shape=(positive_batch.shape[0]+negative_batch_size, 2), dtype=float)
				batch_random = np.random.permutation(positive_batch.shape[0]+negative_batch_size)
				for bi in range(batch_random.size):
					batch_index = batch_random[bi]
					if batch_index<positive_batch.shape[0]:
						train_data[bi] = (positive_batch[batch_index] - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
						train_label[bi][0] = 1
					else:
						train_data[bi] = (negative_batch[batch_index-positive_batch.shape[0]] - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
						train_label[bi][1] = 1
				#train_data = (train_data - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
				train_num += train_data.shape[0]
				for bi in range(0, train_data.shape[0], BATCH_SIZE):
					batch_size = min(BATCH_SIZE, train_data.shape[0]-bi)
					data_batch = train_data[bi:bi+batch_size]
					label_batch = train_label[bi:bi+batch_size]
					#rb1, bb1, wc1, wc2, oc1, ob1, hc1, hc2, hc3, of1, of2, so = sess.run([r_bn1, b_bn1, w_conv1, w_conv2, out_conv1, out_bn1, hidden_conv1, hidden_conv2, hidden_conv3, out_fc1, out_fc2, softmax_out], {volume_input: data_batch, real_label: label_batch})
					_, predictions, losses, accuracies, summary = sess.run([train, out_fc2, focal_loss, correct_prediction, merged], {volume_input: data_batch, real_label: label_batch})
					for di in range(data_batch.shape[0]):
						#print("batch:%d/%d train_loss:%f train_acc:%f label:%d" % (bi+di, train_data.shape[0], losses[di], accuracies[di], label_batch[di][0]))
						train_loss += losses[di]
						train_accuracy += accuracies[di]
						negative_batch_index = batch_random[bi+di] - positive_batch.shape[0]
						if negative_batch_index>=0:
							negative_index = negative_candidate[negative_batch_index]
							negative_importances[negative_index] = losses[di]
					#print("train_loss:%f train_acc:%f" % (train_loss/train_num, train_accuracy/train_num))
				if 'negative_batch' in dir():
					del negative_batch
				del positive_batch
				del train_data

			if train_num > 0:
				train_loss /= train_num
				train_accuracy /= train_num
			#train_writer.add_summary(summary, epoch)

			val_num = 0
			val_loss = 0.0
			val_accuracy = 0.0
			#positive_batch_size = int(BATCH_SIZE/2)
			#positive_batch_size = (1-int(AUGMENTATION))*(int(BATCH_SIZE/2)-1) + 1	#if augmentatiion implemented then set batch size to 1 or half of BATCH_SIZE
			#val_portion_size = int(num_negative/positive_val_num)	
			#val_portion_size = BATCH_SIZE - 1
			for pbi in range(0, positive_val_num, positive_batch_size):
				if AUGMENTATION:
					print("validation process:%d/%d %s" %(pbi, positive_val_num, vpfiles[positive_val_indices[pbi]]))
				#if negative_probabilities.sum()<=0:
				#	break
				#negative_probabilities /= negative_probabilities.sum()
				for pvi in range(pbi, min(pbi+positive_batch_size, positive_val_num)):
					data_index = positive_val_indices[pvi]
					pfile = vpfiles[data_index]
					if pfile.split('/')[0].find("luna")>=0:
						patient_uid, nodule_diameter = mt.get_annotation_informations(pfile, "luna_cubes_56_overbound/luna_annotations.csv")
					elif pfile.split('/')[0].find("tianchi")>=0:
						patient_uid, nodule_diameter = mt.get_annotation_informations(pfile, "tianchi_cubes_56_overbound/tianchi_annotations.csv")
					else:
						patient_uid = mt.get_volume_informations(pfile)[0]
						nodule_diameter = 0
					positive_data = np.load(pfile)
					if "positive_batch" not in dir():
						positive_batch = mt.extract_volumes(positive_data, nodule_diameter=nodule_diameter, scale_augment=AUGMENTATION, translation_augment=AUGMENTATION, rotation_augment=AUGMENTATION)
					else:
						positive_batch = np.concatenate((positive_batch, mt.extract_volumes(positive_data, nodule_diameter=nodule_diameter, scale_augment=AUGMENTATION, translation_augment=AUGMENTATION, rotation_augment=AUGMENTATION)), axis=0)
				#print("validation process:%d/%d" % (pvi+1, positive_val_num))
				#negative_batch_size = min(positive_batch.shape[0], negative_probabilities.nonzero()[0].size)
				negative_batch_size = min(int(math.ceil(positive_batch_size*np_proportion)), negative_probabilities.nonzero()[0].size)
				if negative_batch_size>0:
					negative_batch = np.zeros(shape=[negative_batch_size, REGION_SIZE, REGION_SIZE, REGION_SIZE], dtype=float)
					negative_candidate = np.random.choice(negative_indices, size=negative_batch.shape[0], replace=False, p=negative_probabilities)
					for ni in range(negative_candidate.size):
						negative_batch[ni] = np.load(nfiles[negative_candidate[ni]])
						negative_probabilities[negative_candidate[ni]] = 0
					if negative_probabilities.sum() > 0:
						negative_probabilities /= negative_probabilities.sum()
				fusion_batch_size = positive_batch.shape[0] + negative_batch_size
				val_data = np.zeros(shape=(fusion_batch_size, positive_batch.shape[1], positive_batch.shape[2], positive_batch.shape[3]), dtype=float)
				val_label = np.zeros(shape=(fusion_batch_size, 2), dtype=float)
				batch_random = np.random.permutation(fusion_batch_size)
				for bi in range(batch_random.size):
					batch_index = batch_random[bi]
					if batch_index < positive_batch.shape[0]:
						val_data[bi] = (positive_batch[batch_index] - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
						val_label[bi][0] = 1
					else:
						val_data[bi] = (negative_batch[batch_index - positive_batch.shape[0]] - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
						val_label[bi][1] = 1
				#val_data = (val_data - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
				val_num += val_data.shape[0]
				for bi in range(0, val_data.shape[0], BATCH_SIZE):
					batch_size = min(BATCH_SIZE, val_data.shape[0]-bi)
					data_batch = val_data[bi:bi+batch_size]
					label_batch = val_label[bi:bi+batch_size]
					#hc1, hc2, hc3, of1, df1, wf2, bf2, of2, so = sess.run([hidden_conv1, hidden_conv2, hidden_conv3, out_fc1, dropout_fc1, w_fc2, b_fc2, out_fc2, softmax_out], {volume_input: data_batch, real_label: label_batch})
					losses, accuracies, summary = sess.run([focal_loss, correct_prediction, merged], {volume_input: data_batch, real_label: label_batch})
					for di in range(data_batch.shape[0]):
						#print("batch:%d/%d val_loss:%f val_acc:%f label:%d" % (bi+di, val_data.shape[0], losses[di], accuracies[di], label_batch[di][0]))
						val_loss += losses[di]
						val_accuracy += accuracies[di]
						ncind = batch_random[bi+di] - positive_batch.shape[0]
						if ncind>=0:
							negative_index = negative_candidate[ncind]
							negative_importances[negative_index] = losses[di]
					#print("val_loss:%f val_acc:%f" % (val_loss/val_num, val_accuracy/val_num))
				if 'negative_batch' in dir():
					del negative_batch
				del positive_batch
				del val_data
			
			if (ALLNEGATIVE and overall_epoch%SNAPSHOT_EPOCH == 0) or negative_importances.nonzero()[0].size<=0:
			#if negative_importances.nonzero()[0].size<=0:
			#if ALLNEGATIVE:
				#validate the remaining negative data
				#res_val_num = 0
				#res_val_loss = 0.0
				#res_val_accuracy = 0.0
				if negative_importances.nonzero()[0].size<=0:
					negative_probabilities += 1

				negative_batch = np.zeros(shape=[BATCH_SIZE, REGION_SIZE, REGION_SIZE, REGION_SIZE], dtype=float)
				negative_labels = np.concatenate((np.zeros(shape=(BATCH_SIZE,1),dtype=float), np.ones(shape=(BATCH_SIZE,1),dtype=float)), axis=1)
				negative_batch_indices = np.zeros(shape=(BATCH_SIZE), dtype=int)
				batch_index = 0
				for nvi in range(num_negative):
					if negative_probabilities[nvi] > 0:
						negative_batch_indices[batch_index] = nvi
						negative_batch[batch_index] = np.load(nfiles[nvi])
						batch_index += 1
						if batch_index == BATCH_SIZE:
							print("negative rest validation process:%d/%d" % (nvi + 1, num_negative))
							val_num += batch_index
							losses, accuracies, summary = sess.run([cross_entropy, correct_prediction, merged], {volume_input: negative_batch, real_label: negative_labels})
							for nbi in range(negative_batch.shape[0]):
								val_loss += losses[nbi]
								val_accuracy += accuracies[nbi]
								negative_importances[negative_batch_indices[nbi]] = losses[nbi]
							batch_index = 0
				if batch_index > 0:
					print("negative rest validation process:%d/%d" % (nvi + 1, num_negative))
					negative_batch = np.delete(negative_batch, [i for i in range(batch_index, negative_batch.shape[0])], axis=0)
					negative_labels = np.delete(negative_labels, [i for i in range(batch_index, negative_labels.shape[0])], axis=0)
					negative_batch_indices = np.delete(negative_batch_indices, [i for i in range(batch_index, negative_batch_indices.shape[0])], axis=0)
					val_num += batch_index
					losses, accuracies, summary = sess.run([cross_entropy, correct_prediction, merged], {volume_input: negative_batch, real_label: negative_labels})
					for nbi in range(negative_batch.shape[0]):
						val_loss += losses[nbi]
						val_accuracy += accuracies[nbi]
						negative_importances[negative_batch_indices[nbi]] = losses[nbi]
				#res_val_loss /= res_val_num
				#res_val_accuracy /= res_val_num
				#print("negative_loss:%f negative_acc:%f" % (res_val_loss, res_val_accuracy))

			if val_num > 0:
				val_loss /= val_num
				val_accuracy /= val_num
			else:
				val_loss = 0
				val_accuracy = 1
			train_writer.add_summary(summary, epoch)
			print("train_loss:%f train_acc:%f val_loss:%f val_acc:%f" % (train_loss, train_accuracy, val_loss, val_accuracy))
			if val_accuracy>=best_accuracy:
				best_accuracy = val_accuracy
				best_epoch = overall_epoch
				best_dir = net_store_path+"/"+net_store_name+"_best"
				if os.access(best_dir, os.F_OK):
					shutil.rmtree(best_dir)
				os.mkdir(best_dir)
				saver.save(sess, best_dir+"/"+net_store_name+"_best")
				#saver.save(sess, net_store_path+"/"+net_store_name+"_best")
			training_messages.append([overall_epoch+1, train_loss, train_accuracy, val_loss, val_accuracy])
			#write training process to a logger file
			logger = open(net_store_path + "/" + net_store_name + ".log", 'w')
			logger.write("random validation\n")
			#logger.write("initial_learning_rate:%f decay_rate:%f decay_epoch:%d\n" %(INITIAL_LEARNING_RATE, DECAY_LEARNING_RATE, DECAY_EPOCH))
			logger.write("learning_rates:%f %f %f\n" %(learning_rates[0], learning_rates[1], learning_rates[2]))
			logger.write("epochs:%d %d %d\n" %(epochs[0], epochs[1], epochs[2]))
			logger.write("epoch train_loss train_acc val_loss val_acc\n")
			for tm in range(len(training_messages)):
				logger.write("%d %f %f %f %f\n" %(training_messages[tm][0], training_messages[tm][1], training_messages[tm][2], training_messages[tm][3], training_messages[tm][4]))
			logger.write("best epoch:%d" %(best_epoch+1))
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
negative_losses_output = open((net_store_path + "/negative_losses.txt", "w")
for ni in range(len(negative_importances)):
	negative_losses_output.write("%d:%f\n" %(ni, negative_importances[ni]))
negative_losses_output.close()
print("Overall training done!")
print("The network is saved as:%s" %(net_store_name))
