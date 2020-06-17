from toolbox import MITools as mt
import tensorflow as tf
import numpy as np
import dicom
import os
import glob
import math
import random


NUM_EPOCH = 5
BATCH_SIZE = 54
IMG_WIDTH = 56
IMG_HEIGHT = 56
NUM_VIEW = 9
MAX_BOUND = 700
MIN_BOUND = -1024
PIXEL_MEAN = 0.25

net_store_path = "models/tianchi"
net_store_name = "TIANCHI-CNN3.ckpt"
data_dir = "sample_cubes_56"

#net arrangement
with tf.device('/gpu:0'):
	x = tf.placeholder(tf.float32, [None,IMG_WIDTH,IMG_HEIGHT])
	y_real = tf.placeholder(tf.float32, [None,2])
	x_image = tf.reshape(x, [-1,IMG_WIDTH,IMG_HEIGHT,1])

	w1 = tf.Variable(tf.truncated_normal([5,5,1,24]))
	b1 = tf.Variable(tf.truncated_normal([24]))
	w2 = tf.Variable(tf.truncated_normal([3,3,24,32]))
	b2 = tf.Variable(tf.truncated_normal([32]))
	w3 = tf.Variable(tf.truncated_normal([12*12*32,16]))
	b3 = tf.Variable(tf.truncated_normal([16]))
	w4 = tf.Variable(tf.truncated_normal([16,2]))
	b4 = tf.Variable(tf.truncated_normal([2]))

	y1 = tf.nn.relu(tf.nn.conv2d(x_image, w1, strides=[1,1,1,1], padding='VALID') + b1)
	y1_p = tf.nn.max_pool(y1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
	y2 = tf.nn.relu(tf.nn.conv2d(y1_p, w2, strides=[1,1,1,1], padding='VALID') + b2)
	y2_p = tf.nn.max_pool(y2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
	y2_r = tf.reshape(y2_p, shape=[-1,12*12*32])
	y3 = tf.nn.relu(tf.matmul(y2_r, w3) + b3)
	y4 = tf.nn.relu(tf.matmul(y3, w4) + b4)
	y = tf.nn.softmax(y4)

	#loss = tf.reduce_sum(tf.square(y_real-y))
	loss = tf.losses.softmax_cross_entropy(y_real,y)
	optimizer1 = tf.train.GradientDescentOptimizer(0.0005)
	train1 = optimizer1.minimize(loss)
	optimizer2 = tf.train.GradientDescentOptimizer(0.0001)
	train2 = optimizer2.minimize(loss)
	optimizer3 = tf.train.GradientDescentOptimizer(0.00001)
	train3 = optimizer3.minimize(loss)
	trains = [train1]
	is_correct = tf.equal(tf.argmax(y,1), tf.argmax(y_real,1))
	accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


#data arrangement
train_dir = os.path.join(data_dir,"train")
val_dir = os.path.join(data_dir,"val")

tpdir = os.path.join(train_dir,"npy","*.npy")
tpfiles = glob.glob(tpdir)
train_num_positive = len(tpfiles)
tndir = os.path.join(train_dir,"npy_random","*.npy")
tnfiles = glob.glob(tndir)
train_num_negative = len(tnfiles)
tfiles = tpfiles[:]
tfiles.extend(tnfiles)
vpdir = os.path.join(val_dir,"npy","*.npy")
vpfiles = glob.glob(vpdir)
val_num_positive = len(vpfiles)
vndir = os.path.join(val_dir,"npy_random","*.npy")
vnfiles = glob.glob(vndir)
val_num_negative = len(vnfiles)
vfiles = vpfiles[:]
vfiles.extend(vnfiles)

train_num = train_num_positive + train_num_negative
train_data = np.zeros(shape=(train_num*NUM_VIEW,IMG_WIDTH,IMG_HEIGHT), dtype=float)
train_label = np.zeros(shape=(train_num*NUM_VIEW,2), dtype=bool)
train_indices = range(train_num)
random.shuffle(train_indices)
val_num = val_num_positive + val_num_negative
val_data = np.zeros(shape=(val_num*NUM_VIEW,IMG_WIDTH,IMG_HEIGHT), dtype=float)
val_label = np.zeros(shape=(val_num*NUM_VIEW,2), dtype=bool)
val_indices = range(val_num)
random.shuffle(val_indices)

#patchs extraction
patchs = np.zeros(shape=(NUM_VIEW,IMG_WIDTH,IMG_HEIGHT), dtype = float)
for i in range(train_num):
	label = int(train_indices[i]<train_num_positive)
	data = np.load(tfiles[train_indices[i]])
	patchs = mt.make_patchs(data)
	for j in range(NUM_VIEW):
		train_label[i*NUM_VIEW+j][1-label] = 1
		train_data[i*NUM_VIEW+j] = (patchs[j]-MIN_BOUND)/(MAX_BOUND-MIN_BOUND) - PIXEL_MEAN
for i in range(val_num):
	label = int(val_indices[i]<val_num_positive)
	data = np.load(tfiles[val_indices[i]])
	patchs = mt.make_patchs(data)
	for j in range(NUM_VIEW):
		val_label[i*NUM_VIEW+j][1-label] = 1
		val_data[i*NUM_VIEW+j] = (patchs[j]-MIN_BOUND)/(MAX_BOUND-MIN_BOUND) - PIXEL_MEAN

print("number of training:%d" %(train_num))
print("number of validation:%d" %(val_num))
print("number of views:%d" %(NUM_VIEW))
print("number of total epochs:%d" %(NUM_EPOCH))

num_train_batchs = int(math.ceil(float(train_num*NUM_VIEW) / BATCH_SIZE))
num_val_batchs = int(math.ceil(float(val_num*NUM_VIEW) / BATCH_SIZE))

saver = tf.train.Saver()
init = tf.global_variables_initializer()
'''
sess = tf.train.MonitoredTrainingSession(
	master='n-cnn_monitor',
	is_chief=True,
	checkpoint_dir='./monitors',
	scaffold=None,
	hooks=None,
	chief_only_hooks=None,
	save_checkpoint_secs=10,
	save_summaries_steps=10,
	config=None,
)
'''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(init)
for train in trains:
	for epoch in range(NUM_EPOCH):
		print("stage:%d epoch:%d." %(trains.index(train)+1, epoch+1))
		train_loss = 0.0
		train_accuracy = 0.0
		for batch in range(num_train_batchs):
			batch_size = min(BATCH_SIZE, train_num*NUM_VIEW-batch*BATCH_SIZE)
			#data_batch = np.zeros(shape=(batch_size,IMG_WIDTH,IMG_HEIGHT), dtype=float)
			#label_batch = np.zeros(shape=(batch_size,2), dtype=float)
			label_batch = train_label[(batch*BATCH_SIZE):(batch*BATCH_SIZE+batch_size)]
			data_batch = train_data[(batch*BATCH_SIZE):(batch*BATCH_SIZE+batch_size)]
			l, acc, t = sess.run([loss, accuracy, train], {x:data_batch, y_real:label_batch})
			train_loss += l
			train_accuracy += acc
		train_loss /= num_train_batchs
		train_accuracy /= num_train_batchs
		print("training loss:%f." %(train_loss))
		print("training accuracy:%f." %(train_accuracy))
		val_loss = 0.0
		val_accuracy = 0.0
		for batch in range(num_val_batchs):
			batch_size = min(BATCH_SIZE, val_num*NUM_VIEW-batch*BATCH_SIZE)
			#data_batch = np.zeros(shape=(batch_size,IMG_WIDTH,IMG_HEIGHT), dtype=float)
			#label_batch = np.zeros(shape=(batch_size,2), dtype=float)
			label_batch = val_label[(batch*BATCH_SIZE):(batch*BATCH_SIZE+batch_size)]
			data_batch = val_data[(batch*BATCH_SIZE):(batch*BATCH_SIZE+batch_size)]
			l, acc = sess.run([loss, accuracy], {x:data_batch, y_real:label_batch})
			val_loss += l
			val_accuracy += acc
		val_loss /= num_val_batchs
		val_accuracy /= num_val_batchs
		print("validation loss:%f." %(val_loss))
		print("validation accuracy:%f." %(val_accuracy))

#if not os.access(net_store_path, os.F_OK):
#	os.makedirs(net_store_path)
#saver.save(sess, net_store_path + "/" + net_store_name)

sess.close()
