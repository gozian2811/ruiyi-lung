import sys
import tensorflow as tf
#from tensorflow.python import debug as tfdbg
import numpy as np
from config import config
sys.path.append("/home/fyl/programs/lung_project")
from toolbox import BasicTools as bt
from toolbox import MITools as mt
from toolbox import TensorflowMultiCrop as tmc
from toolbox import LIDCTools as lt
try:
	from tqdm import tqdm  #long waits are not fun
except:
	print('tqdm not installed')
	tqdm = lambda x: x

MAX_BOUND = float(config["MAX_BOUND"])
MIN_BOUND = float(config["MIN_BOUND"])
PIXEL_MEAN = float(config["PIXEL_MEAN"])
NORM_CROP = config["NORM_CROP"]

REGION_SIZE = 64
BATCH_SIZE = 16

net_init_path = "models_tensorflow/lidc_3D_multi_crop_net_64_aug3_stage2"
net_init_file = net_init_path + "/epoch9/epoch9"
test_filelist_path = net_init_path + "/filelist_test.log"
test_files = bt.filelist_load(test_filelist_path)
test_num = len(test_files)

#net construct
volume_input = tf.placeholder(tf.float32, [None, REGION_SIZE, REGION_SIZE, REGION_SIZE])
volume_reshape = tf.reshape(volume_input, [-1, REGION_SIZE, REGION_SIZE, REGION_SIZE, 1])
real_label = tf.placeholder(tf.float32, [None, 2])
#net_outs, _ = tmc.multi_crop_net(volume_reshape, channels=[[32,32,32],[32,2]], poolings=[1,1,2], training=False)
net_outs, _ = tmc.multi_crop_net(volume_reshape, poolings=[1,1,2], training=False)
correct_prediction = tf.equal(tf.argmax(net_outs['sm_out'], 1), tf.argmax(real_label, 1))
batch_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#test process
saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
	#sess = tfdbg.LocalCLIDebugWrapperSession(sess)
	print("loading net from file:{}" .format(net_init_file))
	saver.restore(sess, net_init_file)
	#sess.run(tf.global_variables_initializer())
	
	total_accuracy = 0
	predictions = np.zeros((test_num, 2), dtype=float)
	for bi in tqdm(range(0, test_num, BATCH_SIZE)):
		batchend = min(bi+BATCH_SIZE, test_num)
		batch_size = batchend - bi
		train_data = np.zeros((batch_size, REGION_SIZE, REGION_SIZE, REGION_SIZE), dtype=float)
		train_label = np.zeros((batch_size, 2), dtype=float)
		for ti in range(bi, batchend):
			file = test_files[ti]
			file = "../data_samples/" + file[:23] + "_ipris" + file[23:]
			malignancy = lt.sample_malignancy_label(file, True)
			if malignancy>=0:
				train_label[ti-bi][1-malignancy] = 1
			else:
				print('unknown malignancy, skip')
				continue
			data = np.load(file)
			data_cropped = mt.crop(data, (REGION_SIZE, REGION_SIZE, REGION_SIZE))
			train_data[ti-bi] = mt.medical_normalization(data_cropped, MAX_BOUND, MIN_BOUND, PIXEL_MEAN, NORM_CROP, input_copy=False)
		accuracy, prediction = sess.run([batch_accuracy, net_outs['sm_out']], {volume_input:train_data, real_label:train_label})
		total_accuracy += accuracy * batch_size
		predictions[bi:batchend] = prediction
	total_accuracy /= float(test_num)
	print("accuracy:{}" .format(total_accuracy))
	#print("prediction:\n{}" .format(predictions))