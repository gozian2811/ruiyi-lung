import tensorflow as tf
import numpy as np
import copy
import math
import random

def multi_crop_pooling(input, steps=2):
	inputshape = np.array([input.shape[1].value, input.shape[2].value, input.shape[3].value])
	cropshape = inputshape / pow(2, steps)
	pool = input
	featurelist = []
	for s in range(steps):
		poolshape = np.array([pool.shape[1].value, pool.shape[2].value, pool.shape[3].value])
		poolcrop = pool[:, int((poolshape[0] - cropshape[0]) / 2):int((poolshape[0] + cropshape[0]) / 2),
			    int((poolshape[1] - cropshape[1]) / 2):int((poolshape[1] + cropshape[1]) / 2),
			    int((poolshape[2] - cropshape[2]) / 2):int((poolshape[2] + cropshape[2]) / 2), :]
		featurelist.append(poolcrop)
		pool = tf.nn.max_pool3d(pool, strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')
	featurelist.append(pool)
	'''
	inputcrop = input[:, int((inputshape[0] - cropshape[0]) / 2):int((inputshape[0] + cropshape[0]) / 2),
		    int((inputshape[1] - cropshape[1]) / 2):int((inputshape[1] + cropshape[1]) / 2),
		    int((inputshape[2] - cropshape[2]) / 2):int((inputshape[2] + cropshape[2]) / 2), :]
	featurelist = [inputcrop]
	for s in range(steps):
		pool = tf.nn.max_pool3d(input, strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')
		poolshape = np.array([pool.shape[1].value, pool.shape[2].value, pool.shape[3].value])
		poolcrop = pool[:, int((poolshape[0] - cropshape[0]) / 2):int((poolshape[0] + cropshape[0]) / 2),
			    int((poolshape[1] - cropshape[1]) / 2):int((poolshape[1] + cropshape[1]) / 2),
			    int((poolshape[2] - cropshape[2]) / 2):int((poolshape[2] + cropshape[2]) / 2), :]
		featurelist.append(poolcrop)
	'''

	concfeature = tf.keras.backend.concatenate(featurelist, axis=4)

	return concfeature

def multi_crop_net(input, channels=[[16,16,16],[32,2]], poolings=[1,1,2], rrelu=(3,8), positive_confidence=0.5, weight_decay_coefficiency=None, namescope='multi-crop-net', training=True):
	channels_layerwise = copy.deepcopy(channels)
	poolings_conv = copy.deepcopy(poolings)
	feature = input
	variables = {}
	outputs = {}
	with tf.name_scope(namescope):
		channels_layerwise[0].insert(0, 1)
		poolings_conv.insert(0, 0)
		for l in range(1, len(channels_layerwise[0])):
			w_conv = tf.Variable(tf.random_normal([3, 3, 3, channels_layerwise[0][l-1]*(poolings_conv[l-1]+1), channels_layerwise[0][l]], stddev=0.1), dtype=tf.float32, name='w_conv'+str(l), trainable=training)
			b_conv = tf.Variable(tf.random_normal([channels_layerwise[0][l]], stddev=0.1), dtype=tf.float32, name='b_conv'+str(l), trainable=training)
			if training:
				rr_conv = tf.Variable(tf.random_uniform([1], rrelu[0], rrelu[1], seed=random.random()), name='rr_conv' + str(l), trainable=training)	#whether it's learnable is to be further discussed
			else:
				rr_conv = tf.Variable(tf.random_uniform([1], rrelu[0], rrelu[1], seed=random.random()), trainable=training)
			if weight_decay_coefficiency is not None:
				wd_loss = tf.multiply(weight_decay_coefficiency, tf.nn.l2_loss(w_conv), name='wd_conv'+str(l))
				tf.add_to_collection('losses', wd_loss)
			variables[namescope+'/w_conv'+str(l)] = w_conv
			variables[namescope+'/b_conv'+str(l)] = b_conv
			variables[namescope+'/rr_conv' + str(l)] = rr_conv
			conv_feature = tf.add(tf.nn.conv3d(feature, w_conv, strides=[1, 1, 1, 1, 1], padding='SAME'), b_conv)
			hidden_feature = tf.nn.leaky_relu(conv_feature, 1/rr_conv)
			feature = multi_crop_pooling(hidden_feature, poolings_conv[l])
			outputs['conv'+str(l)+'_out'] = feature
		feature = tf.layers.flatten(feature)
		featurelength = feature.shape[1].value
		channels_layerwise[1].insert(0, featurelength)
		for l in range(1, len(channels_layerwise[1])):
			w_fc = tf.Variable(tf.random_normal([channels_layerwise[1][l-1], channels_layerwise[1][l]], stddev=0.1), name='w_fc'+str(l), trainable=training)
			b_fc = tf.Variable(tf.zeros([channels_layerwise[1][l]]), dtype=tf.float32, name='b_fc'+str(l), trainable=training)
			if training:
				rr_fc = tf.Variable(tf.random_uniform([1], rrelu[0], rrelu[1], seed=random.random()), name='rr_fc'+str(l), trainable=training)
			else:
				rr_fc = tf.Variable(tf.random_uniform([1], rrelu[0], rrelu[1], seed=random.random()), trainable=training)
			if weight_decay_coefficiency is not None:
				wd_loss = tf.multiply(weight_decay_coefficiency, tf.nn.l2_loss(w_fc), name='wd_fc'+str(l))
				tf.add_to_collection('losses', wd_loss)
			variables[namescope+'/w_fc'+str(l)] = w_fc
			variables[namescope+'/b_fc'+str(l)] = b_fc
			variables[namescope+'/rr_fc'+str(l)] = rr_fc
			fc_feature = tf.add(tf.matmul(feature, w_fc), b_fc)
			if l < len(channels_layerwise[1])-1:
				feature = tf.nn.leaky_relu(fc_feature, 1/rr_fc)
				outputs['fc'+str(l)+'_out'] = feature
		softmax_out = tf.nn.softmax(fc_feature)
		outputs['last_out'] = fc_feature
		outputs['sm_out'] = softmax_out
	return outputs, variables