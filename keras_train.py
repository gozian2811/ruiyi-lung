from keras.models import *
from keras.layers import *
from keras import optimizers
from keras import callbacks
from MITools import *
from CTViewer import view_CT
import numpy as np
import dicom
import os
import glob
import math
import random


BATCH_SIZE = 108
#NUM_VIEW = 9
#MAX_BOUND = 700
#MIN_BOUND = -1000
#PIXEL_MEAN = 0.25

store_path = "models"
net_store_name = "tianchi-cnn-2D"
net_init_file = ""
data_dir = "sample_cubes_45"
net_store_path = store_path + "/" + net_store_name

if not os.access(net_store_path, os.F_OK):
	os.makedirs(net_store_path)

IMG_WIDTH, IMG_HEIGHT, NUM_VIEW, MAX_BOUND, MIN_BOUND, PIXEL_MEAN = read_environment("./constants.txt")
IMG_WIDTH = 45
IMG_HEIGHT = 45

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
train_label = np.zeros(shape=(train_num*NUM_VIEW,2), dtype=float)
train_indices = range(train_num)
#random.shuffle(train_indices)
val_num = val_num_positive + val_num_negative
val_data = np.zeros(shape=(val_num*NUM_VIEW,IMG_WIDTH,IMG_HEIGHT), dtype=float)
val_label = np.zeros(shape=(val_num*NUM_VIEW,2), dtype=float)
val_indices = range(val_num)
#random.shuffle(val_indices)

#patchs extraction
patchs = np.zeros(shape=(NUM_VIEW,IMG_WIDTH,IMG_HEIGHT), dtype = float)
for i in range(train_num):
	data = np.load(tfiles[train_indices[i]])
	label = int(train_indices[i]<train_num_positive)
	patchs = make_patchs(data)
	for j in range(NUM_VIEW):
		train_label[i*NUM_VIEW+j][1-label] = 1
		train_data[i*NUM_VIEW+j] = (patchs[j,:,:]-MIN_BOUND)/(MAX_BOUND-MIN_BOUND) - PIXEL_MEAN
for i in range(val_num):
	data = np.load(tfiles[val_indices[i]])
	label = int(val_indices[i]<val_num_positive)
	patchs = make_patchs(data)
	for j in range(NUM_VIEW):
		val_label[i*NUM_VIEW+j][1-label] = 1
		val_data[i*NUM_VIEW+j] = (patchs[j,:,:]-MIN_BOUND)/(MAX_BOUND-MIN_BOUND) - PIXEL_MEAN

np.concatenate((train_data, val_data), axis=0)
np.concatenate((train_label, val_label), axis=0)

#net arrangement
if os.path.isfile(net_init_file):
	model = load_model(net_init_file)
else:
	model = Sequential()
	model.add(Conv2D(24, kernel_size=(6,6), input_shape=(45,45,1), activation='relu', name='conv1'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='pool1'))
	model.add(Conv2D(32, kernel_size=(5,5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	model.add(Conv2D(48, kernel_size=(3,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	model.add(Flatten())
	model.add(Dense(72, activation='relu', input_dim=3*3*48))
	model.add(Dense(2, activation='relu', input_dim=72))
	model.add(Activation('softmax'))

train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], 1)
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=0.0005),
              metrics=['accuracy'])
model.fit(train_data, train_label, epochs=500, callbacks=[callbacks.CSVLogger(net_store_path+"/"+net_store_name+"-stage1.log")], batch_size=BATCH_SIZE, validation_split=0.2, shuffle=True)

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=0.0001),
              metrics=['accuracy'])
model.fit(train_data, train_label, epochs=500, callbacks=[callbacks.CSVLogger(net_store_path+"/"+net_store_name+"-stage2.log")], batch_size=BATCH_SIZE, validation_split=0.2, shuffle=True)

'''
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=0.00001),
              metrics=['accuracy'])
model.fit(train_data, train_label, epochs=200, callbacks=[callbacks.CSVLogger(net_store_path+"/"+net_store_name+"-stage3.log")], batch_size=BATCH_SIZE, validation_split=0.2, shuffle=True)
'''
'''
val_data = val_data.reshape(val_data.shape[0], val_data.shape[1], val_data.shape[2], 1)
isnodules = model.predict(val_data, batch_size=BATCH_SIZE)
'''

model.save(net_store_path + "/" + net_store_name + ".h5")
