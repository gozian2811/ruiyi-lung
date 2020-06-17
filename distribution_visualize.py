import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tensorboardX import SummaryWriter
from toolbox.BasicTools import filelist_load, foldlist

experiments_path = "F:/lung_project2/experiments/experiments_cl/"
rootpath = experiments_path + "MultiTaskNet_classification_fold1_epoch500_train/"
lidcpath = rootpath + "MultiTaskNet_classification_fold1_epoch500_feature"
sphpath = rootpath + "MultiTaskNet_classification_fold1_epoch500_feature2"
#rootpath = "F:/lung_project2/experiments/experiments_cl/"
#lidcpath = rootpath + "BasicNet_lidc_classification_9_epoch335_train/BasicNet_lidc_classification_9_epoch335_feature"
#sphpath = rootpath + "BasicNet_lidc_classification_9_epoch335_train_sph/BasicNet_lidc_classification_9_epoch335_feature"
lidcfilelist = filelist_load(experiments_path + "filelist_train_fold[0, 1, 2, 3, 4].log")
sphfilelist = filelist_load(experiments_path + "filelist2_train_fold[0, 1, 2, 3, 4].log")

lidc_filelist_revised = []
sph_filelist_revised = []

for lidcfile in lidcfilelist:
	filename = os.path.splitext(os.path.basename(lidcfile))[0]
	lidc_filelist_revised.append(filename)
for sphfile in sphfilelist:
	filename = os.path.splitext(os.path.basename(sphfile))[0]
	sph_filelist_revised.append(filename)

lidcfolds, _ = foldlist(lidc_filelist_revised, 5, {'1':1, '2':2, '3':3, '4':4, '5':5})
sphfolds, _ = foldlist(sph_filelist_revised, 5, {'1':1, '2':2, '3':3, '4':4, '5':5})

lidc_feature_filelist = glob(lidcpath + '/*.npy')
sph_feature_filelist = glob(sphpath + '/*.npy')

lidc_features = []
lidc_labels = []
#label_visual = {'1':1, '2':1, '3':1, '4':4, '5':1}
for lidc_feature_file in lidc_feature_filelist:
	filename = os.path.basename(lidc_feature_file[:-6])
	lidc_features.append(np.load(lidc_feature_file))
	for key in lidcfolds.keys():
		if filename in lidcfolds[key]:
			lidc_labels.append(key)
			break
sph_features = []
sph_labels = []
sph_labels_folds = {'1':[], '2':[], '3':[], '4':[], '5':[]}
for sph_feature_file in sph_feature_filelist:
	filename = os.path.basename(sph_feature_file[:-6])
	sph_features.append(np.load(sph_feature_file))
	for key in sphfolds.keys():
		if filename in sphfolds[key]:
			sph_labels.append(key)
			for fkey in sph_labels_folds.keys():
				if fkey==key:
					sph_labels_folds[fkey].append(fkey)
				else:
					sph_labels_folds[fkey].append(0)
			break

features_fuse = []
labels_fuse = []
features_fuse.extend(lidc_features)
labels_fuse.extend(['LIDC-IDRI' for i in range(len(lidc_labels))])
features_fuse.extend([np.zeros(1280) for i in range(100)])
features_fuse.extend(sph_features)
labels_fuse.extend(['SPH' for i in range(len(sph_labels))])
#labels_fuse.extend(sph_labels)

'''
tensorboard_path = rootpath + "tensorboard"
if os.access(tensorboard_path, os.F_OK): shutil.rmtree(tensorboard_path)
os.makedirs(tensorboard_path)
summary_writer = SummaryWriter(log_dir=tensorboard_path)
summary_writer.add_embedding(np.array(lidc_features), lidc_labels, global_step=0, tag='lidc_distribution')
summary_writer.add_embedding(np.array(sph_features), sph_labels, global_step=0, tag='sph_distribution')
for fkey in sph_labels_folds.keys():
	summary_writer.add_embedding(np.array(sph_features), sph_labels_folds[fkey], global_step=int(fkey), tag='sph_distribution')
summary_writer.add_embedding(np.array(features_fuse), labels_fuse, global_step=0, tag='overall_distribution')
summary_writer.close()
'''

plt.imshow(np.array(features_fuse), cmap=plt.cm.gray)
plt.show()