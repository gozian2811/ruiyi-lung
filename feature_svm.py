from sklearn.svm import LinearSVC
from sklearn import metrics
from random import shuffle
from glob import glob
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/fyl/programs/lung_project')
from toolbox import BasicTools as bt
from toolbox import LIDCTools as lt
from toolbox import Evaluations as eva

def data4svm(filelist, mode):
	#the mode lies in {'malignancy', 'binary'}
	feature_list = []
	label_list = []
	for feature_file in filelist:
		feature = np.load(feature_file)
		if len(feature.shape)==1: feature = np.expand_dims(feature, axis=0)
		if mode=='malignancy':
			label = lt.sample_malignancy_label(feature_file, malignant_positive=True)
		elif mode=='binary':
			label = int(os.path.splitext(feature_file)[0].split('_')[-1])
		feature_list.append(feature)
		label_list.extend([label for i in range(len(feature))])
	features = np.concatenate(feature_list, axis=0)
	labels = np.array(label_list)
	return features, labels

sample_path = "../data_samples/lidc_cubes_64_overbound_ipris"
feature_files, folds = lt.filelist_training(sample_path, ["ipris"], True, False, 5, -1, 1)
train_files = feature_files['train']
test_files = feature_files['test']
#train_path = "./experiments/DensecropNet_Iterative_gr14_agp_fold2_stage2_epoch265_feature_train"
#test_path = "./experiments/DensecropNet_Iterative_gr14_agp_fold2_stage2_epoch265_feature_test"
#train_files = glob(train_path + '/*.npy')
#test_files = glob(test_path + '/*.npy')
shuffle(train_files)
shuffle(test_files)

train_features, train_labels = data4svm(train_files, 'malignancy')
test_features, test_labels = data4svm(test_files, 'malignancy')
train_features /= 1000.0
test_features /= 1000.0

svm_clf = LinearSVC()
svm_clf.fit(train_features, train_labels)
decisions = svm_clf.decision_function(test_features)
predictions = 1/(1+np.exp(-decisions))
np.save('experiments/ipris_fold1_svm_retrieval.npy', np.array([predictions, test_labels]))
eva.binclass_evaluation(test_labels, predictions, "experiments/ipris_fold1_svm_roc")
'''
acc = len(np.where((predictions>0.5)==test_labels)[0]) / float(len(test_labels))
fps, tps, thresholds = metrics.roc_curve(test_labels, predictions)
np.save('experiments/ipris_fold5_svm_roc.npy', np.array([tps, fps]))
#plt.plot(fpr, tpr)
#plt.grid(True)
#plt.xlabel("True Positive Rate")
#plt.ylabel("False Positive Rate")
#plt.savefig('ipris_svm_roc.png')
auc = metrics.auc(fps, tps)
conf = metrics.confusion_matrix(test_labels, predictions>0.5)
sens = conf[1][1] / float(conf[1][0] + conf[1][1])
spec = conf[0][0] / float(conf[0][0] + conf[0][1])
print("accuracy:{} auc:{} sensitivity:{} specificity:{}" .format(acc, auc, sens, spec))
'''