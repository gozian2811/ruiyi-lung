import os
import sys
import copy
import glob
import shutil
import skimage
import argparse
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from toolbox import BasicTools as bt
from toolbox import LIDCTools as lt
from toolbox import Evaluations as eva
from toolbox import CTViewer_Multiax as cvm

def mathcurves_paint():
        x = np.arange(-10.,10.,0.01)
        y = 1 / (1 + np.exp(-x))
        plt.plot(x, y)
        plt.grid(True)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig("sigmoid.png", format='png')
        plt.close()

        y = np.tanh(x)
        plt.plot(x, y)
        plt.grid(True)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig("tanh.png", format='png')
        plt.close()

        y = copy.copy(x)
        y[y<0] = 0
        plt.plot(x, y)
        plt.grid(True)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig("ReLU.png", format='png')
        plt.close()

def samples_replace(source_path="/data/fyl/data_samples/tianchild_cubes_overbound/npy_non_fine", target_path="/data/fyl/data_samples/tianchild_cubes_overbound/npy_non"):
	'''
	filelist = glob.glob(source_path+'/*.npy')
	for file in filelist:
		filename = os.path.basename(file)
		target_file = target_path + '/' + filename
		shutil.copyfile(file, target_file)
		print(target_file)
	'''
	bt.directory_arrange(source_path, target_path, mode='move')

def keras_generator_test():
	from keras.preprocessing.image import ImageDataGenerator
	image = skimage.io.imread("F:/lung_project2/lidc_cubes_64_overbound_roughspacing/LIDC-IDRI-1012_1_2.0_[2.       0.722656 0.722656]_oa_s3.png")
	#image = np.random.rand(50, 50, 3)
	plt.imshow(image)
	plt.savefig('temp.png')
	generator = ImageDataGenerator(shear_range=1., data_format='channels_last', fill_mode='constant')
	image_transformed = generator.random_transform(image)
	plt.imshow(image_transformed)
	plt.show()

def list_compare():
	path1 = "/data/fyl/data_samples/sph_overbound/slicewise"
	path2 = "/data/fyl/data_samples/tianchild_cubes_overbound2/npy"
	samples1 = os.listdir(path1)
	samples2 = os.listdir(path2)
	patients1 = []
	patients2 = []
	for sample in samples1:
		info = '_'.join(os.path.basename(sample).split('_')[:2])
		#info = os.path.basename(sample).split('_')[0]
		if info not in patients1:
			patients1.append(info)
	for sample in samples2:
		#info = '_'.join(os.path.basename(sample).split('_')[:2])
		info = os.path.basename(sample).split('_')[0]
		if info not in patients2:
			patients2.append(info)
	'''
	samples1 = os.listdir("/data/fyl/data_samples/sph2015_overbound/npy")
	samples1.extend(os.listdir("/data/fyl/data_samples/sph2015_overbound/npy_non"))
	statsdict1 = {}
	for sample in samples1:
		patient = sample.split('_')[0]
		if patient in statsdict1.keys():
			statsdict1[patient] += 1
		else:
			statsdict1[patient] = 1
	stats1 = []
	for statitem in statsdict1.items():
		stats1.append(statitem[0]+':'+str(statitem[1]))
	stats2 = bt.filelist_load("/home/fyl/programs/lung_project/data_samples/sph2015_overbound/nodstats.log")[:-1]
	'''
	set1 = set(patients1)
	set2 = set(patients2)
	set_diff1 = set2.difference(set1)
	set_diff2 = set1.difference(set2)
	print(set_diff1)
	print(set_diff2)

def detection_evaluation():
	annotation_file = "/data/fyl/datasets/LUNA16/csvfiles/annotations_corrected.csv"
	#result_file = "pytorch_project/experiments_dt/evaluations_test2/result.csv"
	evaluation_path = 'pytorch_project/experiments_dt/evaluations_sliccand_densecropnet_e11_fold4'
	#eva.csv_evaluation(annotation_file, result_file, evaluation_path, exclude_file=None)
	#results = np.load('pytorch_project/experiments_dt/evaluations_sliccand_densecropnet_e11_fold4/result.npy')
	results_csv = pd.read_csv('pytorch_project/experiments_dt/evaluations_sliccand_densecropnet_e11_fold4/result.csv')
	results = []
	for r in range(len(results_csv["seriesuid"].values)):
		uid = results_csv["seriesuid"].values[r]
		coordX = results_csv["coordX"].values[r]
		coordY = results_csv["coordY"].values[r]
		coordZ = results_csv["coordZ"].values[r]
		prob = results_csv["probability"].values[r]
		results.append([uid, coordX, coordY, coordZ, prob])
	#result_order = results[:,-1].argsort()[::-1]
	#for oi in result_order:
	#	print(results[oi])
	assessment = eva.detection_assessment(results, annotation_file, label=1)
	eva.evaluation_vision([], assessment['num_scans'], assessment['FROC'][0], assessment['FROC'][1], assessment['CPM'], assessment['detection_cites'], output_path=evaluation_path)

def filename_arrange(source_path="sph_cubes_64_overbound/npy", target_path="lung adenocarcinoma samples of the private dataset"):
	filenames = os.listdir(source_path)
	ids = []
	for filename in filenames:
		fnsplit = os.path.splitext(filename)[0].split('_')
		if len(fnsplit)>1:
			if (fnsplit[0], fnsplit[1]) not in ids: ids.append((fnsplit[0], fnsplit[1]))
			idx = ids.index((fnsplit[0], fnsplit[1])) + 1
			label = fnsplit[2]
			#pidx = fnsplit[-1][1:]
			#newname = "sample%d_%s_patch%s.png" %(idx, label, pidx)
			newname = "sample%d_%s.npy" %(idx, label)
			newpath = target_path+'/'+label
			if not os.path.exists(newpath):
				os.makedirs(newpath)
			shutil.copy(source_path+'/'+filename, newpath+'/'+newname)

def featuremap_visualize(files, names = [], num_channels = 20):
	fig = plt.figure()
	#fig = plt.figure(figsize=(12, 3.5))
	#plt.title('channels')
	#fig, ax = plt.subplots(len(files), 1)
	fms = []
	axs = []
	for f in range(len(files)):
		if isinstance(files[f], str):
			featuremap = np.load(files[f])
			ax = fig.add_subplot(len(files), 1, f+1)
			fm = ax.imshow(featuremap[0], vmin=0, vmax=1)
			ax.set_xticks(np.append(np.arange(0, featuremap.shape[2], featuremap.shape[2]//num_channels), featuremap.shape[2]-1))
			ax.set_xticklabels(np.arange(0, num_channels+1))
			ax.set_yticks([])
			#ax.set_yticks([0, featuremap.shape[1]-1])
			#ax.set_yticklabels([0, 1])
			#ax.set_title('channels')
			if f==0: 
				ax.set_xlabel('channels')
				ax.xaxis.set_label_position('top')
			ax.set_ylabel('layer %d' %(f + 1))
			axs.append(ax)
			fms.append(fm)
		else:
			aaxs = []
			file_nums = [len(file_approach) for file_approach in files]
			for l in range(len(files[f])):
				featuremap = np.load(files[f][l])
				ax = fig.add_subplot(sum(file_nums), 1, sum(file_nums[:f])+l+1)
				fm = ax.imshow(featuremap[0], vmin=0, vmax=1)
				ax.set_xticks(np.append(np.arange(0, featuremap.shape[2], featuremap.shape[2]//num_channels), featuremap.shape[2]-1))
				ax.set_xticklabels(np.arange(0, num_channels+1))
				ax.set_yticks([])
				ax.set_ylabel('layer %d'%(l*3+1), rotation=0)
				aaxs.append(ax)
				fms.append(fm)
			aaxs[-1].set_xlabel(names[f])
			#aaxs[0].set_title('channels')
			axs.extend(aaxs)
	#plt.subplots_adjust(left=-1, hspace=5)
	fig.colorbar(fms[0], ax=axs)
	plt.show()
	plt.close()
			
featuremap_visualize(files=[['fmn1l1.npy', 'fmn1l4.npy'], ['fmtr1l1.npy', 'fmtr1l4.npy'], ['fmt1l1.npy', 'fmt1l4.npy'], ['fms1l1.npy', 'fms1l4.npy']], names=['DNC learning simply on LIDC-IDRI', 'transfer learning', 'multitask learning', 'bias-undoing learning'])
'''
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
		    const=sum, default=max, help='sum the integers (default: find the max)')
args = parser.parse_args()
print(args.accumulate(args.integers))
'''
#detection_evaluation()
