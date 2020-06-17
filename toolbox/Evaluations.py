import os
import math
import copy
import shutil
import pandas as pd
import numpy as np
from sklearn import metrics
#from scipy.interpolate import spline
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter
try:
	from tqdm import tqdm  # long waits are not fun
except:
	print('tqdm not installed')
	tqdm = lambda x: x

def detection_assessment(results_input, annofile, exclfile=None, label=None, refine=False):
	if len(results_input) == 0:
		print('no results to assess')
		return None
	results = copy.deepcopy(results_input)
	true_positive = 0
	false_positive = 0
	annotated_uids = []
	TPwithFPs = []
	FPsperscan = []
	FPsperscan2 = []
	sensitivities = []
	sensitivities2 = []

	CPMscore = 0
	CPMscore2 = 0
	CPMcount = 0
	CPMcount2 = 0
	targetFPsperscan = [0.125, 0.25, 0.5, 1, 2, 4, 8]
	# fpind = 0

	annotations = pd.read_csv(annofile)
	anno_assessed = np.zeros(shape=len(annotations["seriesuid"].values), dtype=bool)
	anno_detected = np.zeros(shape=len(annotations["seriesuid"].values), dtype=bool)
	if exclfile is not None:
		excludes = pd.read_csv(exclfile)

	'''
	#add a true detection for testing
	results = copy.deepcopy(results_input)
	num_results = len(results)
	uids = []
	for ri in range(num_results):
		result = results[ri]
		uid = result[0]
		if uids.count(uid)==0:
			uids.append(uid)
			annolines = (annotations["seriesuid"].values == uid).nonzero()[0]
			for annoline in annolines:
				anno_assessed[annoline] = True
				for i in range(2):	#insert two true result
					annoX = float(annotations["coordX"].values[annoline]) + random.random()
					annoY = float(annotations["coordY"].values[annoline]) + random.random()
					annoZ = float(annotations["coordZ"].values[annoline]) + random.random()
					results.append([uid, annoX, annoY, annoZ, random.random()])
	'''

	predictions = np.zeros(shape=len(results), dtype=float)
	for r in range(len(results)):
		predictions[r] = results[r][-1]
	confsort = predictions.argsort()[::-1]
	uniquesort, uniquecounts = np.unique(predictions, return_counts=True)
	nodules_detected = 0 - np.ones(shape=len(confsort), dtype=int)
	for ci in range(len(confsort)):
		result = results[confsort[ci]]
		uid = result[0]
		coord = np.array(result[1:4], dtype=float)
		uid_conditions = annotations["seriesuid"].values.astype(type(uid))==uid
		if label is not None and 'label' in annotations.keys():
			uid_conditions = np.logical_and(uid_conditions, annotations["label"].values.astype(type(label))==label)
		annolines = uid_conditions.nonzero()[0]
		if len(annolines) > 0 and annotated_uids.count(uid) == 0:
			annotated_uids.append(uid)
		truenodule = False
		nearestanno = -1
		minsquaredist = 2000
		for annoline in annolines:
			anno_assessed[annoline] = True
			annoX = float(annotations["coordX"].values[annoline])
			annoY = float(annotations["coordY"].values[annoline])
			annoZ = float(annotations["coordZ"].values[annoline])
			annocoord = np.array([annoX, annoY, annoZ])
			if "diameter_mm" in annotations.keys():
				annodiam = float(annotations["diameter_mm"].values[annoline])
			elif "diameterX" in annotations.keys() and "diameterY" in annotations.keys() and "diameterZ" in annotations.keys():
				annodiam = np.array([annotations["diameterX"].values[annoline], annotations["diameterY"].values[annoline], annotations["diameterZ"].values[annoline]])
			else:
				annodiam = 3.0
			if (annodiam/2.0-np.abs(coord-annocoord)).min()>0:
			#if abs(coord[0] - annoX) <= annodiam / 2 and abs(coord[1] - annoY) <= annodiam / 2 and abs(coord[2] - annoZ) <= annodiam / 2:
				# assuming that the annotations do not intersect with each other
				truenodule = True
				squaredist = (coord[0] - annoX) * (coord[0] - annoX) + (coord[1] - annoY) * (
				coord[1] - annoY) + (coord[2] - annoZ) * (coord[2] - annoZ)
				if minsquaredist > squaredist:
					minsquaredist = squaredist
					nearestanno = annoline
		if nearestanno >= 0:
			anno_detected[nearestanno] = True

		if not truenodule:
			suspected = False
			if 'excludes' in dir():
				excllines = np.where(excludes["seriesuid"].values == uid)[0]
				for exclline in excllines:
					exclX = float(excludes["coordX"].values[exclline])
					exclY = float(excludes["coordY"].values[exclline])
					exclZ = float(excludes["coordZ"].values[exclline])
					exclcoord = np.array([exclX, exclY, exclZ])
					if "diameterX" in annotations.keys() and "diameterY" in annotations.keys() and "diameterZ" in annotations.keys():
						diameter = np.array([annotations["diameterX"].values[annoline], annotations["diameterY"].values[annoline], annotations["diameterZ"].values[annoline]])
					elif "diameter_mm" in annotations.keys() and annotations["diameter_mm"].values[annoline]>=0:
						diameter = float(annotations["diameter_mm"].values[annoline])
					else:
						diameter = 3.0
					if (diameter/2.0-np.abs(coord-exclcoord)).min()>0:
						suspected = True
						break
					'''
					diameter = excludes["diameter_mm"].values[exclline]
					if diameter < 0:
					#	diameter = 3
					if abs(coord[0] - exclX) <= diameter / 2.0 and abs(
							coord[1] - exclY) <= diameter / 2.0 and abs(
							coord[2] - exclZ) <= diameter / 2.0:
						# print(uid + ":" + str(exclline) + " ignore detecting coordinate {}" .format(coord))		#the coordinate detected is to suspected nodules, so we ignore it in evaluation.
						suspected = True
						break
					'''
			if suspected:
				nodules_detected[ci] = -2
			else:
				false_positive += 1
				# TPwithFPs.append([true_positive, false_positive])
				TPwithFPs.append([np.count_nonzero(anno_detected), false_positive])  # the number detected may be more than the number of annotations, thus we count the number of annotations detected as TT number
		else:
			uniqueind = np.where(uniquesort==predictions[confsort[ci]])[0][0]
			if uniquecounts[uniqueind]>1:
				retrieval_count = uniquecounts[uniqueind+1:].sum()
				while nodules_detected[retrieval_count]>0: retrieval_count += 1
				nodules_detected[retrieval_count] = nearestanno
				best_false_positive = np.count_nonzero(nodules_detected[:retrieval_count]==-1) + 1
				for tf in range(len(TPwithFPs)):
					if TPwithFPs[tf][1]>=best_false_positive:
						TPwithFPs[tf][0] = np.count_nonzero(anno_detected)	#revise the FROC statistics when multiple predictions are with the same score
			else:
				nodules_detected[ci] = nearestanno
			#true_positive += 1
	num_true = np.count_nonzero(anno_assessed)
	if num_true <= 0:
		print('no real nodules for these scans')
		return None
	# calculate the self version of FROC parameters
	num_scans = len(annotated_uids)
	if refine:
		num_FPs = nodules_detected[nodules_detected == -1].size
		previous_true_positive = 0
		previous_true_positive2 = 0
		for true_positive, false_positive in TPwithFPs:
			if true_positive > previous_true_positive:
				FPsperscan.append(false_positive / float(num_scans))
				sensitivity = true_positive / float(num_true)
				sensitivities.append(sensitivity)
				fpord = 0
				for fpi in range(fpord, len(targetFPsperscan)):
					if false_positive == int(num_scans * targetFPsperscan[fpi]):
						CPMscore += sensitivity
						CPMcount += 1
						fpord = fpi + 1
				previous_true_positive = true_positive
			false_positive2 = true_positive + false_positive - num_true
			if false_positive2 > 0 and true_positive > previous_true_positive2:
				FPsperscan2.append(false_positive2 / float(num_scans))
				sensitivity2 = true_positive / float(num_true)
				sensitivities2.append(sensitivity2)
				fpord = 0
				for fpi in range(fpord, len(targetFPsperscan)):
					if false_positive2 == int(num_scans * targetFPsperscan[fpi]):
						CPMscore2 += sensitivity2
						CPMcount2 += 1
						fpord = fpi + 1
				previous_true_positive2 = true_positive
	else:
		for true_positive, false_positive in TPwithFPs:
			fpord = 0
			for fpi in range(fpord, len(targetFPsperscan)):
				if false_positive == int(num_scans * targetFPsperscan[fpi]):
					# fpind += 1
					FPsperscan.append(targetFPsperscan[fpi])
					sensitivity = true_positive / float(num_true)
					sensitivities.append(sensitivity)
					CPMscore += sensitivity
					CPMcount += 1
					fpord = fpi + 1
				# if fpind>=len(targetFPsperscan):
				#	break
				# calculate the stantard version of FROC parameters
		nodules_detected_nosuspected = nodules_detected[nodules_detected >= -1]		#the suspected cites with -2 are eliminated
		for fpperscan in targetFPsperscan:
			scind = int(num_scans * fpperscan) + num_true
			if scind > 0:
				noduleretrieve = nodules_detected_nosuspected[:scind]
				true_positive = np.unique(noduleretrieve[noduleretrieve >= 0]).size
				sensitivity = true_positive / float(num_true)
			else:
				sensitivity = 0
			sensitivities2.append(sensitivity)
			CPMscore2 += sensitivity
			CPMcount2 += 1
		FPsperscan2 = targetFPsperscan
	if CPMcount > 0:
		CPMscore /= float(CPMcount)
	if CPMcount2 > 0:
		CPMscore2 /= float(CPMcount2)
	assessment = {}
	assessment['num_scans'] = num_scans
	assessment['FROC'] = (FPsperscan, sensitivities)
	assessment['CPM'] = CPMscore
	assessment['prediction_order'] = confsort
	assessment['detection_cites'] = nodules_detected
	return assessment

def evaluation_vision(CPMs, num_scans, FPsperscan, sensitivities, CPMscore, nodules_detected, output_path):
	if not os.access(output_path, os.F_OK):
		os.makedirs(output_path)
	#CPM_output = output_path + "/CPMscores.log"
	#FROC_output = output_path + "/froc.png"
	CPMs.append([CPMscore, sensitivities, num_scans])
	CPMoutput = open(output_path + "/CPMscores.log", "w")
	for CPM, sensitivity_list, num_scan in CPMs:
		CPMoutput.write("CPM:{} sensitivities:{} of {} scans\n".format(CPM, sensitivity_list, num_scan))
	CPMoutput.write("detection order:\n{}".format(nodules_detected))
	CPMoutput.close()
	np.save(output_path + "/detection_order.npy", nodules_detected)
	print("CPM:{} sensitivities:{} of {} scans".format(CPMscore, sensitivities, num_scans))
	if len(sensitivities) != len(FPsperscan):
		print("axis incoorect")
		print("sensitivity:{}".format(sensitivities))
		print("FPs number:{}".format(FPsperscan))
	xaxis_range = [i for i in range(min(len(sensitivities), len(FPsperscan)))]
	plt.plot(xaxis_range, sensitivities[:len(xaxis_range)])
	# plt.ylim(0, 1)
	plt.grid(True)
	plt.xlabel("FPs per scan")
	plt.ylabel("sensitivity")
	plt.xticks(xaxis_range, FPsperscan[:len(xaxis_range)])
	plt.savefig(output_path + "/froc.png")
	plt.close()

def FROC_paint(FPsperscan_list, sensitivities_list, name_list, output_file, smooth=False):
	color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
	if len(color_list) < len(FPsperscan_list):
		print('No enough colors, partial exhibition')
		list_length = len(color_list)
	else:
		list_length = len(FPsperscan_list)
	lines = []
	for li in range(list_length):
		'''
		while len(FPsperscan_list[li])>1 and FPsperscan_list[li][1]<0.125:
			FPsperscan_list[li].pop(0)
			sensitivities_list[li].pop(0)
		sensitivities_list[li][0] += (sensitivities_list[li][1] - sensitivities_list[li][0]) / (FPsperscan_list[li][1] - FPsperscan_list[li][0]) * (0.125 - FPsperscan_list[li][0])
		FPsperscan_list[li][0] = 0.125
		for min_length in range(len(FPsperscan_list[li])):
			if FPsperscan_list[li][min_length]>=8.0:
				sensitivities_list[li][min_length] -= (sensitivities_list[li][min_length] - sensitivities_list[li][min_length-1]) / (FPsperscan_list[li][min_length] - FPsperscan_list[li][min_length-1]) * (FPsperscan_list[li][min_length]-8)
				FPsperscan_list[li][min_length] = 8.0
				break
		min_length += 1
		'''
		min_length = min(len(FPsperscan_list[li]), len(sensitivities_list[li]))
		#xaxis_range = [i for i in range(min_length)]
		#line, = plt.plot(xaxis_range, sensitivities_list[li][:len(xaxis_range)], color=color_list[li])
		#lines.append(line)
		#plt.xticks(xaxis_range, FPsperscan_list[li][:len(xaxis_range)])

		if smooth:
			#xaxis_range = [i for i in range(min_length)]
			xnew = np.linspace(FPsperscan_list[li][0], FPsperscan_list[li][min_length-1], 300)
			senssmooth = spline(FPsperscan_list[li], sensitivities_list[li][:min_length], xnew)
			line, = plt.plot(xnew, senssmooth, color=color_list[li])
		else:
			line, = plt.plot(FPsperscan_list[li][:min_length], sensitivities_list[li][:min_length], color=color_list[li])
		lines.append(line)
	plt.legend((lines[0], lines[1], lines[2], lines[3], lines[4]),
		   (name_list[0], name_list[1], name_list[2], name_list[3], name_list[4]), loc="lower right")
	#plt.ylim(0, 1)
	plt.xlim(0.125, 8)
	plt.grid(True)
	plt.xlabel("Average number of false positives per scan")
	plt.ylabel("sensitivity")
	plt.xscale('log', basex=2)
	ax = plt.gca()
	ax.xaxis.set_major_formatter(FixedFormatter([0.125, 0.25, 0.5, 1, 2, 4, 8]))
	ax.xaxis.set_ticks([0.125, 0.25, 0.5, 1, 2, 4, 8])
	#ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
	#plt.show()
	plt.savefig(output_file, format="pdf")
	plt.close()

def FROC_paint_interval(FPsperscan_list, sensitivities_list, name_list, output_file, smooth=False):
	color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
	if len(color_list) < len(FPsperscan_list):
		print('No enough colors, partial exhibition')
		list_length = len(color_list)
	else:
		list_length = len(FPsperscan_list)
	lines = []
	for li in range(list_length):
		'''
		while len(FPsperscan_list[li])>1 and FPsperscan_list[li][1]<0.125:
			FPsperscan_list[li].pop(0)
			sensitivities_list[li].pop(0)
		sensitivities_list[li][0] += (sensitivities_list[li][1] - sensitivities_list[li][0]) / (FPsperscan_list[li][1] - FPsperscan_list[li][0]) * (0.125 - FPsperscan_list[li][0])
		FPsperscan_list[li][0] = 0.125
		for min_length in range(len(FPsperscan_list[li])):
			if FPsperscan_list[li][min_length]>=8.0:
				sensitivities_list[li][min_length] -= (sensitivities_list[li][min_length] - sensitivities_list[li][min_length-1]) / (FPsperscan_list[li][min_length] - FPsperscan_list[li][min_length-1]) * (FPsperscan_list[li][min_length]-8)
				FPsperscan_list[li][min_length] = 8.0
				break
		min_length += 1
		'''
		min_length = min(len(FPsperscan_list[li]), len(sensitivities_list[li]))
		#xaxis_range = [i for i in range(min_length)]
		#line, = plt.plot(xaxis_range, sensitivities_list[li][:len(xaxis_range)], color=color_list[li])
		#lines.append(line)
		#plt.xticks(xaxis_range, FPsperscan_list[li][:len(xaxis_range)])

		if smooth:
			xaxis_range = [i for i in range(min_length)]
			xnew = np.linspace(xaxis_range[0], xaxis_range[-1], 300)
			senssmooth = spline(xaxis_range, sensitivities_list[li][:min_length], xnew)
			ax = plt.gca()
			ax.set_xticklabels(['0.125', '0.25', '0.5', '1', '2', '4', '8'])
			plt.xlim(0, 6)
			line, = plt.plot(xnew, senssmooth, color=color_list[li])
		else:
			#line, = plt.plot(FPsperscan_list[li][:min_length], sensitivities_list[li][:min_length], color=color_list[li])
			xaxis_range = [i for i in range(min_length)]
			ax = plt.gca()
			ax.set_xticklabels(['0.125', '0.25', '0.5', '1', '2', '4', '8'])
			plt.xlim(0, 6)
			line, = plt.plot(xaxis_range, sensitivities_list[li][:min_length], color=color_list[li])
		lines.append(line)
	plt.legend((lines[0], lines[1], lines[2], lines[3], lines[4]),
		   (name_list[0], name_list[1], name_list[2], name_list[3], name_list[4]), loc="lower right")
	plt.ylim(0, 1)
	plt.grid(True)
	'''
	plt.xlabel("Average number of false positives per scan")
	plt.ylabel("sensitivity")
	plt.xscale('log', basex=2)
	ax = plt.gca()
	ax.xaxis.set_major_formatter(FixedFormatter([0.125, 0.25, 0.5, 1, 2, 4, 8]))
	ax.xaxis.set_ticks([0.125, 0.25, 0.5, 1, 2, 4, 8])
	#ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
	#plt.show()
	'''
	plt.savefig(output_file, format="pdf")
	plt.close()

def FROC_paint_old(FPsperscan_list, sensitivities_list, name_list, output_file):
	min_length = len(FPsperscan_list[0])
	for FPsperscan in FPsperscan_list:
		if min_length > len(FPsperscan):
			min_length = len(FPsperscan)
	for sensitivities in sensitivities_list:
		if min_length > len(sensitivities):
			min_length = len(sensitivities)
	color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
	if len(color_list)<len(FPsperscan_list):
		print('No enough colors, partial exhibition')
		list_length = len(color_list)
	else:
		list_length = len(FPsperscan_list)
	lines = []
	for li in range(list_length):
		xaxis_range = [i for i in range(min_length)]
		line, = plt.plot(xaxis_range, sensitivities_list[li][:len(xaxis_range)], color=color_list[li])
		lines.append(line)
		plt.xticks(xaxis_range, FPsperscan_list[li][:len(xaxis_range)])
	plt.legend((lines[0], lines[1], lines[2], lines[3], lines[4]), (name_list[0], name_list[1], name_list[2], name_list[3], name_list[4]), loc="lower right")
	#plt.ylim(0, 1)
	plt.grid(True)
	plt.xlabel("Average number of false positives per scan")
	plt.ylabel("sensitivity")
	plt.savefig(output_file, format="pdf")
	plt.close()

def csv_evaluation(annotation_file, result_file, evaluation_path, exclude_file=None, path_clear=False):
	if path_clear and os.access(evaluation_path, os.F_OK):
		shutil.rmtree(evaluation_path)
	if not os.access(evaluation_path, os.F_OK):
		os.makedirs(evaluation_path)

	results = pd.read_csv(result_file)
	resultlist = []
	if 'class' in results.keys():
		classes = np.unique(results['class'])
		for label in classes:
			labellines = np.where(results['class']==label)[0]
			for l in range(len(labellines)):
				resind = labellines[l]
				uid = results["seriesuid"].values[resind]
				coordX = results["coordX"].values[resind]
				coordY = results["coordY"].values[resind]
				coordZ = results["coordZ"].values[resind]
				prob = results["probability"].values[resind]
				resultlist.append([uid, coordX, coordY, coordZ, prob])
			assessment = detection_assessment(resultlist, annotation_file, exclude_file, label=label)
			if assessment is None:
				print('assessment failed')
				continue
			num_scans = assessment['num_scans']
			FPsperscan, sensitivities = assessment['FROC']
			CPMscore = assessment['CPM']
			prediction_order = assessment['prediction_order']
			nodules_detected = assessment['detection_cites']

			if len(FPsperscan) <= 0 or len(sensitivities) <= 0:
				print("No results to evaluate, continue")
			else:
				evaluation_vision([], num_scans, FPsperscan, sensitivities, CPMscore, nodules_detected, output_path=evaluation_path+'/'+str(label))
	else:
		for r in range(len(results["seriesuid"].values)):
			uid = results["seriesuid"].values[r]
			coordX = results["coordX"].values[r]
			coordY = results["coordY"].values[r]
			coordZ = results["coordZ"].values[r]
			prob = results["probability"].values[r]
			resultlist.append([uid, coordX, coordY, coordZ, prob])

		assessment = detection_assessment(resultlist, annotation_file, exclude_file)
		if assessment is None:
			print('assessment failed')
			exit()
		num_scans = assessment['num_scans']
		FPsperscan, sensitivities = assessment['FROC']
		CPMscore = assessment['CPM']
		prediction_order = assessment['prediction_order']
		nodules_detected = assessment['detection_cites']
		#num_scans, FPsperscan, sensitivities, CPMscore, FPsperscan2, sensitivities2, CPMscore2, nodules_detected = assessment

		if len(FPsperscan) <= 0 or len(sensitivities) <= 0:
			print("No results to evaluate, continue")
		else:
			evaluation_vision([], num_scans, FPsperscan, sensitivities, CPMscore, nodules_detected, output_path=evaluation_path)
		
		num_positive = (nodules_detected>=0).nonzero()[0].size
		hard_negatives = []
		for ndi in range(len(nodules_detected)):
			if resultlist[prediction_order[ndi]][-1]<=0.5 or (nodules_detected[:ndi]>=0).nonzero()[0].size==num_positive:
				break
			if nodules_detected[ndi]==-1:
				hard_negatives.append(resultlist[prediction_order[ndi]])
		hard_negatives_frame = pd.DataFrame(data=hard_negatives, columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability'])
		hard_negatives_frame.to_csv(evaluation_path + '/hard_negatives.csv', index=False, float_format='%.4f')

def binclass_evaluation(test_labels, predictions, label_reverse=False, prediction_classwise=False, storename=None):
	if prediction_classwise:
		predictions = np.abs(1 - test_labels - predictions)
	if label_reverse:
		test_labels = 1 - test_labels
		predictions = 1 - predictions
	fps, tps, thresholds = metrics.roc_curve(test_labels, predictions)
	if storename is not None:
		np.save(storename, np.array([tps, fps]))
	auc = metrics.auc(fps, tps)
	conf = metrics.confusion_matrix(test_labels, predictions>0.5)
	sens = conf[1][1] / float(conf[1][0] + conf[1][1])
	spec = conf[0][0] / float(conf[0][0] + conf[0][1])
	acc = (conf[0][0] + conf[1][1]) / float(conf.sum())
	print("accuracy:{} auc:{} sensitivity:{} specificity:{}" .format(acc, auc, sens, spec))
	return {'sensitivity': sens, 'specificity': spec, 'auc': auc, 'accuracy': acc}

def regression_evaluation(gtlevels, scores):
	mse = metrics.mean_squared_error(gtlevels, scores)
	mae = metrics.mean_absolute_error(gtlevels, scores)
	print("root mean squared error:{}" .format(math.sqrt(mse)))
	print("mean squared error:{}".format(mse))
	print("mean absolute error:{}" .format(mae))

def FROC_comparison_main():
	annotation_file = "../LUNA16/csvfiles/annotations_corrected.csv"
	exclude_file = "../LUNA16/csvfiles/annotations_excluded_corrected.csv"
	result_files = ["../results/experiment1/evaluations_20_excluded_lowthreshold/result.csv",
			"../results/experiment1/evaluations_30_excluded_lowthreshold/result.csv",
			"../results/experiment1/evaluations_40_excluded_lowthreshold2/result.csv",
			"../results/experiment1/evaluation_committefusion_excluded_lowthreshold3/result.csv",
			"../results/experiment1/evaluation_latefusion_excluding_lowthreshold/result.csv"]
	evaluation_path = "../results/evaluations_test"
	name_list = ['CNN-20','CNN-30','CNN-40','committe-fusion','late-fusion']
	#csv_evaluation(annotation_file, result_files[2], "../results/experiments/evaluation_40", exclude_file)
	FPsperscan_list = []
	sensitivities_list = []
	for result_file in result_files:
		results = pd.read_csv(result_file)
		resultlist = []
		for r in range(len(results["seriesuid"].values)):
			uid = results["seriesuid"].values[r]
			coordX = results["coordX"].values[r]
			coordY = results["coordY"].values[r]
			coordZ = results["coordZ"].values[r]
			prob = results["probability"].values[r]
			resultlist.append([uid, coordX, coordY, coordZ, prob])

		assessment = detection_assessment(resultlist, annotation_file, exclude_file, True)
		if assessment is None:
			print('{} assessment failed' .format(result_file))
			continue
		num_scans, FPsperscan, sensitivities, CPMscore, FPsperscan2, sensitivities2, CPMscore2, nodules_detected = assessment
		if len(FPsperscan) <= 0 or len(sensitivities) <= 0:
			print("No results to evaluate, continue")
		else:
			FPsperscan_list.append(FPsperscan)
			sensitivities_list.append(sensitivities)
	FROC_paint(FPsperscan_list, sensitivities_list, name_list, evaluation_path + "/froc_comparison.pdf", False)
