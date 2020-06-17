from toolbox import BasicTools as bt

def extract_predictions(strlist):
	predlist = ([],[])
	for predstr in strlist:
		predsplit = predstr.split(' ')
		#predlist.append([predsplit[0], float(predsplit[1])])
		predlist[0].append(predsplit[0])
		predlist[1].append(float(predsplit[1]))
	return predlist

def prediction_filter(predlist, lower, upper):
	filteredlist = ([],[])
	for pi in range(len(predlist[0])):
		if predlist[1][pi]>=lower and predlist[1][pi]<=upper:
			filteredlist[0].append(predlist[0][pi])
			filteredlist[1].append(predlist[1][pi])
	return filteredlist

strlist1 = bt.filelist_load('predictions/train/DensecropNet_Iterative_detection_2_stage2_epoch4_corrpreds.log')
predlist1 = extract_predictions(strlist1)
filteredlist1 = prediction_filter(predlist1, 0, 0.1)
predset1 = set(filteredlist1[0])
strlist2 = bt.filelist_load('predictions/train/DensecropNet_Iterative_detection_3_epoch6_corrpreds.log')
predlist2 = extract_predictions(strlist2)
filteredlist2 = prediction_filter(predlist2, 0, 0.1)
predset2 = set(filteredlist2[0])
strlist3 = bt.filelist_load('predictions/train/DensecropNet_Iterative_detection_4_epoch4_corrpreds.log')
predlist3 = extract_predictions(strlist3)
filteredlist3 = prediction_filter(predlist3, 0, 0.1)
predset3 = set(filteredlist3[0])
intersectset = set.intersection(predset1, predset2, predset3)
unionset = set.union(predset1, predset2, predset3)
print(len(predset1), len(predset2), len(predset3))
print(len(intersectset))
print(len(unionset))
intersectlist = list(intersectset)
bt.filelist_store(intersectlist, 'easy_samples_train.log')