import os
import glob
import fire
import models
import torch
import numpy as np
from collections import OrderedDict
from configs.config_ss import DefaultConfig
opt = DefaultConfig()

def parameters_distribution():
	model_paths = ["/data/fyl/models_pytorch/BaseweightSharedNet_classification_rfold1/BaseweightSharedNet_classification_rfold1_epoch747",
			"/data/fyl/models_pytorch/BaseweightSharedNet_classification_rfold2/BaseweightSharedNet_classification_rfold2_epoch617",
			"/data/fyl/models_pytorch/BaseweightSharedNet_classification_rfold3/BaseweightSharedNet_classification_rfold3_epoch627",
			"/data/fyl/models_pytorch/BaseweightSharedNet_classification_rfold4/BaseweightSharedNet_classification_rfold4_epoch552",
			"/data/fyl/models_pytorch/BaseweightSharedNet_classification_rfold5/BaseweightSharedNet_classification_rfold5_epoch740"]
	model = models.BaseweightSharedNet(input_size=opt.input_size, num_blocks=opt.num_blocks, growth_rate=opt.growth_rate, num_fin_growth=opt.num_final_growth, drop_rate=opt.drop_rate, output_size=2).eval()
	paramdict = OrderedDict()

	for path in model_paths:
		model.load(path)
		statedict = model.state_dict()
		for sdkey in statedict.keys():
			paramflatten = statedict[sdkey].view(-1)
			pdkey = sdkey[:-2]
			if pdkey in paramdict.keys():
				paramdict[pdkey].append(paramflatten)
			else:
				paramdict[pdkey] = [paramflatten]

	for pkey in paramdict.keys():
		paramcat = torch.cat(paramdict[pkey])
		print("{}: mean {} standard deviation {}" .format(pkey, torch.mean(paramcat), torch.sqrt(torch.var(paramcat))))
	print()
	for pkey in paramdict.keys():
		paramcat = torch.cat(paramdict[pkey])
		print("{}: min {} max {}" .format(pkey, torch.min(paramcat), torch.max(paramcat)))

def features_distribution():
	feature_paths = ["experiments_cl/BaseweightSharedNet_classification_rfold1_epoch1027_val/BaseweightSharedNet_classification_rfold1_epoch1027_feature",
			"experiments_cl/BaseweightSharedNet_classification_rfold2_epoch521_val/BaseweightSharedNet_classification_rfold2_epoch521_feature",
			"experiments_cl/BaseweightSharedNet_classification_rfold3_epoch658_val/BaseweightSharedNet_classification_rfold3_epoch658_feature",
			"experiments_cl/BaseweightSharedNet_classification_rfold4_epoch1024_val/BaseweightSharedNet_classification_rfold4_epoch1024_feature",
			"experiments_cl/BaseweightSharedNet_classification_rfold5_epoch845_val/BaseweightSharedNet_classification_rfold5_epoch845_feature"]
	feature_paths2 = ["experiments_cl/BaseweightSharedNet_classification_rfold1_epoch747_val/BaseweightSharedNet_classification_rfold1_epoch747_feature2",
			"experiments_cl/BaseweightSharedNet_classification_rfold2_epoch617_val/BaseweightSharedNet_classification_rfold2_epoch617_feature2",
			"experiments_cl/BaseweightSharedNet_classification_rfold3_epoch627_val/BaseweightSharedNet_classification_rfold3_epoch627_feature2",
			"experiments_cl/BaseweightSharedNet_classification_rfold4_epoch552_val/BaseweightSharedNet_classification_rfold4_epoch552_feature2",
			"experiments_cl/BaseweightSharedNet_classification_rfold5_epoch740_val/BaseweightSharedNet_classification_rfold5_epoch740_feature2"]
	features_total = []
	for feature_path in feature_paths:
		features = []
		feature_files = glob.glob(feature_path + '/*.npy')
		for feature_file in feature_files:
			filename = os.path.basename(feature_file[:-6])
			features.append(np.load(feature_file))
			features_total.append(np.load(feature_file))
		featurearray = np.array(features)
		print("feature: mean {} standard deviation {}" .format(featurearray.mean(), np.sqrt(featurearray.var())))
	print("feature_total: mean {} standard deviation {}" .format(np.array(features_total).mean(), np.sqrt(np.array(features_total).var())))
	features_total2 = []
	for feature_path2 in feature_paths2:
		features2 = []
		feature_files2 = glob.glob(feature_path2 + '/*.npy')
		for feature_file2 in feature_files2:
			filename2 = os.path.basename(feature_file2[:-6])
			features2.append(np.load(feature_file2))
			features_total2.append(np.load(feature_file2))
		featurearray2 = np.array(features2)
		print("feature2: mean {} standard deviation {}" .format(featurearray2.mean(), np.sqrt(featurearray2.var())))
	print("feature_total2: mean {} standard deviation {}" .format(np.array(features_total2).mean(), np.sqrt(np.array(features_total2).var())))
	
if __name__ == "__main__":
	fire.Fire()
