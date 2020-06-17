import os
import glob
import shutil
import random
from collections import OrderedDict
try:
	from tqdm import tqdm
except:
	print('tqdm not installed')
	tqdm = lambda x: x

'''
def parse_commandline(*argvs):
	arguments = OrderedDict()
	for a in range(1, len(argvs), 2):
		assert argvs[a][0]=='-', "\'-\' not found in argument names!"
		argname = argvs[a][0]
		arguments[argname] = argvs[a][1]
	return arguments
'''

def read_environment(envfile):
	ef = open(envfile)
	environment = ef.readlines()
	img_width = int(environment[0])
	img_height = int(environment[1])
	num_view = int(environment[2])
	max_bound = int(environment[3])
	min_bound = int(environment[4])
	pixel_mean = float(environment[5])
	ef.close()
	return img_width, img_height, num_view, max_bound, min_bound, pixel_mean

def read_constants(consfile):
	cf = open(consfile)
	constant_dictionary = {}
	parameters = cf.readlines()
	for parameter in parameters:
		parameter_split = parameter.split(':')
		if parameter_split[1].find('\'')>=0:
			constant_dictionary[parameter_split[0]] = parameter_split[1]
		elif parameter_split[1].find('.')>=0:
			constant_dictionary[parameter_split[0]] = float(parameter_split[1])
		else:
			constant_dictionary[parameter_split[0]] = int(parameter_split[1])
	return constant_dictionary

def tqdm_zip(*iterators):
	minidx = 0
	minlength = len(iterators[0])
	for i in range(len(iterators)):
		if minlength > len(iterators[i]):
			minlength = len(iterators[i])
			minidx = i
	iterators_list = list(iterators)
	iterators_list[minidx] = tqdm(iterators_list[minidx])
	return zip(*iterators_list)

def array_couples_combine(array):
	couples = []
	for i in range(len(array)):
		for j in range(i+1, len(array)):
			couples.append((array[i], array[j]))
	return couples

def get_dirs(path):
	dirs = []
	entries = os.listdir(path)
	for entry in entries:
		entry_path = os.path.join(path, entry)
		if os.path.isdir(entry_path):
			dirs.append(entry_path)
	return dirs
	
def filelist_store(filelist, name):
	filelist_storage = open(name, "w")
	for file in filelist:
		filelist_storage.write("{}\n" .format(file))
	filelist_storage.close()

def filelist_load(name):
	filelist_load = open(name, "r")
	filelist = filelist_load.readlines()
	for fi in range(len(filelist)):
		end = len(filelist[fi]) - 1
		while filelist[fi][end]=='\n' or filelist[fi][end]=='\r': end -= 1
		filelist[fi] = filelist[fi][:end+1]
	filelist_load.close()
	return filelist

def dictionary_load(name):
	dictionary = {}
	strload = filelist_load(name)
	for string in strload:
		dictsplit = string.split(':')
		dictionary[dictsplit[0]] = dictsplit[1]
	return dictionary
	
def dict2list(indict):
	outlist = []
	for key in indict.keys():
		outlist.append(indict[key])
	return outlist
	
def dictionary_extend(targdict, sourdict, prefix='', seperator='/'):
	for skey in sourdict.keys():
		targdict[prefix+seperator+skey] = sourdict[skey]
	return targdict

def filelist_eliminate(filelist, eliminatelist):
	fileset = set(filelist)
	eliminateset = set(eliminatelist)
	eliminatedset = fileset.difference(eliminateset)
	eliminatedlist = list(eliminatedset)
	return eliminatedlist

def directory_arrange(source_path, target_path, mode='copy'):
	filelist = os.listdir(source_path)
	for filename in tqdm(filelist):
		file = source_path + '/' + filename
		target_file = target_path + '/' + filename
		if mode=='copy':
			shutil.copyfile(file, target_file)
		else:
			shutil.move(file, target_file)
		#print(target_file)

def foldlist(filelist, num_folds, foldinddict):
	num_total = len(filelist)
	fold_size = int(num_total/float(num_folds)+0.5)
	folds = []
	for i in range(num_folds):
		fold_begin = i * fold_size
		if i<num_folds-1:
			fold_end = (i+1)*fold_size
		else:
			fold_end = len(filelist)
		folds.append(filelist[fold_begin:fold_end])
	
	filelists = {}
	folddict = {}
	foldinds = [i for i in range(len(folds))]
	for foldname in foldinddict.keys():
		foldindex = foldinddict[foldname]
		if foldindex >=1 and foldindex <= num_folds:
			print('{} list selected fold {}' .format(foldname, foldindex))
			filelists[foldname] = folds[foldindex-1]
			folddict[foldname] = foldindex - 1
			foldinds.remove(foldindex-1)
		elif foldindex == 0:
			random_fold = random.choice(foldinds)
			print('{} list randomly selected fold {}' .format(foldname, random_fold+1))
			filelists[foldname] = folds[random_fold]
			folddict[foldname] = random_fold
			foldinds.remove(random_fold)

	for filelistname in filelists.keys():
		folds.remove(filelists[filelistname])
	#for foldname in foldinddict.keys():
	#	folds.pop(foldinddict[foldname])
	filelists['train'] = [file for fold in folds for file in fold]	#the remaining folds are packed as the training set
	folddict['train'] = foldinds
	
	return filelists, folddict

def filelist_training(pfilelist_path=None, nfilelist_path=None, luna_dir=None, luna_trainsets=["subset0", "subset1", "subset2", "subset3", "subset4", "subset5", "subset6", "subset7", "subset8"], luna_valsets=["subset9"], tianchi_dir=None, tianchi_trainsets=["train"], tianchi_valsets=["val"], slh_dir=None, valrate=None, list_store_path=None):
	if pfilelist_path is not None:
		print("read pfilelist from: %s" %(pfilelist_path))
		pfiles = filelist_load(pfilelist_path)
	else:
		pfiles = []
		if luna_dir is not None:
			for set in luna_trainsets:
				luna_traindir = os.path.join(luna_dir, set)
				pdir = os.path.join(luna_traindir,"npy","*.npy")
				pfiles.extend(glob.glob(pdir))
		if tianchi_dir is not None:
			for set in tianchi_trainsets:
				tianchi_traindir = os.path.join(tianchi_dir, set)
				pdir = os.path.join(tianchi_traindir,"npy","*.npy")
				pfiles.extend(glob.glob(pdir))
		if slh_dir is not None:
			pfiles.extend(glob.glob(os.path.join(slh_dir,"npy","*.npy")))
		random.shuffle(pfiles)

	if nfilelist_path is not None:
		print("read nfilelist from: %s" % (nfilelist_path))
		nfiles = filelist_load(nfilelist_path)
	else:
		nfiles = []
		if luna_dir is not None:
			for set in luna_trainsets:
				luna_traindir = os.path.join(luna_dir, set)
				ndir = os.path.join(luna_traindir, "npy_non", "*.npy")
				nfiles.extend(glob.glob(ndir))
		random.shuffle(nfiles)
	if list_store_path is not None:
		filelist_store(pfiles, list_store_path + "/pfilelist.log")
		filelist_store(nfiles, list_store_path + "/nfilelist.log")

	num_positive = len(pfiles)
	num_negative = len(nfiles)
	#num_positive = 10
	#num_negative = 200
	if num_positive==0:
		print("no positive training file found")
		return None
	if valrate is not None:
		positive_val_num = int(num_positive * valrate)
		#positive_val_num = 1
		positive_train_num = num_positive - positive_val_num
		negative_val_num = int(num_negative * valrate)
		negative_train_num = num_negative - negative_val_num
		tpfiles = pfiles[:positive_train_num]
		vpfiles = pfiles[positive_train_num:num_positive]
		tnfiles = nfiles[:negative_train_num]
		vnfiles = nfiles[negative_train_num:num_negative]
	else:
		tpfiles = pfiles
		tnfiles = nfiles
		vpfiles = []
		if luna_dir is not None:
			for set in luna_valsets:
				luna_valdir = os.path.join(luna_dir, set)
				pdir = os.path.join(luna_valdir,"npy","*.npy")
				vpfiles.extend(glob.glob(pdir))
		if tianchi_dir is not None:
			for set in tianchi_valsets:
				tianchi_valdir = os.path.join(tianchi_dir, set)
				pdir = os.path.join(tianchi_valdir,"npy","*.npy")
				vpfiles.extend(glob.glob(pdir))
		vnfiles = []
		if luna_dir is not None:
			for set in luna_valsets:
				luna_valdir = os.path.join(luna_dir, set)
				ndir = os.path.join(luna_valdir, "npy_non", "*.npy")
				vnfiles.extend(glob.glob(ndir))
	return {'tpfiles':tpfiles, 'vpfiles':vpfiles, 'tnfiles':tnfiles, 'vnfiles':vnfiles}
