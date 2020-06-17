import os
import sys
import glob
import torch
import random
from tqdm import tqdm
from torch.autograd import Variable as vb
sys.path.append("/home/fyl/programs/lung_project")
from toolbox import BasicTools as bt
from toolbox import LIDCTools as lt
from configs import conf
opt = conf.DefaultConfig()

save_path = opt.model_root + '/' + opt.env
	
def get_pathology_filelists(datasetidx=''):
	root_name = 'root_dir'
	if datasetidx is not None: root_name += str(datasetidx)
	if hasattr(opt, root_name) and getattr(opt, root_name) is not None:
		#filelists, folddict = lt.filelist_training(opt.root_dir, remove_uncertain=opt.remove_uncertain, shuffle = opt.filelist_shuffle, cross_fold = opt.num_cross_folds, val_fold = opt.val_fold, test_fold = opt.test_fold)
		file_root = getattr(opt, root_name)
		if os.path.isfile(file_root):
			filelist = bt.filelist_load(file_root)
		else:
			filelist = glob.glob(os.path.join(getattr(opt, root_name), "npy", "*.npy"))
		#mt.sample_statistics(filelist, True)
		if 'lidc' in getattr(opt, root_name) and hasattr(opt, 'remove_uncertain') and opt.remove_uncertain:
			filelist = lt.filelist_remove_uncertain(filelist)
		if opt.filelist_shuffle:
			random.shuffle(filelist)
		filelists, folddict = bt.foldlist(filelist, opt.num_cross_folds, {'val': getattr(opt, 'val_fold'+str(datasetidx)), 'test': getattr(opt, 'test_fold'+str(datasetidx))})
		if 'train' in filelists.keys():
			bt.filelist_store(filelists['train'], save_path+'/filelist'+str(datasetidx)+'_train_fold'+str(folddict['train'])+'.log')
		if 'val' in filelists.keys():
			bt.filelist_store(filelists['val'], save_path+'/filelist'+str(datasetidx)+'_val_fold'+str(folddict['val'])+'.log')
		if 'test' in filelists.keys():
			bt.filelist_store(filelists['test'], save_path+'/filelist'+str(datasetidx)+'_test_fold'+str(folddict['test'])+'.log')
		#vis.log("trainfold:{} valfold:{} testfold:{}" .format(folddict['train'], folddict['val'], folddict['test']))
		print("filelist generated")
	else:
		filelists = {}
		filelists['train'] = bt.filelist_load(getattr(opt,'filelists'+str(datasetidx))['train'])
		filelists['val'] = bt.filelist_load(getattr(opt,'filelists'+str(datasetidx))['val'])
		bt.filelist_store(filelists['train'], save_path+'/filelist_train.log')
		bt.filelist_store(filelists['val'], save_path+'/filelist_val.log')
		if 'test' in getattr(opt,'filelists'+str(datasetidx)).keys():
			filelists['test'] = bt.filelist_load(getattr(opt,'filelists'+str(datasetidx))['test'])
			bt.filelist_store(filelists['test'], save_path+'/filelist_test.log')
		print("filelist loaded")
	return filelists

'''
def get_pathology_filelists(datasetidx='', config):
	save_path = config['model_root'] + '/' + config['env']
	root_name = 'root_dir'
	if datasetidx is not None: root_name += str(datasetidx)
	if config[root_name] and config[root_name] is not None:
		file_root = config[root_name]
		if os.path.isfile(file_root):
			filelist = bt.filelist_load(file_root)
		else:
			filelist = glob.glob(os.path.join(config[root_name], "npy", "*.npy"))
		#mt.sample_statistics(filelist, True)
		if 'lidc' in config[root_name] and 'remove_uncertain' in config.keys() and config['remove_uncertain']:
			filelist = lt.filelist_remove_uncertain(filelist)
		if config['filelist_shuffle']:
			random.shuffle(filelist)
		filelists, folddict = bt.foldlist(filelist, config['num_cross_folds'], {'val': config['val_fold'+str(datasetidx)], 'test': config['test_fold'+str(datasetidx)})
		if 'train' in filelists.keys():
			bt.filelist_store(filelists['train'], save_path+'/filelist'+str(datasetidx)+'_train_fold'+str(folddict['train'])+'.log')
		if 'val' in filelists.keys():
			bt.filelist_store(filelists['val'], save_path+'/filelist'+str(datasetidx)+'_val_fold'+str(folddict['val'])+'.log')
		if 'test' in filelists.keys():
			bt.filelist_store(filelists['test'], save_path+'/filelist'+str(datasetidx)+'_test_fold'+str(folddict['test'])+'.log')
		#vis.log("trainfold:{} valfold:{} testfold:{}" .format(folddict['train'], folddict['val'], folddict['test']))
		print("filelist generated")
	else:
		filelists = {}
		filelists['train'] = bt.filelist_load(config['filelists'+str(datasetidx)]['train'])
		filelists['val'] = bt.filelist_load(config['filelists'+str(datasetidx)]['val'])
		bt.filelist_store(filelists['train'], save_path+'/filelist_train.log')
		bt.filelist_store(filelists['val'], save_path+'/filelist_val.log')
		if 'test' in config['filelists'+str(datasetidx)].keys():
			filelists['test'] = bt.filelist_load(config['filelists'+str(datasetidx)]['test'])
			bt.filelist_store(filelists['test'], save_path+'/filelist_test.log')
		print("filelist loaded")
	return filelists
'''

def get_filelists_patientwise(patient_uids=None, filelists=None, fileext='npy', datasetidx='', config={}):
	save_path = config['model_root'] + '/' + config['env']
	if filelists is None:
		filelistdict = {}
		filelistdict['train'] = bt.filelist_load(config['filelists'+str(datasetidx)]['train'])
		filelistdict['val'] = bt.filelist_load(config['filelists'+str(datasetidx)]['val'])
		bt.filelist_store(filelistdict['train'], save_path+'/filelist_train.log')
		bt.filelist_store(filelistdict['val'], save_path+'/filelist_val.log')
		if 'test' in config['filelists'+str(datasetidx)].keys():
			filelistdict['test'] = bt.filelist_load(config['filelists'+str(datasetidx)]['test'])
			bt.filelist_store(filelistdict['test'], save_path+'/filelist_test.log')
		print("filelist loaded")
	else:
		#filelists=["/home/fyl/datasets/luna_64/train", "/home/fyl/datasets/luna_64/test", "/home/fyl/datasets/npy_non_set"]
		#filelist = []
		if patient_uids is None:
			patient_uids = []
			for filelist in filelists:
				files = glob.glob(filelist+'/*.'+fileext)
				#filelist.extend(files)
				for file in files:
					filename = os.path.basename(file)
					filenamenoext = os.path.splitext(filename)[0]
					fileinfos = filenamenoext.split('_')
					patient_uid = fileinfos[0]
					if patient_uid not in patient_uids: patient_uids.append(patient_uid)
		elif type(patient_uids)==str:
			patient_uids = bt.filelist_load(patient_uids)
		if config['filelist_shuffle']: random.shuffle(patient_uids)
		bt.filelist_store(patient_uids, save_path+'/patientlist'+str(datasetidx)+'.log')
		#bt.filelist_store('luna16samplelist.log', filelist)
		patient_folds, folddict = bt.foldlist(patient_uids, config['num_cross_folds'], {'val': config['val_fold'+str(datasetidx)], 'test': config['test_fold'+str(datasetidx)]})
		filelist_overall = []
		filelistdict = {}
		for setname in patient_folds.keys():
			filelistdict[setname] = []
			'''
			print("filelist {} generating" .format(setname))
			for patient_uid in tqdm(patient_folds[setname]):
				for filelist in filelists:
					if os.path.isfile(filelist):
						files = bt.filelist_load(filelist)
					else:
						files = glob.glob(filelist+'/%s*.%s' %(patient_uid, fileext))
					if 'lidc' in filelist and hasattr(opt, 'remove_uncertain') and opt.remove_uncertain:
						files = lt.filelist_remove_uncertain(files)
					filelist_overall.extend(files)
					filelistdict[setname].extend(files)
			bt.filelist_store(filelistdict[setname], save_path+'/filelist'+str(datasetidx)+'_'+setname+'_fold'+str(folddict[setname])+'.log')
			'''
		for filelist in filelists:
			if os.path.isfile(filelist):
				files = bt.filelist_load(filelist)
			else:
				files = glob.glob(filelist+'/*.%s' %(fileext))
			filelist_overall.extend(files)
			if 'lidc' in filelist and 'remove_uncertain' in config.keys() and config['remove_uncertain']:
				filelist_overall = lt.filelist_remove_uncertain(filelist_overall)
		if config['filelist_shuffle']: random.shuffle(filelist_overall)
		for file in filelist_overall:
			patient_uid = os.path.basename(file).split('_')[0]
			for setname in patient_folds.keys():
				if patient_uid in patient_folds[setname]:
					filelistdict[setname].append(file)
		for setname in patient_folds.keys():
			bt.filelist_store(filelistdict[setname], save_path+'/filelist'+str(datasetidx)+'_'+setname+'_fold'+str(folddict[setname])+'.log')
		bt.filelist_store(filelist_overall, save_path+'/filelist'+str(datasetidx)+'.log')
		print("filelist generated")
	return filelistdict

def get_detection_filelists(patient_uids=None, filepaths=None, easy_eliminate_filelist=None, config={}):
	save_path = config['model_root'] + '/' + config['env']
	if easy_eliminate_filelist is not None: easy_eliminate_filelist = bt.filelist_load(easy_eliminate_filelist)
	if filepaths is None:
		filelists = {}
		filelists['train'] = bt.filelist_load(config['filelists']['train'])
		filelists['val'] = bt.filelist_load(config['filelists']['val'])
		if easy_eliminate_filelist is not None:
			filelists['train'] = bt.filelist_eliminate(filelists['train'], easy_eliminate_filelist)
			filelists['val'] = bt.filelist_eliminate(filelists['val'], easy_eliminate_filelist)
		bt.filelist_store(filelists['train'], save_path+'/filelist_train.log')
		bt.filelist_store(filelists['val'], save_path+'/filelist_val.log')
		print("filelist loaded")
	else:
		#filepaths=["/home/fyl/datasets/luna_64/train", "/home/fyl/datasets/luna_64/test", "/home/fyl/datasets/npy_non_set"]
		#filelist = []
		if patient_uids is None:
			patient_uids = []
			for filepath in filepaths:
				files = glob.glob(filepath+'/*.npy')
				#filelist.extend(files)
				for file in files:
					filename = os.path.basename(file)
					filenamenoext = os.path.splitext(filename)[0]
					fileinfos = filenamenoext.split('_')
					annolabel = fileinfos[-1]
					patient_uid = fileinfos[0]
					if patient_uid not in patient_uids: patient_uids.append(patient_uid)
		elif type(patient_uids)==str:
			patient_uids = bt.filelist_load(patient_uids)
		#patient_temp = patient_uids[int(len(patient_uids)/10.0+0.5)*3:]
		#random.shuffle(patient_temp)
		#patient_uids[int(len(patient_uids)/10.0+0.5)*3:] = patient_temp
		if config['filelist_shuffle']: random.shuffle(patient_uids)
		bt.filelist_store(patient_uids, save_path+'/patientlist.log')
		patient_folds, folddict = bt.foldlist(patient_uids, config['num_cross_folds'], {'val':config['val_fold'], 'test':config['test_fold']})
		
		filelist_overall = []
		filelists = {}
		for setname in patient_folds.keys():
			filelists[setname] = []
		for filelist in filepaths:
			if os.path.isfile(filelist):
				files = bt.filelist_load(filelist)
			else:
				files = os.listdir(filelist)
				for f in range(len(files)):
					files[f] = filelist+'/'+files[f]
				#files = glob.glob(filelist+'/*.%s' %(fileext))
			filelist_overall.extend(files)
			if 'lidc' in filelist and 'remove_uncertain' in config.keys() and config['remove_uncertain']:
				filelist_overall = lt.filelist_remove_uncertain(filelist_overall)
		if easy_eliminate_filelist is not None: filelist_overall = bt.filelist_eliminate(filelist_overall, easy_eliminate_filelist)
		if config['filelist_shuffle']: random.shuffle(filelist_overall)
		for file in filelist_overall:
			filename_split = os.path.splitext(os.path.basename(file))[0].split('_')
			#if 'label_choice' in config.keys() and filename_split[-1]=='annotation' and  config['label_choice']!=filename_split[2]: continue
			patient_uid = filename_split[0]
			for setname in patient_folds.keys():
				if patient_uid in patient_folds[setname]:
					filelists[setname].append(file)
		for setname in patient_folds.keys():
			bt.filelist_store(filelists[setname], save_path+'/filelist_'+setname+'_fold'+str(folddict[setname])+'.log')
		bt.filelist_store(filelist_overall, save_path+'/filelist.log')
		print("filelist generated")
	'''
	filelists = get_filelists_patientwise(patient_uids, filepaths, fileext='npy', config=config)
	for setname in filelists.keys():
		if easy_eliminate_filelist is not None:
			filelists[setname] = bt.filelist_eliminate(filelists[setname], easy_eliminate_filelist)
	'''
	return filelists
	
def dataflow_tuplewise(data_batch, model, criterion=None):
	flowout = {}
	if type(data_batch[0])==list:
		#(data1, label1, files1), (data2, label2, files2) = data_batch
		data1 = vb(data_batch[0][0])
		data2 = vb(data_batch[1][0])
		target1 = vb(data_batch[0][1])
		target2 = vb(data_batch[1][1])
		if opt.label_mode=='bincregression':
			target1 = (target1.float() - 3) / 2.0
			target2 = (target2.float() - 3) / 2.0
		if opt.use_gpu:
			data1 = data1.cuda()
			data2 = data2.cuda()
			target1 = target1.cuda()
			target2 = target2.cuda()
		if opt.model=='DensecropNet_MultiTask':
			output1, output2, output_relation = model(data1, data2)
			if criterion is not None:
				flowout['loss'] = criterion([output1, output2, output_relation], [target1, target2, (target1-target2).abs()]).data
			flowout['predicted'] = (output1[:,0].data<0, output2[:,0].data<0)
			flowout['target'] = (target1.data, target2.data, (target1-target2).abs().data)
			flowout['label'] = (target1.data<0, target2.data<0)
			if opt.label_mode=='bincregression':
				#flowout['output'] = (torch.cat((output1, -output1), dim=1), torch.cat((output2, -output2), dim=1), output_relation)
				output1 = torch.cat((output1, -output1), dim=1)
				output2 = torch.cat((output2, -output2), dim=1)
			flowout['output'] = (output1.data, output2.data, output_relation.data)
		else:
			(output1, feature1), (output2, feature2) = model((data1, data2), fin_feature=True)
			if criterion is not None:
				if opt.use_gpu: criterion.cuda()
				if model.training and opt.model in ('RegularlySharedNet', 'BaseweightSharedNet'):
					flowout['loss'] = criterion((output1, output2), (target1, target2), model.norms()).data
				else:
					flowout['loss'] = criterion((output1, output2), (target1, target2)).data
				#flowout['loss'] = cirterion((output1, output2), (target1, target2))
			flowout['predicted'] = (torch.max(output1.data, 1)[1], torch.max(output2.data, 1)[1])
			flowout['label'] = (target1.data, target2.data)
			flowout['output'] = (output1.data, output2.data)
			flowout['feature'] = (feature1.data, feature2.data)
	else:
		#data, label, files = data_batch
		data = vb(data_batch[0])
		target = vb(data_batch[1])
		if opt.label_mode=='bincregression': target = (target.float() - 3) / 2.0
		if opt.use_gpu:
			data = data.cuda()
			target = target.cuda()
		#output, feature = models.ArchTest(data)
		output, feature = model(data, fin_feature='avg_pool')
		if criterion is not None:
			if opt.use_gpu: criterion.cuda()
			if model.training and opt.model in ('RegularlySharedNet', 'BaseweightSharedNet'):
				flowout['loss'] = criterion(output, target, model.norms()).data
			else:
				flowout['loss'] = criterion(output, target).data
		
		if opt.label_mode in ('binclass', 'ternclass'):
			_, predicted = torch.max(output.data, 1)
			label = target.data
			output_vector = output.data
		elif opt.label_mode=='bincregression':
			#the class 'malignant' is of label 0, while the 'benign' is of label 1
			predicted = output[:,0].data<0
			label = target.data<0
			flowout['target'] = target.data
			output_vector = torch.cat((output, -output), dim=1).data
		else:
			error('unaccepted label mode in function train')
			exit()
		flowout['predicted'] = predicted
		flowout['label'] = label
		flowout['output'] = output_vector
		flowout['feature'] = feature.data
	return flowout
	
def dataflow(data_batch, model, criterion=None, feature_form=None):
	flowout = {}
	if isinstance(data_batch, dict):
		#data_batch = {'data', 'label', 'path'}
		if isinstance(data_batch['data'], list):
			data = []
			for d in range(len(data_batch['data'])):
				datavar = vb(data_batch['data'][d])
				if opt.use_gpu: datavar = datavar.cuda()
				data.append(datavar)
		else:
			data = vb(data_batch['data'])
			if opt.use_gpu: data = data.cuda()
		modelout = model(data, fin_feature=feature_form)
		if isinstance(modelout, torch.Tensor):
			output = modelout
		else:
			output = modelout[0]
			feature = modelout[1]

		if 'feature' in dir():
			flowout['feature'] = feature
		if opt.label_mode is None:
			flowout['output'] = output
		else:
			target = vb(data_batch['label'])
			if opt.label_mode=='bincregression': target = (target.float() - 3) / 2.0
			if opt.use_gpu: target = target.cuda()
			if criterion is not None:
				if opt.use_gpu: criterion.cuda()
				if model.training and opt.model in ('RegularlySharedNet', 'BaseweightSharedNet'):
					flowout['loss'] = criterion(output, target, model.norms())
				else:
					flowout['loss'] = criterion(output, target)
			
			if 'class' in opt.label_mode:
				_, predicted = torch.max(output.data, 1)
				label = target.data
				flowout['output'] = output
				flowout['predicted'] = predicted
				flowout['label'] = label
			elif opt.label_mode=='bincregression':
				#the class 'malignant' is of label 0, while the 'benign' is of label 1
				predicted = output[:,0].data<0
				label = target.data<0
				flowout['target'] = target.data
				flowout['output'] = torch.cat((output, -output), dim=1)
				flowout['predicted'] = predicted
				flowout['label'] = label
			else:
				flowout['target'] = target.data
				flowout['output'] = output
			#else:
			#	error('unaccepted label mode in function train')
			#	exit()
	else:
		#{data1, label1, files1}, {data2, label2, files2} = data_batch
		#data1 = vb(data_batch[0]['data'])
		#data2 = vb(data_batch[1]['data'])
		datas = []
		for g in range(len(data_batch)):
			if isinstance(data_batch[g]['data'], list):
				data = []
				for d in range(len(data_batch[g]['data'])):
					datavar = vb(data_batch[g]['data'][d])
					if opt.use_gpu: datavar = datavar.cuda()
					data.append(datavar)
			else:
				data = vb(data_batch[g]['data'])
				if opt.use_gpu: data = data.cuda()
			datas.append(data)
		target1 = vb(data_batch[0]['label'])
		target2 = vb(data_batch[1]['label'])
		if opt.label_mode=='bincregression':
			target1 = (target1.float() - 3) / 2.0
			target2 = (target2.float() - 3) / 2.0
		if opt.use_gpu:
			#datas[0] = datas[0].cuda()
			#datas[1] = datas[1].cuda()
			target1 = target1.cuda()
			target2 = target2.cuda()
		if opt.model=='DensecropNet_MultiTask':
			output1, output2, output_relation = model(datas[0], datas[1])
			if criterion is not None:
				flowout['loss'] = criterion([output1, output2, output_relation], [target1, target2, (target1-target2).abs()])
			if opt.label_mode=='binclass':
				flowout['predicted'] = (output1[:,0].data<0, output2[:,0].data<0)
				flowout['label'] = (target1.data<0, target2.data<0)
				if opt.label_mode=='bincregression':
					#flowout['output'] = (torch.cat((output1, -output1), dim=1), torch.cat((output2, -output2), dim=1), output_relation)
					output1 = torch.cat((output1, -output1), dim=1)
					output2 = torch.cat((output2, -output2), dim=1)
			flowout['target'] = (target1.data, target2.data, (target1-target2).abs().data)
			flowout['output'] = (output1, output2, output_relation)
		else:
			#this branch do not consider regression tasks
			modelout = model((datas[0], datas[1]), fin_feature=feature_form)
			if isinstance(modelout[0], torch.Tensor):
				output1 = modelout[0]
				output2 = modelout[1]
			else:
				output1 = modelout[0][0]
				output2 = modelout[1][0]
				feature1 = modelout[0][1]
				feature2 = modelout[1][1]
			flowout['output'] = (output1, output2)
			if 'feature1' in dir() and 'feature2' in dir():
				flowout['feature'] = (feature1, feature2)
			if opt.label_mode is not None:
				if criterion is not None:
					if opt.use_gpu: criterion.cuda()
					if model.training and hasattr(model, 'norms'):
						flowout['loss'] = criterion((output1, output2), (target1, target2), model.norms())
					else:
						flowout['loss'] = criterion((output1, output2), (target1, target2))
					#flowout['loss'] = cirterion((output1, output2), (target1, target2))
				flowout['predicted'] = (torch.max(output1.data, 1)[1], torch.max(output2.data, 1)[1])
				flowout['label'] = (target1.data, target2.data)
	return flowout
