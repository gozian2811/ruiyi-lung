import os
import sys
import copy
import time
import math
import fire
import shutil
import numpy as np
import pandas as pd
from configs import conf
opt = conf.DefaultConfig()
os.environ['OMP_NUM_THREADS'] = opt.Num_Threads
os.environ['CUDA_VISIBLE_DEVICES'] = opt.Devices_ID
import torch
import models
import datasets
from utils import Visualizer
from torch.utils.data import DataLoader
from torch import nn
from torchnet import meter
from tensorboardX import SummaryWriter
from sklearn.metrics import mean_squared_error
sys.path.append("/home/fyl/programs/lung_project")
from toolbox import BasicTools as bt
from auxiliary import get_filelists_patientwise, get_detection_filelists, dataflow

try:
	from tqdm import tqdm # long waits are not fun
except:
	print('tqdm not installed')
	tqdm = lambda x : x

output_size_dict = {'bincregression':1, 'sinclass':1, 'binclass':2, 'ternclass':3}
if opt.label_mode in output_size_dict.keys():
	output_size = output_size_dict[opt.label_mode]
	confusion_size = max(2, output_size)
elif opt.label_mode is not None and 'class' in opt.label_mode:
	output_size = 2
	confusion_size = 2
else:
	output_size = 1

def train(**kwargs):
	opt.parse(kwargs)
	e = math.e
	if hasattr(opt, 'lr_decay'):
		y = opt.lr_decay['fin']*e/(e-1)-opt.lr_decay['ini']/(e-1)
		x = (opt.lr_decay['fin']-opt.lr_decay['ini'])*e/(e-1)
	save_path = opt.model_root + '/' + opt.env
	tensorboard_path = save_path + '/tensorboard'
	if os.access(save_path, os.F_OK):
		if opt.save_path_clear:
			shutil.rmtree(save_path)
	else:
		os.makedirs(save_path)
	if os.access(tensorboard_path, os.F_OK):
		shutil.rmtree(tensorboard_path)
	os.makedirs(tensorboard_path)
	summary_writer = SummaryWriter(logdir=tensorboard_path)
	vis = Visualizer(opt.env)
	sumtext = ''
	'''
	if opt.model not in ('MulticropNet', 'EnsembleNet', 'resnet50'):
		#if opt.model not in ('BasicNet', 'BNNet', 'WeaklySharedNet', 'WeaklySharedBNNet', 'MultiTaskNet'):
		if opt.model in ('DensecropNet_Iterative', 'DensecropNet', 'DensecropNet_MultiTask', 'DenseCropNet_ParamWise'):
			#vis.log("connection_mode:{} average_pool:{}" .format(opt.connection_mode, opt.average_pool))
			sumtext += "connection_mode:{} average_pool:{}\n" .format(opt.connection_mode, opt.average_pool)
		#vis.log("num_blocks:{} growth_rate:{} num_final_growth:{}" .format(opt.num_blocks, opt.growth_rate, opt.num_final_growth))
		sumtext += "num_blocks:{} growth_rate:{} num_final_growth:{}\n" .format(opt.num_blocks, opt.growth_rate, opt.num_final_growth)
	'''
	for skey in opt.model_setup.keys():
		sumtext += "{}:{}\n" .format(skey, opt.model_setup[skey])
	if hasattr(opt, 'param_train_mode'):
		sumtext += "param_train_mode:{}\n" .format(opt.param_train_mode)
	if hasattr(opt, 'loss_weightlist'):
		sumtext += "loss_weightlist:{}\n" .format(opt.loss_weightlist)
	#if hasattr(opt, 'normps'):
	#	sumtext += "normps:{}\n" .format(opt.normps)
	sumtext += "environment:{} input_size:{} batch_size:{}\n" .format(opt.env, opt.input_size, opt.batch_size)
	sumtext += "label_mode:{} loss_exp:{} balancing:{}\n" .format(opt.label_mode, opt.loss_exp, opt.balancing)
	for pkey in opt.data_preprocess.keys():
		sumtext += "{}:{}\n" .format(pkey, opt.data_preprocess[pkey])
	vis.log(sumtext)
	summary_writer.add_text(time.strftime('%m%d_%H%M%S'), sumtext)
		
	#model_setup_dict = {}
	#if opt.model in ('MulticropNet', 'DenseNet_Iterative', 'DensecropNet_Iterative', 'DensecropNet', 'DensecropNet_MultiTask', 'BasicNet', 'ParamDenseCropNet', 'BNNet', 'WeaklySharedNet', 'Weakly_sharedBNNet', 'BaseweightSharedNet', 'RegularlySharedNet', 'MultiTaskNet'):
	#	model_setup_dict['input_size'] = opt.input_size

	#model_setup_dict = dict(
		#input_size = opt.input_size,
		#mode=opt.connection_mode,
		#num_blocks=opt.num_blocks,
		#growth_rate=opt.growth_rate,
		#num_fin_growth=opt.num_final_growth,
		#channels = [[64,64,64], [32,output_size]],
		#avg_pool=opt.average_pool,
		#drop_rate=opt.drop_rate,
		#output_size=output_size,

		#pretrained = opt.pretrained,
		#num_copies = num_copies
	#)
	model_setup = opt.model_setup
	if 'EnsembleNet' not in opt.model and 'resnet' not in opt.model.lower() and opt.model!='BaseweightSharedBNNet':
		model_setup['input_size'] = opt.input_size
	if opt.model=='MulticropNet':
		model_setup['channels'][1].append(output_size)
	elif opt.model=='MultiTaskNet':
		model_setup['output_sizes'] = (output_size, output_size)
	else:
		model_setup['output_size'] = output_size
		if hasattr(opt, 'patch_mode'):
			if opt.patch_mode=='ensemble': num_copies = 3
			else: num_copies = 1
			model_setup['num_copies'] = num_copies
		if hasattr(opt, 'patch_mode2'):
			if opt.patch_mode2=='ensemble': num_copies2 = 3
			else: num_copies2 = 1
			model_setup['num_copies2'] = num_copies2
	model = getattr(models, opt.model)(**model_setup)
	'''
	if opt.model=='MulticropNet':
		model = getattr(models, opt.model)(input_size=opt.input_size, channels=[[64,64,64],[32,output_size]])
	elif opt.model=='DenseNet_Iterative':
		model = getattr(models, opt.model)(input_size=opt.input_size, growth_rate=opt.growth_rate, drop_rate=opt.drop_rate, avgpool_size=opt.average_pool, output_size=output_size)
		self._parameters = parameters
		self._buffers = buffers
		self.layers_group = []
		for bg in range(self.num_block_groups):
			blocks = self._make_layer(self.block, self.layer_nums[bg], stride=self.strides[bg])
			for b in range(len(blocks)):
				key_prefix = 'g%d_b%d' %(bg, b)
				bparams = dictionary_extract(parameters, key_prefix)
				bbuffers = dictionary_extract(buffers, key_prefix)
				blocks[b].params = bparams
				blocks[b].buffers = bbuffers
			self.layers_group.append(blocks)
	elif opt.model in ('DensecropNet_Iterative', 'DensecropNet', 'DensecropNet_MultiTask'):
		model = getattr(models, opt.model)(mode=opt.connection_mode, input_size=opt.input_size, num_blocks=opt.num_blocks, growth_rate=opt.growth_rate, num_fin_growth=opt.num_final_growth, avg_pool=opt.average_pool, drop_rate=opt.drop_rate, output_size=output_size)
	elif opt.model in ('BasicNet', 'ParamDenseCropNet', 'BNNet', 'WeaklySharedNet', 'WeaklySharedBNNet'):
		model = getattr(models, opt.model)(input_size=opt.input_size, num_blocks=opt.num_blocks, growth_rate=opt.growth_rate, num_fin_growth=opt.num_final_growth, drop_rate=opt.drop_rate, output_size=output_size)
	elif opt.model in ('BaseweightSharedNet', 'RegularlySharedNet'):
		#hyper_params = dict(input_size=opt.input_size, num_blocks=opt.num_blocks, growth_rate=opt.growth_rate, num_fin_growth=opt.num_final_growth, drop_rate=opt.drop_rate, output_size=output_size)
		#if hasattr(opt, 'normps'): hyper_params['normps'] = opt.normps
		#model = getattr(models, opt.model)(**hyper_params)
		model = getattr(models, opt.model)(normps=opt.normps, input_size=opt.input_size, num_blocks=opt.num_blocks, growth_rate=opt.growth_rate, num_fin_growth=opt.num_final_growth, drop_rate=opt.drop_rate, output_size=output_size)
	elif opt.model=='MultiTaskNet':
		model = getattr(models, opt.model)(input_size=opt.input_size, num_blocks=opt.num_blocks, growth_rate=opt.growth_rate, num_fin_growth=opt.num_final_growth, drop_rate=opt.drop_rate, output_sizes=(output_size, output_size))
	elif opt.model=='EnsembleNet':
		if opt.patch_mode=='ensemble': num_copies = 3
		else: num_copies = 1
		model = getattr(models, opt.model)(pretrained=opt.pretrained, num_copies=num_copies, output_size=output_size)
	else:
		print('Use default config for the network.')
		model = getattr(models, opt.model)()
	'''

	if opt.load_model_path is not None:
		#model.setstream(1)
		model.load(opt.load_model_path)
		#model.setstream(2)
		#model.net1.paramfix()
	else:
		model.save(save_path, opt.env+"_epoch0")
	if opt.transfer: model.transfer(paramfix=False)
	if opt.use_gpu: model.cuda()
	
	if hasattr(opt, 'dataset') and opt.dataset=='Detection_Dataset':
		if hasattr(opt, 'root_dirs'): filepaths = opt.root_dirs
		else: filepaths = [opt.root_dir+'/npy', opt.root_dir+'/npy_non']
		filelists = get_detection_filelists(patient_uids=opt.patientlist, filepaths=filepaths, easy_eliminate_filelist=opt.filelist_easy, config=opt.extract(['model_root', 'env', 'filelists', 'filelist_shuffle', 'num_cross_folds', 'val_fold', 'test_fold']))
		#filelists = get_filelists_patientwise(patient_uids=opt.patientlist, filelists=[opt.rootdir+'/npy', opt.root_dir+'/npy_non'], config=opt.extract(['model_root', 'env', 'filelists', 'filelist_shuffle', 'num_cross_folds', 'val_fold', 'test_fold']))
	else:
		if hasattr(opt, 'dataset'):
			if 'Slicewise' in opt.dataset:
				subdir = 'slicewise'
				fileext = 'png'
			else:
				subdir = 'npy'
				fileext = 'npy'
			if hasattr(opt, 'root_dir'):
				filelists = [opt.root_dir+'/'+subdir]
			else:
				filelists = None
			filelists = get_filelists_patientwise(patient_uids=opt.patientlist, filelists=filelists, fileext=fileext, config=opt.extract(['model_root', 'env', 'filelists', 'filelist_shuffle', 'num_cross_folds', 'val_fold', 'test_fold', 'remove_uncertain']))
		if hasattr(opt, 'dataset2'):
			if 'Slicewise' in opt.dataset2:
				subdir2 = 'slicewise'
				fileext2 = 'png'
			else:
				subdir2 = 'npy'
				fileext2 = 'npy'
			if hasattr(opt, 'root_dir2'):
				filelists2 = [opt.root_dir2+'/'+subdir2]
			else:
				filelists2 = None
			filelists2 = get_filelists_patientwise(patient_uids=opt.patientlist2, filelists=filelists2, fileext=fileext, datasetidx=2, config=opt.extract(['model_root', 'env', 'filelists2', 'filelist_shuffle', 'num_cross_folds', 'val_fold2', 'test_fold2', 'remove_uncertain']))
	dataargs = {'data_size': opt.input_size, 'mode': opt.label_mode}
	dataprepargs = opt.data_preprocess
	if 'filelists' in dir():
		dataargs1 = copy.copy(dataargs)
		if hasattr(opt, 'patch_mode'):
			dataargs1['patch_mode'] = opt.patch_mode
		if hasattr(opt, 'label_collection_file'):
			dataargs1['label_collection'] = opt.root_dir+'/'+opt.label_collection_file
		if 'train' in filelists.keys():
			#train_data = getattr(datasets, opt.dataset)(filelists['train'], opt.input_size, opt.label_mode, file_retrieval=True, translation_num=opt.translation_num, translation_range=opt.translation_range, rotation_num=opt.rotation_num, flip_num=opt.flip_num, noise_range=opt.noise_range)
			train_data = getattr(datasets, opt.dataset)(filelists['train'], file_retrieval=True, **dataargs1, **dataprepargs)
			train_dataloader = DataLoader(train_data,
				batch_size=opt.batch_size,
				shuffle=True,
				num_workers=opt.num_workers,
				drop_last=False)
		if 'val' in filelists.keys():
			val_data = getattr(datasets, opt.dataset)(filelists['val'], **dataargs1)
			val_dataloader = DataLoader(val_data,
				batch_size=opt.batch_size,
				shuffle=False,
				num_workers=opt.num_workers,
				drop_last=False)
		if 'test' in filelists.keys():
			test_data = getattr(datasets, opt.dataset)(filelists['test'], **dataargs1)
			test_dataloader = DataLoader(test_data,
				batch_size=opt.batch_size,
				shuffle=False,
				num_workers=opt.num_workers,
				drop_last=False)
	if 'filelists2' in dir():
		dataargs2 = dataargs
		if hasattr(opt, 'patch_mode2'):
			dataargs2['patch_mode'] = opt.patch_mode2
		if 'train' in filelists2.keys():
			#train_data2 = getattr(datasets, opt.dataset2)(filelists2['train'], (opt.input_size, opt.input_size, opt.input_size), opt.label_mode, file_retrieval=True, translation_num=opt.translation_num, translation_range=opt.translation_range, rotation_num=opt.rotation_num, flip_num=opt.flip_num, noise_range=opt.noise_range)
			train_data2 = getattr(datasets, opt.dataset2)(filelists2['train'], file_retrieval=True, **dataargs2, **dataprepargs)
			if 'train_data' in dir(): batch_size2 = int(opt.batch_size/float(len(train_data))*len(train_data2)+0.5)
			else: batch_size2 = opt.batch_size
			train_dataloader2 = DataLoader(train_data2,
				batch_size=batch_size2,
				shuffle=True,
				num_workers=opt.num_workers,
				drop_last=False)
		if 'val' in filelists2.keys():
			val_data2 = getattr(datasets, opt.dataset2)(filelists2['val'], **dataargs2)
			val_dataloader2 = DataLoader(val_data2,
				batch_size=opt.batch_size,
				shuffle=False,
				num_workers=opt.num_workers,
				drop_last=False)
		if 'test' in filelists2.keys():
			test_data2 = getattr(datasets, opt.dataset2)(filelists2['test'], **dataargs2)
			test_dataloader2 = DataLoader(test_data2,
				batch_size=opt.batch_size,
				shuffle=False,
				num_workers=opt.num_workers,
				drop_last=False)

	lr = opt.lr
	if not hasattr(opt, 'param_train_mode') or opt.param_train_mode=='overall':
		optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=opt.weight_decay)
	else:
		if opt.param_train_mode=='separate':
			optimizer = torch.optim.SGD(model.basic_parameters(), lr = lr, weight_decay=opt.weight_decay)
		elif opt.param_train_mode=='sharing':
			optimizer = torch.optim.SGD(model.sharing_parameters(), lr = lr, weight_decay=opt.weight_decay)
		elif opt.param_train_mode=='inturn':
			optimizers = (torch.optim.SGD(model.basic_parameters(), lr = lr, weight_decay=opt.weight_decay),
				      torch.optim.SGD(model.sharing_parameters(), lr = lr*10, weight_decay=opt.weight_decay))

	if hasattr(opt, 'loss_weightlist'): weightlist = opt.loss_weightlist
	else: weightlist = [1.0]
	#if opt.label_mode in ('sinclass', 'binclass', 'ternclass'):
	if 'class' in opt.label_mode:
		#if opt.model in ('WeaklySharedNet', 'WeaklySharedBNNet', 'RegularlySharedNet', 'BaseweightSharedNet', 'MultiTaskNet'): weightlist = opt.loss_weightlist
		#else: weightlist = [1.0]
		criterion = models.HybridLoss(models.FocalLoss(opt.loss_exp, opt.balancing), weightlist)
	else:
		#if opt.model=='DensecropNet_MultiTask': weightlist = opt.loss_weightlist
		#else: weightlist = [1.0]
		criterion = models.MultiTaskDifferenceLoss(weightlist=weightlist)
	loss_meter = meter.AverageValueMeter()
	if 'confusion_size' in globals().keys():
		confusion_matrix = meter.ConfusionMeter(confusion_size)
		accuracy_meter = meter.AverageValueMeter()
	if 'class' not in opt.label_mode:
		error_meter = meter.AverageValueMeter()
		if opt.model=='DensecropNet_MultiTask': relation_error_meter = meter.AverageValueMeter()
	train_performs = {'loss':[], 'accuracy':[]}
	if 'val_dataloader' in dir(): val_performs = {}
	if 'val_dataloader2' in dir(): val_performs2 = {}
	
	for epoch in range(opt.max_epoch):
		print('{} epoch:{}/{}' .format(opt.env, epoch+1, opt.max_epoch))
		#if opt.dataset=='LIDC_Pathology_Relation_Dataset': train_data.shuffle()
		if 'optimizers' in dir():
			optimizer = optimizers[epoch%len(optimizers)]
		loss_meter.reset()
		if 'confusion_matrix' in dir(): confusion_matrix.reset()
		if 'accuracy_meter' in dir(): accuracy_meter.reset()
		if 'error_meter' in dir(): error_meter.reset()
		if 'relation_error_meter' in dir(): relation_error_meter.reset()
		
		if 'train_dataloader' in dir() and 'train_dataloader2' in dir():
			#batch_iteration = zip(tqdm(train_dataloader), train_dataloader2)
			batch_iteration = bt.tqdm_zip(train_dataloader, train_dataloader2)
			num_iteration = len(train_dataloader)
		elif 'train_dataloader' in dir():
			batch_iteration = tqdm(train_dataloader)
			num_iteration = len(train_dataloader)
		else: 
			batch_iteration = tqdm(train_dataloader2)
			num_iteration = len(train_dataloader2)
		for ii, data_batch in enumerate(batch_iteration):
			optimizer.zero_grad()
			dataflowout = dataflow(data_batch, model, criterion)
			loss = dataflowout['loss']
			loss.backward()
			optimizer.step()
			loss_meter.add(loss.data.item())
			if type(dataflowout['output'])==tuple:
				if 'accuracy_meter' in dir():
					accuracy_meter.add((dataflowout['predicted'][0]==dataflowout['label'][0]).sum().float().cpu()/float(dataflowout['label'][0].size(0)))
					accuracy_meter.add((dataflowout['predicted'][1]==dataflowout['label'][1]).sum().float().cpu()/float(dataflowout['label'][1].size(0)))
				if 'confusion_matrix' in dir():
					confusion_matrix.add(dataflowout['output'][0].data, dataflowout['label'][0])
					confusion_matrix.add(dataflowout['output'][1].data, dataflowout['label'][1])
				if 'error_meter' in dir():
					error_meter.add((dataflowout['target'][0]-dataflowout['output'][0][:,0].data).abs().mean().cpu())
					error_meter.add((dataflowout['target'][1]-dataflowout['output'][1][:,0].data).abs().mean().cpu())
				if 'relation_error_meter' in dir(): 
					relation_error_meter.add((dataflowout['target'][2]-dataflowout['output'][2].data).abs().mean())
			else:
				if 'accuracy_meter' in dir():
					accuracy_meter.add((dataflowout['predicted']==dataflowout['label']).sum().float().cpu()/float(dataflowout['label'].size(0)))
				if 'confusion_matrix' in dir():
					confusion_matrix.add(dataflowout['output'].data, dataflowout['label'])
				if 'target' in dataflowout.keys() and 'error_meter' in dir():
					error_meter.add((dataflowout['target']-dataflowout['output'][:,0].data).abs().mean().cpu())

			if (ii+epoch*num_iteration)%opt.print_freq == opt.print_freq - 1:
				vis.plot('train_loss', loss_meter.value()[0])
				if 'accuracy_meter' in dir(): vis.plot('train_accuracy', accuracy_meter.value()[0])
				if 'error_meter' in dir(): vis.plot('train_error', error_meter.value()[0])
				if 'relation_error_meter' in dir(): vis.plot('train_relation_error', relation_error_meter.value()[0])
				#save_name = opt.env+"_step"+str(epoch*num_iteration+ii)
				#model.save(save_path, save_name+'.pkl')
				#bt.filelist_store(files, save_path+'/'+save_name+'.log')
			if opt.save_freq<0 and (ii+epoch*num_iteration)%(-opt.save_freq) == -opt.save_freq - 1:
				valresult = val(model, val_dataloader, criterion)
				for valkey in valresult.keys():
					if valkey != 'confusion_matrix':
						vis.plot('val_'+valkey, valresult[valkey])
						if valkey not in val_performs.keys(): val_performs[valkey] = []
						val_performs[valkey].append(valresult[valkey])
				save_name = opt.env+"_epoch"+str(epoch+1)+"_batch"+str(ii+1)
				model.save(save_path, save_name)
				print("model saved at:{}" .format(save_path))
			del dataflowout

		train_performs['loss'].append(loss_meter.value()[0])
		if 'accuracy_meter' in dir():
			train_performs['accuracy'].append(accuracy_meter.value()[0])
		if 'error_meter' in dir():
			if 'error' not in train_performs.keys(): train_performs['error'] = []
			train_performs['error'].append(error_meter.value()[0])
		
		if 'test_dataloader' in dir():
			#test_cm,test_accuracy = val(model, test_dataloader)
			testresult = val(model, test_dataloader)
			#vis.plot('test_accuracy', valresult['accuracy'])
			for testkey in testresult.keys():
				if testkey != 'confusion_matrix':
					vis.plot('test_'+testkey, testresult[testkey])
		
		if opt.save_freq>=0:
			if 'val_dataloader' in dir():
				valresult = val(model, val_dataloader, criterion)
				for valkey in valresult.keys():
					if valkey != 'confusion_matrix':
						#vis.plot('val_'+valkey, valresult[valkey])
						vis.plot('val_'+val_data.dataset_name+'_'+valkey, valresult[valkey])
						#summary_writer.add_scalar('val_'+val_data.dataset_name+'_'+valkey, valresult[valkey], epoch)
						if valkey not in val_performs.keys(): val_performs[valkey] = []
						val_performs[valkey].append(valresult[valkey])
			if 'val_dataloader2' in dir():
				oristream = model.stream
				model.setstream(2)
				valresult2 = val(model, val_dataloader2, criterion)
				for valkey in valresult2.keys():
					if valkey != 'confusion_matrix':
						vis.plot('val_'+val_data2.dataset_name+'_'+valkey, valresult2[valkey])
						#summary_writer.add_scalar('val_'+val_data2.dataset_name+'_'+valkey, valresult2[valkey], epoch)
						if valkey not in val_performs2.keys(): val_performs2[valkey] = []
						val_performs2[valkey].append(valresult2[valkey])
				model.setstream(oristream)
			if opt.save_freq==0 or (epoch+1)%opt.save_freq==0 or epoch==0:
				save_confirm = not opt.save_filter	#this assignment controls whether we save model checkpoints when the validation performance decreases
				if 'valresult' in dir():
					if 'accuracy' in valresult.keys():
						current_error = 1 - valresult['accuracy']
					else:
						current_error = valresult['error']
					if not save_confirm and (not hasattr(opt, 'task_bias') or opt.task_bias==1 or opt.task_bias<=0) and ('previous_error' not in dir() or current_error<=previous_error):
						save_confirm = True
					previous_error = current_error
					for valkey in valresult.keys():
						if valkey != 'confusion_matrix':
							vis.plot('val_'+valkey, valresult[valkey])
							summary_writer.add_scalar('val_'+val_data.dataset_name+'_'+valkey, valresult[valkey], epoch)
				if 'valresult2' in dir():
					if 'accuracy' in valresult2.keys():
						current_error2 = 1 - valresult2['accuracy']
					else:
						current_error2 = valresult2['error']
					if not save_confirm and (not hasattr(opt, 'task_bias') or opt.task_bias==2 or opt.task_bias<=0) and ('previous_error2' not in dir() or current_error2<=previous_error2):
						save_confirm = True
					previous_error2 = current_error2
					for valkey in valresult2.keys():
						if valkey != 'confusion_matrix':
							if opt.save_freq>1: vis.plot('val2_'+valkey, valresult2[valkey])
							summary_writer.add_scalar('val_'+val_data2.dataset_name+'_'+valkey, valresult2[valkey], epoch)
			if opt.save_freq>0 and (epoch+1)%opt.save_freq==0 and save_confirm:
				save_name = opt.env+"_epoch"+str(epoch+1)
				model.save(save_path, save_name)
				print("model saved at:{}" .format(save_path))
		for trainkey in train_performs.keys():
			np.save(save_path + '/train_'+trainkey+'.npy', np.array(train_performs[trainkey]))
		if 'val_performs' in dir():
			for valkey in val_performs.keys():
				np.save(save_path + '/val_'+valkey+'.npy', np.array(val_performs[valkey]))
		if 'val_performs2' in dir():
			for valkey in val_performs2.keys():
				np.save(save_path + '/val2_'+valkey+'.npy', np.array(val_performs2[valkey]))
			
		sumtext = "device:{device}, epoch:{epoch}, lr:{lr}, weight_decay:{weight_decay}, loss:{loss}\n".format(
				epoch = epoch,
				device = opt.Devices_ID,
				loss = loss_meter.value()[0],
				lr = lr,
				weight_decay = opt.weight_decay)
		if 'confusion_matrix' in dir():
			sumtext += 'train_cm:{}\n'. format(str(confusion_matrix.value()))
			if 'valresult' in dir():
				sumtext += "val_cm:{}\n". format(str(valresult['confusion_matrix'].value()))
			if 'valresult2' in dir():
				sumtext += "val_cm2:{}". format(str(valresult2['confusion_matrix'].value()))
		vis.log(sumtext)
		summary_writer.add_text(time.strftime('%m%d_%H%M%S'), sumtext, epoch)
		printtext = "train_loss:{}" .format(loss_meter.value()[0])
		if 'accuracy_meter' in dir(): printtext += " train_accuracy:{}" .format(accuracy_meter.value()[0])
		if 'error_meter' in dir(): printtext += " train_error:{}" .format(error_meter.value()[0])
		if 'valresult' in dir():
			if 'accuracy' in valresult.keys(): printtext += " val_accuracy:{}" .format(valresult['accuracy'])
			if 'error' in valresult.keys(): printtext += " val_error:{}" .format(valresult['error'])
		if 'valresult2' in dir():
			if 'accuracy' in valresult2.keys(): printtext += " val_accuracy2:{}" .format(valresult2['accuracy'])
			if 'error' in valresult2.keys(): printtext += " val_error2:{}" .format(valresult2['error'])
		print(printtext)
		if hasattr(opt, 'lr_decay'):
			freq = int(opt.lr_decay['freq_ini'] + float(opt.lr_decay['freq_fin'] - opt.lr_decay['freq_ini']) * epoch / opt.max_epoch)
			if epoch%freq == freq - 1:
				n = int(epoch/opt.lr_decay['freq_ini'])
				lr_decay = y - x*math.exp(1.0/(1.0+n)-1)
				lr = lr*lr_decay
				for param_group in optimizer.param_groups:
					param_group['lr'] = lr
		if hasattr(opt, 'lr_changes'):
			for change in opt.lr_changes:
				if epoch+1==change[0]:
					lr = change[1]
					for param_group in optimizer.param_groups:
						param_group['lr'] = lr
					break

	save_name = opt.env+"_epoch"+str(opt.max_epoch)
	model.save(save_path, save_name)
	summary_writer.close()

def val(model, dataloader, criterion=None):

	model.eval()

	if 'confusion_size' in globals().keys():
		confusion_matrix = meter.ConfusionMeter(confusion_size)
	if criterion is not None:
		loss_meter = meter.AverageValueMeter()
	if 'class' not in opt.label_mode:
		error_meter = meter.AverageValueMeter()
		if opt.model == 'DensecropNet_MultiTask': relation_error_meter = meter.AverageValueMeter()
	for ii, data in enumerate(tqdm(dataloader)):
		dataflowout = dataflow(data, model, criterion)
		if type(dataflowout['output'])==tuple:
			if 'confusion_matrix' in dir():
				confusion_matrix.add(dataflowout['output'][0].data, dataflowout['label'][0])
				confusion_matrix.add(dataflowout['output'][1].data, dataflowout['label'][1])
			if 'error_meter' in dir():
				error_meter.add((dataflowout['target'][0]-dataflowout['output'][0][:,0].data).abs().mean().cpu())
				error_meter.add((dataflowout['target'][1]-dataflowout['output'][1][:,0].data).abs().mean().cpu())
			if 'relation_error_meter' in dir(): relation_error_meter.add((dataflowout['target'][2]-dataflowout['output'][2].data).abs().mean())
			if 'loss' in dataflowout.keys():
				loss_meter.add(dataflowout['loss'].data.abs().cpu())
		else:
			if opt.label_mode in output_size_dict.keys() or 'class' in opt.label_mode:
				confusion_matrix.add(dataflowout['output'].data,  dataflowout['label'])
			if 'loss' in dataflowout.keys():
				loss_meter.add(dataflowout['loss'].data.abs().cpu())
			if 'target' in dataflowout.keys() and 'error_meter' in dir():
				error_meter.add((dataflowout['target']-dataflowout['output'][:,0].data).abs().mean().cpu())
		del dataflowout

	model.train()

	valresult = {}
	if 'confusion_matrix' in dir():
		cm_value = confusion_matrix.value()
		correct = 0
		for index in range(confusion_size):
			correct += cm_value[index][index]
		accuracy = correct/(cm_value.sum())
		valresult['confusion_matrix'] = confusion_matrix
		valresult['accuracy'] = accuracy
	if 'loss_meter' in dir(): valresult['loss'] = loss_meter.value()[0]
	if 'error_meter' in dir(): valresult['error'] = error_meter.value()[0]
	if 'relation_error_meter' in dir(): valresult['relation_error'] = relation_error_meter.value()[0]
	return valresult

def inference():
	if hasattr(opt, 'inference_setup'):
		for epoch in opt.inference_setup['epochs']:
			load_model_path = '{root}/{name}/{name}_epoch{epoch}' .format(root=opt.model_root, name=opt.inference_setup['model_name'], epoch=epoch)
			if os.path.exists(load_model_path):
				test(load_model_path)
			else:
				print("file \"{}\" not exists." .format(load_model_path))
	elif hasattr(opt, 'load_model_path'):
		test(opt.load_model_path)
	else:
		print('no test model to load')
	
def test(load_model_path=opt.load_model_path):
	tensorboard_path = opt.model_root + '/' + opt.env + '/tensorboard'
	if load_model_path is not None:
		storepath = "./experiments_cl"
		storename = os.path.splitext(os.path.basename(load_model_path))[0]
		testpathsplit = os.path.basename(opt.filelists['test']).split('_')
		resultpath = storepath + '/' + storename
		if len(testpathsplit)>1:
			setlabel = testpathsplit[1]
			resultpath += '_' + setlabel
		if hasattr(opt, 'task_bias') and opt.task_bias>0:
			resultpath += '_' + str(opt.task_bias)
		tensorboard_path = os.path.dirname(load_model_path) + '/distribution'
		if os.access(resultpath, os.F_OK): shutil.rmtree(resultpath)
		os.makedirs(resultpath)
		if opt.dataset!='Detection_Dataset':
			featurepath = resultpath+'/'+storename+'_feature'
			if not os.access(featurepath, os.F_OK): os.makedirs(featurepath)
			if hasattr(opt, 'dataset2'):
				featurepath2 = resultpath+'/'+storename+'_feature2'
				if not os.access(featurepath2, os.F_OK): os.makedirs(featurepath2)
		#if os.access(tensorboard_path, os.F_OK): shutil.rmtree(tensorboard_path)
		#os.makedirs(tensorboard_path)
		#summary_writer = SummaryWriter(logdir=tensorboard_path)

	model_setup = opt.model_setup
	if 'EnsembleNet' not in opt.model and 'resnet' not in opt.model.lower() and opt.model!='BaseweightSharedBNNet':
		model_setup['input_size'] = opt.input_size
	if opt.model=='MulticropNet':
		model_setup['channels'][1].append(output_size)
	elif opt.model=='MultiTaskNet':
		model_setup['output_sizes'] = (output_size, output_size)
	else:
		model_setup['output_size'] = output_size
		if hasattr(opt, 'patch_mode'):
			if opt.patch_mode=='ensemble': num_copies = 3
			else: num_copies = 1
			model_setup['num_copies'] = num_copies
		if hasattr(opt, 'patch_mode2'):
			if opt.patch_mode2=='ensemble': num_copies2 = 3
			else: num_copies2 = 1
			model_setup['num_copies2'] = num_copies2
	model = getattr(models, opt.model)(**model_setup).eval()

	if load_model_path is not None:
		print("load model from: {}" .format(load_model_path))
		model.load(load_model_path)
	else:
		print("no model loaded, randomly initialized")
	if opt.use_gpu:model.cuda()
	#model.print_distributions()
	#exit()

	dataargs = {'data_size': opt.input_size, 'mode':opt.label_mode}
	if hasattr(opt, 'label_collection_file'):
		dataargs['label_collection'] = opt.root_dir+'/'+opt.label_collection_file
	dataargs1 = copy.copy(dataargs)
	if hasattr(opt, 'patch_mode'):
		dataargs1['patch_mode'] = opt.patch_mode
	test_data = getattr(datasets, opt.dataset)(opt.filelists['test'], file_retrieval=True, **dataargs1)
	test_dataloader = DataLoader(test_data,
		batch_size=opt.batch_size,
		shuffle=False,
		num_workers=opt.num_workers,
		drop_last=False)
	if hasattr(opt, 'dataset2'):
		dataargs2 = copy.copy(dataargs)
		if hasattr(opt, 'patch_mode2'):
			dataargs2['patch_mode'] = opt.patch_mode2
		test_data2 = getattr(datasets, opt.dataset2)(opt.filelists2['test'], file_retrieval=True, **dataargs2)
		batch_size2 = opt.batch_size/float(len(test_data))*len(test_data2)
		if opt.task_bias==1:
			batch_size2 = int(math.floor(batch_size2))
		elif opt.task_bias==2:
			batch_size2 = int(math.ceil(batch_size2))
		else:
			batch_size2 = int(batch_size2+0.5)
		#if math.ceil(len(test_data)/opt.batch_size)!=math.ceil(len(test_data2)/batch_size2):
		#	batch_size2 = opt.batch_size
		test_dataloader2 = DataLoader(test_data2,
			batch_size=batch_size2,
			shuffle=False,
			num_workers=opt.num_workers,
			drop_last=False)
	'''
	for bs in range(opt.batch_size):
		batch_num = math.ceil(len(test_data)/opt.batch_size)
		batch_num2 = math.ceil(len(test_data2)/int(opt.batch_size/float(len(test_data))*len(test_data2)+0.5))
		if batch_num==batch_num2:
			print(bs)
	'''

	scores = []
	#diameters = []
	if 'confusion_size' in globals().keys():
		retrieval = [[], []]
		corrpreds = []
		incorrpreds = []
		feature_labels = [[], []]
		Confusion_Matrix = meter.ConfusionMeter(confusion_size)
		if confusion_size==2:
			AUC_Calculator = meter.AUCMeter()
	if opt.label_mode!=None and 'class' not in opt.label_mode:
		mses = []
		rretrieval = [[], []]
		regrs = []
		if opt.model=='DensecropNet_MultiTask': rmses = []
	if 'Slicewise' in opt.dataset:
		retrslic = {'data':[], 'pred':[], 'label':[], 'num':[]}
		if 'confusion_size' in globals().keys():
			Confusion_Matrix_Slicesum = meter.ConfusionMeter(confusion_size)
			if confusion_size==2:
				AUC_Calculator_Slicesum = meter.AUCMeter()
	if hasattr(opt, 'dataset2'):
		scores2 = []
		if 'confusion_size' in globals().keys():
			retrieval2 = [[], []]
			corrpreds2 = []
			incorrpreds2 = []
			feature_labels2 = [[], []]
			Confusion_Matrix2 = meter.ConfusionMeter(confusion_size)
			if confusion_size==2:
				AUC_Calculator2 = meter.AUCMeter()
		if 'Slicewise' in opt.dataset2:
			retrslic2 = {'data':[], 'pred':[], 'label':[], 'num':[]}
			if 'confusion_size' in globals().keys():
				Confusion_Matrix_Slicesum2 = meter.ConfusionMeter(confusion_size)
				if confusion_size==2:
					AUC_Calculator_Slicesum2 = meter.AUCMeter()

	if 'test_dataloader2' in dir():
		#batch_iteration = zip(tqdm(test_dataloader), test_dataloader2)
		batch_iteration = bt.tqdm_zip(test_dataloader, test_dataloader2)
	else:
		batch_iteration = tqdm(test_dataloader)
	for index, batch in enumerate(batch_iteration):
		dataflowout = dataflow(batch, model, feature_form='avg_pool')
		score = dataflowout['output']
		if 'label' in dataflowout.keys(): label = dataflowout['label']
		if 'predicted' in dataflowout.keys(): predicted = dataflowout['predicted']
		if 'target' in dataflowout.keys(): target = dataflowout['target']
		if not isinstance(batch, dict):
			if len(score)>=3:
				rscore = score[2]
				rtarget = target[2]
			score2 = score[1]
			score = score[0]
			if 'label' in dir():
				label2 = label[1]
				label = label[0]
			if 'predicted' in dir():
				predicted2 = predicted[1]
				predicted = predicted[0]
			if 'target' in dataflowout.keys():
				target2 = target[1]
				target = target[0]
		if len(score)==1 or ('score2' in dir() and len(score2)==1):
			print("1D list length, continue")
			continue	#The AUC_Calculator is not able to handle input with size 1.
		if 'feature' in dataflowout.keys():
			if isinstance(dataflowout['feature'], tuple):
				feature = dataflowout['feature'][0]
				feature2 = dataflowout['feature'][1]
			else:
				feature = dataflowout['feature']
		if score.size(1)>1:
			prediction = nn.functional.softmax(score, dim=1).data
			probability = prediction[:,label].diag()
			if 'score2' in dir():
				prediction2 = nn.functional.softmax(score2, dim=1).data
				probability2 = prediction2[:,label2].diag()
			if 'confusion_size' in globals().keys():
				Confusion_Matrix.add(score.data, label)
				if 'Confusion_Matrix2' in dir(): Confusion_Matrix2.add(score2.data, label2)
				if confusion_size==2:
					confidence = nn.functional.softmax(score, dim=1)[:,1].data
					AUC_Calculator.add(1-confidence,1-label.squeeze())	#the label is initially defined as 'malignant':0, 'benign':1
					if 'AUC_Calculator2' in dir():
						confidence2 = nn.functional.softmax(score2, dim=1)[:,1].data
						AUC_Calculator2.add(confidence2, label2.squeeze())
		if 'mses' in dir(): mses.extend([mean_squared_error(target.cpu().numpy(), score[:,0].data.cpu().numpy()) for i in range(len(target))])
		if 'rmses' in dir(): rmses.extend([mean_squared_error(rtarget.cpu().numpy(), rscore.data.cpu().numpy()) for i in range(len(target))])
		
		for d in range(len(score)):
			if type(batch)==dict: dataname = os.path.basename(batch['path'][d])
			else: dataname = os.path.basename(batch[0]['path'][d])
			sco = score.data.cpu().numpy()[d]
			if 'target' in dir():
				targ = target.cpu().numpy()[d]
			scores.append(sco)
			#diaminfo = dataname.split('_')[:3]
			#diaminfo.append(sco[0]+35)
			#diameters.append(diaminfo)
			if sco.size>1:
				prob = probability.cpu().numpy()[d]
				lab = label.cpu().numpy()[d]
				pred = predicted.cpu().numpy()[d]
				if pred==lab:
					corrpreds.append(dataname+' '+str(prob))
				else:
					error = 1 - prob
					insidx = 0
					while insidx<len(incorrpreds):
						if error>abs(1-incorrpreds[insidx][1]): break
						insidx += 1
					#incorrpreds.insert(insidx, [dataname, prob, lab])	#order the prediction list by probabilities
					incorrpreds.append([dataname, prob, lab])
				retrieval[0].append(prob)
				retrieval[1].append(lab)
				if 'retrslic' in dir():
					if type(batch)==dict:
						paths = batch['path']
					else:
						paths = batch[0]['path']
					data_name = os.path.basename(paths[d]).split('_')[:-1]
					predn = prediction[d]
					if data_name in retrslic['data']:
						didx = retrslic['data'].index(data_name)
						retrslic['pred'][didx] += predn
						retrslic['num'][didx] += 1
					else:
						retrslic['data'].append(data_name)
						retrslic['pred'].append(predn)
						retrslic['label'].append(lab)
						retrslic['num'].append(1)
				if 'feature' in dir() and 'featurepath' in dir():
					feature_labels[0].append(feature[d].data.cpu().numpy())
					feature_labels[1].append(lab)
					np.save(featurepath + '/' + os.path.splitext(dataname)[0] + '_' + str(lab), feature[d].data.cpu().numpy())
			if 'regrs' in dir():
				error = abs(targ-sco[0])
				insidx = 0
				while insidx<len(regrs):
					if error>abs(regrs[insidx][1]-regrs[insidx][2]): break
					insidx += 1
				#regrs.append([dataname, targ, sco[0]])
				regrs.insert(insidx, [dataname, targ, sco[0]])
			if 'rretrieval' in dir():
				rretrieval[0].append(sco[0])
				rretrieval[1].append(targ)
		if type(batch)!=dict:
			for d in range(len(score2)):
				dataname2 = os.path.basename(batch[1]['path'][d])
				sco2 = score2.data.cpu().numpy()[d]
				scores2.append(sco2)
				if sco2.size>1:
					prob2 = probability2.cpu().numpy()[d]
					lab2 = label2.cpu().numpy()[d]
					pred2 = predicted2.cpu().numpy()[d]
					if pred2==lab2:
						corrpreds2.append(dataname2+' '+str(prob2))
					else:
						error2 = 1 - prob2
						insidx = 0
						while insidx<len(incorrpreds2):
							if error2>abs(1-incorrpreds2[insidx][1]): break
							insidx += 1
						#incorrpreds2.insert(insidx, [dataname2, prob2, lab2])
						incorrpreds2.append([dataname2, prob2, lab2])
					retrieval2[0].append(prob2)
					retrieval2[1].append(lab2)
					if 'retrslic2' in dir():
						paths2 = batch[1]['path']
						data_name2 = os.path.basename(paths2[d]).split('_')[:-1]
						predn2 = prediction2[d]
						if data_name2 in retrslic2['data']:
							didx2 = retrslic2['data'].index(data_name2)
							retrslic2['pred'][didx2] += predn2
							retrslic2['num'][didx2] += 1
						else:
							retrslic2['data'].append(data_name2)
							retrslic2['pred'].append(predn2)
							retrslic2['label'].append(lab2)
							retrslic2['num'].append(1)
					if 'feature2' in dir() and 'featurepath2' in dir():
						feature_labels2[0].append(feature2[d].data.cpu().numpy())
						feature_labels2[1].append(lab2)
						np.save(featurepath2 + '/' + os.path.splitext(dataname2)[0] + '_' + str(lab), feature2[d].data.cpu().numpy())
		del dataflowout

	result_output = open(resultpath+'/'+storename+'_eval.log', 'w')
	if not hasattr(opt, 'task_bias') or opt.task_bias!=2:
		#summary_writer.add_embedding(np.stack(feature_labels[0]), feature_labels[1], test_data.patch_concatenate(), global_step=0, tag='feature')
		#summary_writer.add_embedding(np.stack(feature_labels[0]), feature_labels[1], global_step=1, tag='feature')
		#summary_writer.add_embedding(np.array(scores), feature_labels[1], test_data.patch_concatenate(), global_step=2, tag='feature')
		#summary_writer.add_embedding(np.array(scores), test_data.filenames(), test_data.patch_concatenate(), global_step=3, tag='feature')
		#summary_writer.add_embedding(np.array(scores), feature_labels[1], global_step=4, tag='feature')
		np.save(resultpath+'/'+storename+'_scores.npy', np.array(scores))
		#pd.DataFrame(data=diameters, columns=['patient_id', 'nodule_id', 'diagnosis', 'diameter']).to_excel(resultpath+'/'+'diameters.xlsx', index=False)
		if 'feature_labels' in dir(): np.save(resultpath+'/'+storename+'_labels.npy', np.array(feature_labels[1]))
		if 'corrpreds' in dir(): bt.filelist_store(corrpreds, resultpath+'/'+storename+'_corrpreds.log')
		if 'incorrpreds' in dir(): bt.filelist_store(incorrpreds, resultpath+'/'+storename+'_incorrpreds.log')
		if 'regrs' in dir():
			easiests = []
			for regr in regrs:
				if abs(regr[1]-regr[2])<1.0: easiests.append(regr[0])
			bt.filelist_store(easiests, resultpath+'/'+storename+'_easys.log')
			#bt.filelist_store(regrs, resultpath+'/'+storename+'_regressions.log')
			pd.DataFrame(data=regrs, columns=['data', 'target', 'estimation']).to_csv(resultpath+'/'+storename+'_regressions.csv', index=False)
		if 'rretrieval' in dir(): np.save(resultpath+'/'+storename+'_regressionretrieval', np.array(rretrieval))
		if 'retrieval' in dir(): np.save(resultpath+'/'+storename+'_retrieval', np.array(retrieval))
		if 'Confusion_Matrix' in dir():
			cm_value = Confusion_Matrix.value()
			accuracies = np.diag(cm_value).astype(float) / cm_value.sum(axis=1)
			total_accuracy = np.diag(cm_value).sum().astype(float) / cm_value.sum()
			print("ACC:{} {}" .format(accuracies, total_accuracy))
			result_output.write("ACC:{} {}\n" .format(accuracies, total_accuracy))
		if 'AUC_Calculator' in dir():
			print("AUC: %f"%(AUC_Calculator.value()[0]))	#the values stored in the AUC_Calculator are successively AUC score, True Positive sequence, False Positive sequence
			result_output.write("AUC: %f\n"%(AUC_Calculator.value()[0]))
			np.save(resultpath+'/'+storename+'_roc', np.array([AUC_Calculator.value()[1], AUC_Calculator.value()[2]]))
		if 'Confusion_Matrix_Slicesum' in dir():
			predictions_slicesum = torch.stack(retrslic['pred']).cpu() / torch.Tensor(retrslic['num'])[:,None].float()
			predicted_slicesum = torch.argmax(predictions_slicesum, dim=1)
			labels_slicesum = torch.Tensor(retrslic['label']).long()
			probabilities_slicesum = predictions_slicesum[:,labels_slicesum].diag()
			retrievals_slicesum = torch.stack((probabilities_slicesum, labels_slicesum.float()), dim=0)
			corrpreds_slicesum = []
			incorrpreds_slicesum = []
			for i in range(len(retrslic['data'])):
				if predicted_slicesum[i]==labels_slicesum[i]:
					corrpreds_slicesum.append(('_'.join(retrslic['data'][i]), probabilities_slicesum.numpy()[i], labels_slicesum.numpy()[i]))
				else:
					incorrpreds_slicesum.append(('_'.join(retrslic['data'][i]), probabilities_slicesum.numpy()[i], labels_slicesum.numpy()[i]))
			np.save(resultpath+'/'+storename+'_retrieval_slicesum.npy', retrievals_slicesum.cpu().numpy())
			bt.filelist_store(corrpreds_slicesum, resultpath+'/'+storename+'_corrpreds_slicesum.log')
			bt.filelist_store(incorrpreds_slicesum, resultpath+'/'+storename+'_incorrpreds_slicesum.log')
			Confusion_Matrix_Slicesum.add(predictions_slicesum, labels_slicesum)
			cm_value_slicesum = Confusion_Matrix_Slicesum.value()
			accuracies_slicesum = np.diag(cm_value_slicesum).astype(float) / cm_value_slicesum.sum(axis=1)
			precisions_slicesum = np.diag(cm_value_slicesum).astype(float) / cm_value_slicesum.sum(axis=0)
			total_accuracy_slicesum = np.diag(cm_value_slicesum).sum().astype(float) / cm_value_slicesum.sum()
			print("accuracies slicesum:{} {} {}". format(accuracies_slicesum, precisions_slicesum, total_accuracy_slicesum))
			result_output.write("accuracies slicesum:{} {} {}\n". format(accuracies_slicesum, precisions_slicesum, total_accuracy_slicesum))
		if 'AUC_Calculator_Slicesum' in dir():
			confidence_slicesum = predictions_slicesum[:,0].data
			AUC_Calculator_Slicesum.add(confidence_slicesum, 1-labels_slicesum.squeeze())	#the label is initially defined as 'malignant':0, 'benign':1
			print("AUC Slicesum: %f"%(AUC_Calculator_Slicesum.value()[0]))	#the values stored in the AUC_Calculator are successively AUC score, True Positive sequence, False Positive sequence
			result_output.write("AUC Slicesum: %f\n"%(AUC_Calculator_Slicesum.value()[0]))
		if 'mses' in dir():
			print("MSE: %f" %(np.sqrt(np.array(mses).mean())))
			result_output.write("MSE: %f\n" %(np.sqrt(np.array(mses).mean())))
		if 'rmses' in dir():
			print("Relation MSE: %f" %(np.sqrt(np.array(rmses).mean())))
			result_output.write("Relation MSE: %f\n" %(np.sqrt(np.array(rmses).mean())))

	if 'test_data2' in dir() and (not hasattr(opt, 'task_bias') or opt.task_bias!=1):
		#summary_writer.add_embedding(np.stack(feature_labels2[0]), feature_labels2[1], test_data2.patch_concatenate(), global_step=0, tag='feature2')
		#summary_writer.add_embedding(np.stack(feature_labels2[0]), feature_labels2[1], global_step=1, tag='feature2')
		#summary_writer.add_embedding(np.array(scores2), feature_labels2[1], test_data2.patch_concatenate(), global_step=2, tag='feature2')
		#summary_writer.add_embedding(np.array(scores2), test_data2.filenames(), test_data2.patch_concatenate(), global_step=3, tag='feature2')
		#summary_writer.add_embedding(np.array(scores2), feature_labels2[1], global_step=4, tag='feature2')
		np.save(resultpath+'/'+storename+'_scores2.npy', np.array(scores2))
		if 'feature_labels2' in dir(): np.save(resultpath+'/'+storename+'_labels2.npy', np.array(feature_labels2[1]))
		if 'corrpreds2' in dir(): bt.filelist_store(corrpreds2, resultpath+'/'+storename+'_corrpreds2.log')
		if 'incorrpreds2' in dir(): bt.filelist_store(incorrpreds2, resultpath+'/'+storename+'_incorrpreds2.log')
		if 'retrieval2' in dir(): np.save(resultpath+'/'+storename+'_retrieval2', np.array(retrieval2))
		if 'Confusion_Matrix2' in dir():
			cm_value2 = Confusion_Matrix2.value()
			accuracies2 = np.zeros(confusion_size)
			corrects2 = np.zeros(confusion_size)
			for c in range(confusion_size):
				corrects2[c] = np.diag(cm_value2)[c]
				accuracies2[c] = corrects2[c] / cm_value2[c].sum()
			total_accuracy2 = corrects2.sum() / cm_value2.sum()
			print("ACC2:{} {}" .format(accuracies2, total_accuracy2))
			result_output.write("ACC2:{} {}\n" .format(accuracies2, total_accuracy2))
			if 'AUC_Calculator2' in dir():
				print("AUC2: %f"%(AUC_Calculator2.value()[0]))	#the values stored in the AUC_Calculator are successively AUC score, True Positive sequence, False Positive sequence
				result_output.write("AUC2: %f\n"%(AUC_Calculator2.value()[0]))
				np.save(resultpath+'/'+storename+'_roc2', np.array([AUC_Calculator2.value()[1], AUC_Calculator2.value()[2]]))
			if 'Confusion_Matrix_Slicesum2' in dir():
				predictions_slicesum2 = torch.stack(retrslic2['pred']).cpu() / torch.Tensor(retrslic2['num'])[:,None].float()
				predicted_slicesum2 = torch.argmax(predictions_slicesum2, dim=1)
				labels_slicesum2 = torch.Tensor(retrslic2['label']).long()
				probabilities_slicesum2 = predictions_slicesum2[:,labels_slicesum2].diag()
				retrievals_slicesum2 = torch.stack((probabilities_slicesum2, labels_slicesum2.float()), dim=0)
				corrpreds_slicesum2 = []
				incorrpreds_slicesum2 = []
				for i in range(len(retrslic2['data'])):
					if predicted_slicesum2[i]==labels_slicesum2[i]:
						corrpreds_slicesum2.append(('_'.join(retrslic2['data'][i]), probabilities_slicesum2.numpy()[i], labels_slicesum2.numpy()[i]))
					else:
						incorrpreds_slicesum2.append(('_'.join(retrslic2['data'][i]), probabilities_slicesum2.numpy()[i], labels_slicesum2.numpy()[i]))
				np.save(resultpath+'/'+storename+'_retrieval_slicesum2.npy', retrievals_slicesum2.cpu().numpy())
				bt.filelist_store(corrpreds_slicesum2, resultpath+'/'+storename+'_corrpreds_slicesum2.log')
				bt.filelist_store(incorrpreds_slicesum2, resultpath+'/'+storename+'_incorrpreds_slicesum2.log')
				Confusion_Matrix_Slicesum2.add(predictions_slicesum2, labels_slicesum2)
				cm_value_slicesum2 = Confusion_Matrix_Slicesum2.value()
				accuracies_slicesum2 = np.diag(cm_value_slicesum2).astype(float) / cm_value_slicesum2.sum(axis=1)
				precisions_slicesum2 = np.diag(cm_value_slicesum2).astype(float) / cm_value_slicesum2.sum(axis=0)
				total_accuracy_slicesum2 = np.diag(cm_value_slicesum2).sum().astype(float) / cm_value_slicesum2.sum()
				print("accuracies slicesum2:{} {} {}". format(accuracies_slicesum2, precisions_slicesum2, total_accuracy_slicesum2))
				result_output.write("accuracies slicesum2:{} {} {}\n". format(accuracies_slicesum2, precisions_slicesum2, total_accuracy_slicesum2))
		if 'AUC_Calculator_Slicesum2' in dir():
			confidence_slicesum2 = predictions_slicesum2[:,0].data
			AUC_Calculator_Slicesum2.add(confidence_slicesum2, 1-labels_slicesum2.squeeze())	#the label is initially defined as 'malignant':0, 'benign':1
			print("AUC Slicesum2: %f"%(AUC_Calculator_Slicesum2.value()[0]))	#the values stored in the AUC_Calculator are successively AUC score, True Positive sequence, False Positive sequence
			result_output.write("AUC Slicesum2: %f\n"%(AUC_Calculator_Slicesum2.value()[0]))
	result_output.close()
	#summary_writer.close()

if __name__ == '__main__':
	fire.Fire()
