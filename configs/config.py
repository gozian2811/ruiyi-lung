import warnings
import os
class BasicConfig(object):
	model_root = 'models_pytorch'
	model_setup = {}
	Num_Threads = '1'

	patientlist = None

	save_path_clear = False
	load_model_path = None
	transfer = False

	use_gpu = True
	num_workers = 0

	max_epoch = 500

	lr = 0.005
	#lr_decay = dict(
	#    freq_ini = 32
	#    freq_fin = 45
	#    ini = 1.0
	#    fin = 1.0
	#)

	loss_exp = 0
	balancing = None
	#drop_rate = 0.15
	weight_decay = 7.4e-4

	#save_freq<0: a shotcut is saved after each '-save_freq' steps of batches;
	#save_freq==0: no shotcut is saved;
	#save_freq>0: a shotcut is saved after each 'save_freq' steps of epochs.
	save_freq = 0
	save_filter = True

	compression_factor = 0.5
	def parse(self,kwargs):
		for k,v in kwargs.items():
			if ':' in k:
				ks = k.split(':')
				if not hasattr(self, ks[0]):
					raise ValueError('Error: opt has no attribute %s' %k)
				attr = getattr(self, ks[0])
				for ki in range(1, len(ks)-1):
					if ks[ki] not in attr.keys():
						raise ValueError('Error: opt has no attribute %s' %k)
					attr = attr[ks[ki]]
				attr[ks[-1]] = v
			if not hasattr(self,k):
				warnings.warn('Warning: opt has no attribute %s' %k)
			setattr(self,k,v)
		print('user config:')
		for k,v in self.__class__.__dict__.items():
			if not k.startswith('__'):
				print(k,getattr(self,k))

	def extract(self, keys):
		confdict = {}
		for key in keys:
			if hasattr(self, key):
				confdict[key] = getattr(self, key)
			else:
				warnings.warn('Warning: opt has no attribute %s' %key)
		return confdict
