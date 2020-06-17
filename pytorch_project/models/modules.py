import os
import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V

class BasicModule(nn.Module):
	'''
	Recapsule nn.Module
	-------------------
	New Methods:
	@Module.save(name):saving model with given name.
	@Module.load(path):load parameters from given path.
	@Module.make_layer(inchannel,outchannel,block,num=1):using given
	block to make a layer which contains number = num blocks.
	'''
	def __init__(self,opt = None):
		'''
		Instantiating with input
		------------------------
		@opt:class defautconfig.
		'''
		super(BasicModule,self).__init__()
		self.model_name = str(type(self))#the default name of model.
		self.opt = opt
	def load(self,path = None):
		'''
		load model with given path
		----
		@path:optional,if not defined, then load by default.
		'''
		if path is None:
			path = self.opt.load_model_path
		state_dict = torch.load(path)
		self_dict = self.state_dict()
		#self_keys = self_dict.keys()
		if state_dict.keys() != self_dict.keys():
			print("state dict do not match.")
			for skey in list(state_dict.keys()):
				if skey not in self_dict.keys():
					print("delete state '%s'" %(skey))
					del state_dict[skey]
			for skey in list(self_dict.keys()):
				if skey not in state_dict.keys():
					print("missing state '%s'" %(skey))
					state_dict[skey] = self_dict[skey]
		self.load_state_dict(state_dict)
	def save(self, path = 'checkpoints', name = None):
		'''
		save model with given name(optional)
		----
		@name: optional, if not define,save with default format(model_name+timepoint).
		'''
		if not os.access(path, os.F_OK):
			os.makedirs(path)
		if name is None:
			name = time.strftime(self.model_name+'_%m%d_%H:%M:%S.pkl')
			#prefix = path+'/'+self.model_name+'_'
			#name = time.strftime(prefix+'%m%d_%H:%M:%S.pkl')
		torch.save(self.state_dict(), path+'/'+name)
		return name
	def print_distributions(self):
		paramdict = self.state_dict()
		distributions = parameter_distributions(paramdict)
		for dkey in distributions.keys():
			print("{}:{}" .format(dkey, distributions[dkey]))

class ParamModule(BasicModule):
	def __init__(self):
		super(ParamModule, self).__init__()
		self.params = None

	def setparams(self, params):
		self.params = params

	def paramfix(self, lastfix=True):
		fix_parameters(self.params, lastfix)
		#for pkey in self.params.keys():
		#	if lastfix or pkey.split('_')[0] != 'lr':
		#		self.params[pkey].requires_grad = False

class ConvModule(ParamModule):
	def __init__(self, input_size=64, num_blocks=3, growth_rate=14, num_fin_growth=1, feature_mode=None, drop_rate=0.2, bare=False):
		super(ConvModule, self).__init__()
		self.num_blocks = num_blocks
		self.drop_rate = drop_rate
		if not bare:
			self.num_blocks = num_blocks
			self.params = nn.ParameterDict()
			num_channels = [max(1, nci*growth_rate) for nci in range(num_blocks)]
			#num_channels.insert(0, num_init_growth*growth_rate)
			num_channels.append(num_fin_growth*growth_rate)
			feature_size = input_size
			for nci in range(num_blocks):
				prefix = 'l' + str(nci)
				conv_size = 3
				#if feature_size>5: feature_size /= 2
				#else: feature_size -= conv_size - 1
				feature_size /= 2
				if nci==num_blocks-1:
					if feature_mode=='final_fuse':
						conv_size = int(input_size/2**nci)
						feature_size = 1
					elif feature_mode=='avg_pool':
						feature_size = 1
				stdv = 1. / math.sqrt(num_channels[nci]*conv_size**3)
				self.params[prefix+'_conv_w'] = nn.Parameter(torch.Tensor(num_channels[nci+1], num_channels[nci], conv_size, conv_size, conv_size).uniform_(-stdv, stdv))
				self.params[prefix+'_conv_b'] = nn.Parameter(torch.Tensor(num_channels[nci+1]).uniform_(-stdv, stdv))
			self.end_feature_size = feature_size

	def forward(self, input):
		fm = input
		for l in range(self.num_blocks):
			convkey = 'l' + str(l) + '_conv'
			#if fm.size(2)>5 and fm.size(2)>self.params[convkey+'_w'].size(2):
			#	fm = F.conv3d(fm, self.params[convkey+'_w'], self.params[convkey+'_b'], padding=1)
			#	fm = F.max_pool3d(fm, 2, 2)
			#else:
			#	fm = F.conv3d(fm, self.params[convkey+'_w'], self.params[convkey+'_b'])
			fm = F.conv3d(fm, self.params[convkey+'_w'], self.params[convkey+'_b'], padding=1)
			fm = F.max_pool3d(fm, 2, 2)
			fm = F.dropout(fm, p=self.drop_rate, training=self.training) 
			if l!=self.num_blocks-1: fm = F.relu(fm)
			if self.end_feature_size==1 and fm.shape[-1]!=1:
				fm = F.avg_pool3d(fm, fm.shape[-3:])
		return fm

class DenseCropModule(ParamModule):
	def __init__(self, input_size=64, num_blocks=3, growth_rate=14, num_fin_growth=1, final_fuse=False, drop_rate=0.2, bare=False):
		super(DenseCropModule, self).__init__()
		self.num_blocks = num_blocks
		self.drop_rate = drop_rate
		if not bare:
			self.num_blocks = num_blocks
			self.params = nn.ParameterDict()
			num_channels = [1+nci*growth_rate for nci in range(num_blocks)]
			#num_channels.insert(0, num_init_growth*growth_rate)
			#num_channels.append(num_fin_growth*growth_rate)
			feature_size = input_size
			for nci in range(num_blocks):
				prefix = 'l' + str(nci)
				conv_size = 3
				#if feature_size>5: feature_size /= 2
				#else: feature_size -= conv_size - 1
				feature_size /= 2
				self.params[prefix+'_conv_w'] = nn.Parameter(torch.Tensor(growth_rate, num_channels[nci], conv_size, conv_size, conv_size))
				nn.init.kaiming_uniform_(self.params[prefix+'_conv_w'], a=math.sqrt(5))
				stdv = 1. / math.sqrt(num_channels[nci]*conv_size**3)
				self.params[prefix+'_conv_b'] = nn.Parameter(torch.Tensor(growth_rate).uniform_(-stdv, stdv))
			if final_fuse:
				conv_size = int(input_size/2**nci)
				feature_size = 1
			else:
				conv_size = 3
				feature_size /= 2
			self.params['l%d_conv_w'%(num_blocks)] = nn.Parameter(torch.Tensor(num_fin_growth*growth_rate, 1+num_blocks*growth_rate, conv_size, conv_size, conv_size))
			nn.init.kaiming_uniform_(self.params['l%d_conv_w'%(num_blocks)], a=math.sqrt(5))
			stdv = 1. / math.sqrt(num_fin_growth*growth_rate*conv_size**3)
			self.params['l%d_conv_b'%(num_blocks)] = nn.Parameter(torch.Tensor(num_fin_growth*growth_rate).uniform_(-stdv, stdv))
			self.end_feature_size = int(feature_size)

	def forward(self, input, fin_feature=None):
		fms_pre = [input]
		for l in range(self.num_blocks):
			fms = []
			for fmi in range(len(fms_pre)):
				fms.append(crop(fms_pre[fmi]).contiguous())
			convkey = 'l' + str(l) + '_conv'
			fm = torch.cat(fms_pre, dim=1).contiguous()
			fm = F.conv3d(fm, self.params[convkey+'_w'], self.params[convkey+'_b'], padding=1)
			fm = F.max_pool3d(fm, 2, 2)
			fm = F.dropout(fm, p=self.drop_rate, training=self.training) 
			fm = F.relu(fm)
			fms.append(fm)
			fms_pre = fms
		convkey = 'l' + str(self.num_blocks) + '_conv'
		fm = torch.cat(fms_pre, dim=1).contiguous()
		fm = F.conv3d(fm, self.params[convkey+'_w'], self.params[convkey+'_b'], padding=1)
		fm = F.max_pool3d(fm, 2, 2)
		fmfinal = F.dropout(fm, p=self.drop_rate, training=self.training) 
		return fmfinal

class SoftlySharedModule(BasicModule):
	def __init__(self, NetModule, **kwargs):
		super(SoftlySharedModule, self).__init__()
		self.NetModule = NetModule
		self.kwargs = kwargs
		self.stream = None

		self.net1 = NetModule(**kwargs)
		self.net2 = NetModule(**kwargs)
		#self.net2.load_state_dict(self.net1.state_dict())	#ensure the two architectures have equal initial parameters
		
	def load(self, path=None):
		if self.stream==1:
			self.net1.load(path)
			self.net1.paramfix()
			self.net2.load_state_dict(self.net1.state_dict())
			self.net2.transfer(paramfix=False)
		elif self.stream==2:
			self.net2.load(path)
			self.net2.paramfix()
			self.net1.load_state_dict(self.net2.state_dict())
			self.net1.transfer(paramfix=False)
		else: super(SoftlySharedModule, self).load(path)

	def setstream(self, stream):
		self.stream = stream

	def forward(self, input, fin_feature=None):
		if type(input)==tuple:
			return self.net1(input[0], fin_feature), self.net2(input[1], fin_feature)
		else:
			if self.stream==2: return self.net2(input, fin_feature)
			else: return self.net1(input, fin_feature)

def crop(x):
	x_size = x.shape[2]
	size = int(x_size/2)
	begin = int(x_size/4)
	end = size + begin
	crop = x[:, :, begin:end, begin:end, begin:end]
	return crop

def concatenate_parameters(params):
	paramlist = []
	for pkey in params.keys():
		if pkey.split('_')[0] != 'lr':
			paramlist.append(params[pkey].view(-1))
	if len(paramlist)==0:
		return 0
	else:
		return torch.cat(paramlist)

def fix_parameters(params, lastfix=True):
	for pkey in params.keys():
		if lastfix or pkey.split('_')[0] != 'lr':
			params[pkey].requires_grad = False

def parameter_distributions(params):
	distributions = OrderedDict()
	for pkey in params.keys():
		distributions[pkey] = (torch.mean(params[pkey]), torch.var(params[pkey], False))
	return distributions
	
def dictionary_extend(targdict, sourdict, prefix='', seperator='_'):
	for skey in sourdict.keys():
		targdict[prefix+seperator+skey] = sourdict[skey]
	return targdict

def dictionary_extract(sourdict, prefix, seperator='_', prefix_include=True):
	resdict = OrderedDict()
	for skey in sourdict.keys():
		if (prefix in skey) == prefix_include:
			rkey = skey
			if prefix_include:
				rkey = rkey.replace(prefix+seperator, '')
			resdict[rkey] = sourdict[skey]
	return resdict

def crop_pool3d(x, steps=1):
	flist = []
	for s in range(steps):
		xc = x
		for c in range(s, steps):
			xc = crop(xc)
		flist.append(xc)
		x = F.max_pool3d(x, 2)
	flist.append(x)
	return torch.cat(flist, dim=1).contiguous()

class multi_crop_pooling(nn.Module):
	def __init__(self, steps):
		super(multi_crop_pooling, self).__init__()
		self.steps = steps
	def forward(self, x):
		flist = []
		for s in range(self.steps):
			xc = x
			for c in range(s, self.steps):
				xc = crop(xc)
			flist.append(xc)
			x = F.max_pool3d(x, 2)
		flist.append(x)
		return torch.cat(flist, dim=1).contiguous()

class denseblock(nn.Module):
	def __init__(self, num_transform, growth_rate, drop_rate):
		super(denseblock, self).__init__()
		self.convs = []
		for t in range(num_transform):
			conv = nn.Sequential(
				nn.BatchNorm3d((t+1)*growth_rate),
				nn.ReLU(inplace = True),
				nn.Conv3d((t+1)*growth_rate, growth_rate, kernel_size = 1, stride = 1, bias = False),
				nn.BatchNorm3d(growth_rate),
				nn.ReLU(inplace = True),
				nn.Conv3d(growth_rate, growth_rate, kernel_size = 3, stride = 1, padding = 1))
			self.convs.append(conv)
			setattr(self, 'conv'+str(t), conv)
	def forward(self, x):
		fminputs = [x]
		for c in range(len(self.convs)-1):
			conv = self.convs[c]
			fminputs.append(conv.forward(torch.cat(fminputs, dim=1).contiguous()))
		return fminputs[-1]
		
class pooling(nn.Module):
	def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
		super(pooling, self).__init__()
		self.conv = nn.Sequential(
			nn.BatchNorm3d(num_input_features),
			nn.ReLU(inplace=True),
			nn.Conv3d(num_input_features, bn_size*growth_rate, kernel_size=1, stride=1, bias=False),
			nn.BatchNorm3d(bn_size * growth_rate),
			nn.ReLU(inplace=True),
			nn.Conv3d(bn_size * growth_rate, num_input_features,kernel_size=3, stride=1, padding=1, bias=False))	#the modified residual network structure
		self.pool = nn.Sequential(
			nn.BatchNorm3d(num_input_features),
			nn.ReLU(inplace = True),
			nn.Conv3d(num_input_features,growth_rate,kernel_size = 3,stride=1,padding=1),
			nn.MaxPool3d(2,2))	#the mixed operation of convolution and pooling
		self.drop_rate = drop_rate
	def forward(self, x):
		x_conved = self.conv(x)
		pool = self.pool(x+x_conved)
		if self.drop_rate > 0:
			pool = F.dropout(pool, p=self.drop_rate, training=self.training)
		return pool

class bottleneck(nn.Module):
	def __init__(self, num_input_features, growth_rate, drop_rate, final = False):
		super(doubconv_pool, self).__init__()
		self.convpool = nn.Sequential(
			nn.BatchNorm3d(num_input_features),
			nn.ReLU(inplace = True),
			nn.Conv3d(num_input_features, growth_rate, kernel_size = 1, stride = 1, bias = False),
			nn.BatchNorm3d(growth_rate),
			nn.ReLU(inplace = True),
			nn.Conv3d(growth_rate, growth_rate, kernel_size = 3, stride = 1, padding = 1, bias = final),
			nn.MaxPool3d(2))	#the mixed operation with convolution and pooling
		self.drop_rate = drop_rate
	def forward(self, x):
		feature_out = self.convpool(x)
		if self.drop_rate > 0:
			feature_out = F.dropout(feature_out, p=self.drop_rate, training=self.training)
		return feature_out

class simpleconv(nn.Module):
	def __init__(self,inchannels,outchannels,drop_rate):
		super(simpleconv,self).__init__()
		self.conv = nn.Sequential(
			nn.BatchNorm3d(inchannels),
			nn.ReLU(inplace=True),
			nn.Conv3d(inchannels,inchannels,kernel_size=1,bias=False),
			nn.BatchNorm3d(inchannels),
			nn.ReLU(inplace=True),
			nn.Conv3d(inchannels,outchannels,kernel_size=3,padding=1,bias = False)
		)
		self.pool = nn.Sequential(
			nn.BatchNorm3d(outchannels),
			nn.ReLU(inplace=True),
			nn.Conv3d(outchannels,outchannels,kernel_size=3,padding=1),
			nn.MaxPool3d(kernel_size=2)
		)
		self.drop_rate = drop_rate
	def forward(self,x):
		out = self.conv(x)
		out = self.pool(out)
		if self.drop_rate > 0:
			out = F.dropout(out, p=self.drop_rate, training=self.training)
		return out
		
class conv_pool(nn.Module):
	def __init__(self, num_input_features, growth_rate, drop_rate, final = False):
		super(conv_pool, self).__init__()
		self.convpool = nn.Sequential(
			nn.BatchNorm3d(num_input_features),
			nn.ReLU(inplace = True),
			nn.Conv3d(num_input_features, growth_rate, kernel_size = 3, stride = 1, padding = 1, bias = final),
			nn.MaxPool3d(2))	#the mixed operation with convolution and pooling
		self.drop_rate = drop_rate
	def forward(self, x):
		feature_out = self.convpool(x)
		if self.drop_rate > 0:
			feature_out = F.dropout(feature_out, p=self.drop_rate, training=self.training)
		return feature_out
		
class doubconv_pool(nn.Module):
	def __init__(self, num_input_features, growth_rate, drop_rate, final = False):
		super(doubconv_pool, self).__init__()
		self.convpool = nn.Sequential(
			nn.BatchNorm3d(num_input_features),
			nn.ReLU(inplace = True),
			nn.Conv3d(num_input_features, growth_rate, kernel_size = 3, stride = 1, padding = 1, bias = False),
			nn.BatchNorm3d(growth_rate),
			nn.ReLU(inplace = True),
			nn.Conv3d(growth_rate, growth_rate, kernel_size = 3, stride = 1, padding = 1, bias = final),
			nn.MaxPool3d(2))	#the mixed operation with convolution and pooling
		self.drop_rate = drop_rate
	def forward(self, x):
		feature_out = self.convpool(x)
		if self.drop_rate > 0:
			feature_out = F.dropout(feature_out, p=self.drop_rate, training=self.training)
		return feature_out
		
class triconv_pool(nn.Module):
	def __init__(self, num_input_features, growth_rate, drop_rate):
		super(triconv_pool, self).__init__()
		self.convpool = nn.Sequential(
			nn.BatchNorm3d(num_input_features),
			nn.ReLU(inplace = True),
			nn.Conv3d(num_input_features, growth_rate, kernel_size = 1, stride = 1, bias = False),
			nn.BatchNorm3d(growth_rate),
			nn.ReLU(inplace = True),
			nn.Conv3d(growth_rate, growth_rate, kernel_size = 3, stride = 1, padding = 1),
			nn.BatchNorm3d(growth_rate),
			nn.ReLU(inplace = True),
			nn.Conv3d(growth_rate, growth_rate, kernel_size = 3, stride = 1, padding = 1),
			nn.MaxPool3d(2))	#the mixed operation with convolution and pooling
		self.drop_rate = drop_rate
	def forward(self, x):
		feature_out = self.convpool(x)
		if self.drop_rate > 0:
			feature_out = F.dropout(feature_out, p=self.drop_rate, training=self.training)
		return feature_out

class doubconv_pool_pwisebn(nn.Module):
	def __init__(self, num_input_features, growth_rate, drop_rate, final=False, inidev=0.01, use_gpu=True):
		super(doubconv_pool_pwise, self).__init__()
		if use_gpu:
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')
		self.params = nn.ParameterDict()
		self.params['convw1'] = nn.Parameter(V(torch.randn((growth_rate, num_input_features, 3, 3, 3), device=device)*inidev, requires_grad=True))
		self.params['convw2'] = nn.Parameter(V(torch.randn((growth_rate, growth_rate, 3, 3, 3), device=device)*inidev, requires_grad=True))
		if final: self.params['convb2'] = nn.Parameter(V(torch.randn((growth_rate), device=device)*inidev, requires_grad=True))
		self.params['bnw1'] = nn.Parameter(V(torch.randn(num_input_features, device=device)*inidev, requires_grad=True))
		self.params['bnb1'] = nn.Parameter(V(torch.randn(num_input_features, device=device)*inidev, requires_grad=True))
		self.params['bnw2'] = nn.Parameter(V(torch.randn(growth_rate, device=device)*inidev, requires_grad=True))
		self.params['bnb1'] = nn.Parameter(V(torch.randn(growth_rate, device=device)*inidev, requires_grad=True))
		self.drop_rate = drop_rate
	def forward(self, x):
		bndims = [d for d in range(x.dim())]
		bndims.pop(1)
		self.params['bnmean1'] = nn.Parameter(V(torch.mean(x, bndims)), requires_grad=False)
		self.params['bnvar1'] = nn.Parameter(V(torch.var(x, bndims)), requires_grad=False)
		bno1 = F.relu(F.batch_norm(x, self.params['bnmean1'], self.params['bnvar1'], self.params['bnw1'], self.params['bnb1']), inplace=True)
		convo1 = F.conv3d(bno1, self.params['convw1'], padding=1)
		self.params['bnmean2'] = nn.Parameter(V(torch.mean(convo1, bndims)), requires_grad=False)
		self.params['bnvar2'] = nn.Parameter(V(torch.var(convo1, bndims)), requires_grad=False)	
		bno2 = F.relu(F.batch_norm(x, self.params['bnmean2'], self.params['bnvar2'], self.params['bnw2'], self.params['bnb2']), inplace=True)
		if 'convb2' in self.params.keys():
			convo2 = F.conv3d(bno2, self.params['convw2'], self.params['convb2'], padding=1)
		else:
			convo2 = F.conv3d(bno2, self.params['convw2'], padding=1)
		poolo = F.max_pool3d(convo2, 2)
		feature_out = F.dropout(poolo, p=self.drop_rate, training=self.training)
		return feature_out

class doubconv_pool_pwise(nn.Module):
	def __init__(self, num_input_features, growth_rate, drop_rate, final=False, inidev=0.01, use_gpu=True):
		super(doubconv_pool_pwise, self).__init__()
		if use_gpu:
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')
		self.params = nn.ParameterDict()
		self.params['convw1'] = nn.Parameter(V(torch.randn((growth_rate, num_input_features, 3, 3, 3), device=device)*inidev, requires_grad=True))
		self.params['convw2'] = nn.Parameter(V(torch.randn((growth_rate, growth_rate, 3, 3, 3), device=device)*inidev, requires_grad=True))
		if final: self.params['convb2'] = nn.Parameter(V(torch.randn((growth_rate), device=device)*inidev, requires_grad=True))
		self.drop_rate = drop_rate
	def forward(self, x):
		rlo1 = F.relu(x, inplace=True)
		convo1 = F.conv3d(rlo1, self.params['convw1'], padding=1)
		rlo2 = F.relu(convo1, inplace=True)
		if 'convb2' in self.params.keys():
			convo2 = F.conv3d(rlo2, self.params['convw2'], self.params['convb2'], padding=1)
		else:
			convo2 = F.conv3d(rlo2, self.params['convw2'], padding=1)
		poolo = F.max_pool3d(convo2, 2)
		feature_out = F.dropout(poolo, p=self.drop_rate, training=self.training)
		return feature_out

class triconv_pool_pwise(nn.Module):
	def __init__(self, num_input_features, growth_rate, drop_rate, final=False, inidev=0.01, use_gpu=True):
		super(triconv_pool_pwise, self).__init__()
		if use_gpu:
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')
		self.params = nn.ParameterDict()
		self.params['conv1'] = nn.Parameter(V(torch.randn((growth_rate, num_input_features, 1, 1, 1), device=device)*inidev, requires_grad=True))
		self.params['conv2'] = nn.Parameter(V(torch.randn((growth_rate, growth_rate, 3, 3, 3), device=device)*inidev, requires_grad=True))
		self.params['conv3'] = nn.Parameter(V(torch.randn((growth_rate, growth_rate, 3, 3, 3), device=device)*inidev, requires_grad=True))
		if final: self.params['cbias3'] = nn.Parameter(V(torch.randn((growth_rate), device=device)*inidev, requires_grad=True))
		self.bn1 = nn.BatchNorm3d(num_input_features)
		self.bn2 = nn.BatchNorm3d(growth_rate)
		self.bn3 = nn.BatchNorm3d(growth_rate)
		self.drop_rate = drop_rate
	def forward(self, x):
		bno1 = F.relu(self.bn1(x), inplace=True)
		convo1 = F.conv3d(bno1, self.params['conv1'])
		bno2 = F.relu(self.bn2(convo1), inplace=True)
		convo2 = F.conv3d(bno2, self.params['conv2'], padding=1)
		bno3 = F.relu(self.bn3(convo2), inplace=True)
		if 'cbias3' in self.params.keys():
			convo3 = F.conv3d(bno3, self.params['conv3'], self.params['cbias3'], padding=1)
		else:
			convo3 = F.conv3d(bno3, self.params['conv3'], padding=1)
		poolo = F.max_pool3d(convo3, 2)
		feature_out = F.dropout(poolo, p=self.drop_rate, training=self.training)
		return feature_out
		
class densecropblock(nn.Module):
	def __init__(self, mode, transform, num_blocks, growth_rate, num_init_growth, num_fin_growth, drop_rate):
		super(densecropblock, self).__init__()
		self.mode = mode
		inichan = max(1, num_init_growth*growth_rate)
		self.pools = []
		if self.mode=='noconnection':
			for nbi in range(num_blocks):
				pool = transform(max(inichan,(nbi>0)*growth_rate), growth_rate, drop_rate)
				setattr(self, 'pool'+str(nbi), pool)
				poolings = [pool]
				self.pools.append(poolings)
			final_pooling = transform(growth_rate, num_fin_growth*growth_rate, drop_rate, final=True)
		elif self.mode=='singleline':
			for nbi in range(num_blocks):
				pool = transform(inichan+nbi*growth_rate, growth_rate, drop_rate)
				setattr(self, 'pool'+str(nbi), pool)
				poolings = [pool]
				self.pools.append(poolings)
			final_pooling = transform(inichan+num_blocks*growth_rate, num_fin_growth*growth_rate, drop_rate, final=True)
		elif self.mode=='fullline':
			for nbi in range(num_blocks):
				poolings = []
				for pi in range(2**nbi-1):
					pool = transform(growth_rate, growth_rate, drop_rate)
					setattr(self, 'pool'+str(nbi)+str(pi), pool)
					poolings.append(pool)
				pool = transform(inichan+(2**nbi-1)*growth_rate, growth_rate, drop_rate)
				setattr(self, 'pool'+str(nbi)+'c', pool)
				poolings.append(pool)
				self.pools.append(poolings)
			final_pooling = transform(inichan+(2**num_blocks-1)*growth_rate, num_fin_growth*growth_rate, drop_rate, final=True)
		elif self.mode=='fullconcat':
			for nbi in range(num_blocks):
				poolings = []
				for pi in range(2**nbi-1):
					pool = transform(inichan+pi*growth_rate, growth_rate, drop_rate)
					setattr(self, 'pool'+str(nbi)+str(pi), pool)
					poolings.append(pool)
				pool = transform(inichan+(2**nbi-1)*growth_rate, growth_rate, drop_rate)
				setattr(self, 'pool'+str(nbi)+'c', pool)
				poolings.append(pool)
				self.pools.append(poolings)
			final_pooling = transform(inichan+(2**num_blocks-1)*growth_rate, num_fin_growth*growth_rate, drop_rate, final=True)
		else:
			print('Failed to build model:{} network mode incorrect.' .format(self.model_name))
			exit()
		setattr(self, 'final_pool', final_pooling)
		self.pools.append(final_pooling)
		
	def forward(self, input):
		fms_pre = [input]
		for pi in range(len(self.pools)-1):
			poolings = self.pools[pi]
			fms = []
			if self.mode!='noconnection':
				for fmi in range(len(fms_pre)-1):
					if self.mode=='singleline':
						fms.append(crop(fms_pre[fmi]).contiguous())
					elif self.mode=='fullline':
						fms.append(crop(fms_pre[fmi]).contiguous())
						fms.append(poolings[int(fmi/2)].forward(fms_pre[fmi]))
					elif self.mode=='fullconcat':
						fms.append(crop(fms_pre[fmi]).contiguous())
						fms.append(poolings[fmi].forward(torch.cat(fms_pre[:fmi+1], dim=1).contiguous()))
				fms.append(crop(fms_pre[-1]).contiguous())
			fms.append(poolings[-1].forward(torch.cat(fms_pre, dim=1).contiguous()))
			fms_pre = fms
		fmfinal = self.pools[-1].forward(torch.cat(fms_pre, dim=1).contiguous())
		return fmfinal
