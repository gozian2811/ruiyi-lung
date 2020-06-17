import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
from copy import deepcopy
from functools import partial
from collections import OrderedDict
from .modules import BasicModule, ConvModule, DenseCropModule, SoftlySharedModule, crop, doubconv_pool_pwise, doubconv_pool, concatenate_parameters, dictionary_extend, dictionary_extract
from .resnet import ParamResNet50

class ParamTest(BasicModule):
	def __init__(self, input_size=64, use_gpu=True, bare=False):
		super(ParamTest, self).__init__()
		if not bare:
			if use_gpu:
				device = torch.device('cuda')
			else:
				device = torch.device('cpu')
			params = OrderedDict()
			params['w_conv1'] = V(torch.randn((16, 1, 5, 5, 5), device=device)*0.01, requires_grad=True)
			params['w_conv2'] = V(torch.randn((32, 16, 3, 3, 3), device=device)*0.01, requires_grad=True)
			params['w_fc1'] = V(torch.randn((2, int(input_size/4)**3*32), device=device)*0.01, requires_grad=True)
			self.params = nn.ParameterDict({'w_conv1': nn.Parameter(params['w_conv1']), 'w_conv2': nn.Parameter(params['w_conv2']), 'w_fc1': nn.Parameter(params['w_fc1'])})

	def setparams(self, params):
		self.params = params
		#self.paramdict = nn.ParameterDict()
		#for key in params.keys():
		#	self.paramdict[key] = nn.Parameter(params[key])

	def forward(self, input, fin_feature=None):
		out_conv1 = F.conv3d(input, self.params['w_conv1'], padding=2)
		out_rl1 = F.relu(out_conv1)
		out_pool1 = F.max_pool3d(out_rl1, 2, 2)

		out_conv2 = F.conv3d(out_pool1, self.params['w_conv2'], padding=1)
		out_rl2 = F.relu(out_conv2)
		out_pool2 = F.max_pool3d(out_rl2, 2, 2)

		feature_flattened = out_pool2.view(out_pool2.size(0), -1)
		output = F.linear(feature_flattened, self.params['w_fc1'])

		if fin_feature is None:
			return output
		else:
			return output, feature_flattened

class BasicNet(ConvModule):
	def __init__(self, growth_rate=14, num_fin_growth=1, output_size=2, bare=False, **kwargs):
		super(BasicNet, self).__init__(growth_rate=growth_rate, num_fin_growth=num_fin_growth, bare=bare, **kwargs)
		if not bare:
			feature_size = growth_rate * num_fin_growth * int(self.end_feature_size)**3
			stdv = 1. / math.sqrt(feature_size)
			self.params['lr_fc_w'] = nn.Parameter(torch.Tensor(output_size, feature_size).uniform_(-stdv, stdv))
			self.params['lr_fc_b'] = nn.Parameter(torch.Tensor(output_size).uniform_(-stdv, stdv))
			
	def transfer(self, paramfix=False):
		feature_size = self.params['lr_fc_w'].size(1)
		stdv = 1. / math.sqrt(feature_size)
		self.params['lr_fc_w'].data.uniform_(-stdv, stdv)
		self.params['lr_fc_b'].data.uniform_(-stdv, stdv)
		if paramfix:
			self.paramfix(lastfix=False)
			#for pkey in self.params.keys():
			#	if pkey.split('_')[0] != 'lr':
			#		self.params[pkey].requires_grad = False

	def forward(self, input, fin_feature=None):
		fm = super(BasicNet, self).forward(input)
		feature = fm.view(fm.size(0), -1)
		output = F.linear(feature, self.params['lr_fc_w'], self.params['lr_fc_b'])
		if fin_feature is None:
			return output
		else:
			return output, feature

class ParamDenseCropNet(DenseCropModule):
	#def __init__(self, input_size=64, num_blocks=3, growth_rate=14, num_init_growth=1, num_fin_growth=1, final_fuse=False, drop_rate=0.2, output_size=2, bare=False):
	def __init__(self, growth_rate=14, num_fin_growth=1, output_size=2, bare=False, **kwargs):
		super(ParamDenseCropNet, self).__init__(growth_rate=growth_rate, num_fin_growth=num_fin_growth, bare=bare, **kwargs)
		if not bare:
			feature_size = growth_rate * num_fin_growth * self.end_feature_size**3
			stdv = 1. / math.sqrt(feature_size)
			self.params['lr_fc_w'] = nn.Parameter(torch.Tensor(output_size, feature_size))
			nn.init.kaiming_uniform_(self.params['lr_fc_w'], a=math.sqrt(5))
			self.params['lr_fc_b'] = nn.Parameter(torch.Tensor(output_size).uniform_(-stdv, stdv))
			
	def forward(self, input, fin_feature=None):
		fmfinal = super(ParamDenseCropNet, self).forward(input)
		feature = fmfinal.view(fmfinal.size(0), -1)
		output = F.linear(feature, self.params['lr_fc_w'], self.params['lr_fc_b'])
		if fin_feature is None:
			return output
		else:
			return output, feature

class BNNet(BasicModule):
	def __init__(self, input_size=64, num_blocks=3, growth_rate=14, num_fin_growth=1, drop_rate=0.2, output_size=2, bare=False):
		super(BNNet, self).__init__()
		self.num_blocks = num_blocks
		self.drop_rate = drop_rate
		if not bare:
			self.num_blocks = num_blocks
			num_channels = [max(1, nci*growth_rate) for nci in range(num_blocks)]
			#num_channels.insert(0, num_init_growth*growth_rate)
			num_channels.append(num_fin_growth*growth_rate)
			for nci in range(num_blocks):
				prefix = 'l' + str(nci)
				stdv = 1. / math.sqrt(num_channels[nci]*3**3)
				self.register_parameter(prefix+'_conv_w', nn.Parameter(torch.Tensor(num_channels[nci+1], num_channels[nci], 3, 3, 3).uniform_(-stdv, stdv)))
				self.register_parameter(prefix+'_bn_w', nn.Parameter(torch.Tensor(num_channels[nci+1]).uniform_(-stdv, stdv)))
				self.register_parameter(prefix+'_bn_b', nn.Parameter(torch.Tensor(num_channels[nci+1]).uniform_(-stdv, stdv)))
				#self.norms[prefix+'_running_mean'] = torch.zeros(num_channels[nci+1])
				#self.norms[prefix+'_running_var'] = torch.ones(num_channels[nci+1])
				self.register_buffer(prefix+'_running_mean', torch.zeros(num_channels[nci+1]))
				self.register_buffer(prefix+'_running_var', torch.ones(num_channels[nci+1]))

			feature_size = num_channels[-1] * int(input_size/2**num_blocks)**3
			stdv = 1. / math.sqrt(feature_size)
			self.register_buffer('lr_fc_w', nn.Parameter(torch.Tensor(output_size, feature_size).uniform_(-stdv, stdv)))
			self.register_buffer('lr_fc_b', nn.Parameter(torch.Tensor(output_size).uniform_(-stdv, stdv)))

	def reset_running_stats(self):
		for nci in range(self.num_blocks):
			prefix = 'l' + str(nci)
			getattr(self, prefix+'_running_mean').zero_()
			getattr(self, prefix+'_running_var').fill_(1)

	def setparams(self, parameters, buffers):
		self._parameters = parameters
		self._buffers = buffers
		#for key in buffers.keys():
		#	self.register_buffer(key, buffers[key])
	
	def forward(self, input, fin_feature=None):
		fm = input
		layer = 0
		for l in range(self.num_blocks):
			prefix = 'l' + str(l)
			fm = F.conv3d(fm, getattr(self, prefix+'_conv_w'), padding=1)
			fm = F.batch_norm(fm, getattr(self, prefix+'_running_mean'), getattr(self, prefix+'_running_var'), getattr(self, prefix+'_bn_w'), getattr(self, prefix+'_bn_b'), training=self.training)
			fm = F.max_pool3d(fm, 2, 2)
			fm = F.dropout(fm, p=self.drop_rate, training=self.training) 
			if l!=self.num_blocks-1: fm = F.relu(fm)
		feature = fm.view(fm.size(0), -1)
		output = F.linear(feature, self.lr_fc_w, self.lr_fc_b)
		if fin_feature is None:
			return output
		else:
			return output, feature

class EnsembleNet_Paramwise(BasicModule):
	def __init__(self, NetModule=ParamResNet50, output_size=2, pretrained=True, num_copies=3, bare=False):
		super(EnsembleNet_Paramwise, self).__init__()
		self.nets = []
		for nc in range(num_copies):
			net = NetModule(output_size=output_size, pretrained=pretrained, bare=bare)
			self.nets.append(net)
			setattr(self, 'net'+str(nc+1), net)
		#self.ffc = nn.Linear(num_copies*output_size, output_size)
		if num_copies>1 and not bare:
			stdv = 1. / math.sqrt(num_copies*output_size)
			self.register_parameter('fc_w', nn.Parameter(torch.Tensor(output_size, num_copies*output_size)))
			self.register_parameter('fc_b', nn.Parameter(torch.Tensor(output_size).uniform_(-stdv, stdv)))
			nn.init.kaiming_uniform_(self.fc_w, a=math.sqrt(5))

	def eval(self):
		super(EnsembleNet_Paramwise, self).eval()
		for net in self.nets:
			net.eval()
		return self

	def cuda(self):
		super(EnsembleNet_Paramwise, self).cuda()
		for net in self.nets:
			net.assign_block_variables()
		return self

	def setchildren(self, parameters, buffers):
		for n in range(len(self.nets)):
			net_prefix = 'n%d' %(n)
			net_parameters = dictionary_extract(parameters, net_prefix)
			net_buffers = dictionary_extract(buffers, net_prefix)
			self.nets[n]._parameters = net_parameters
			self.nets[n]._buffers = net_buffers
			self.nets[n].layers_group = []
			for bg in range(self.nets[n].num_block_groups):
				blocks = self.nets[n]._make_layer(self.nets[n].block, self.nets[n].layer_nums[bg], stride=self.nets[n].strides[bg])
				for b in range(len(blocks)):
					key_prefix = 'g%d_b%d' %(bg, b)
					bparams = dictionary_extract(net_parameters, key_prefix)
					bbuffers = dictionary_extract(net_buffers, key_prefix)
					blocks[b].params = bparams
					blocks[b].buffers = bbuffers
				self.nets[n].layers_group.append(blocks)
		self_parameters = dictionary_extract(parameters, 'n', prefix_include=False)
		self._parameters = self_parameters

	def transfer(self, paramfix=False):
		if hasattr(self, 'fc_w'):
			feature_size = self.fc_w.size(1)
			stdv = 1. / math.sqrt(feature_size)
			nn.init.kaiming_uniform_(self.fc_w, a=math.sqrt(5))
			self.fc_b.data.uniform_(-stdv, stdv)
		else:
			for net in self.nets:
				feature_size = net.fc_w.size(1)
				stdv = 1. / math.sqrt(feature_size)
				nn.init.kaiming_uniform_(net.fc_w, a=math.sqrt(5))
				net.fc_b.data.uniform_(-stdv, stdv)
		#if paramfix:
		#	self.paramfix(lastfix=False)

	def forward(self, input, fin_feature=None):
		if isinstance(input, torch.Tensor): input = (input,)
		if len(input) != len(self.nets):
			print('number of input and networks do not match.')
			return None
		outputs = []
		for i in range(len(input)):
			outputs.append(self.nets[i](input[i]))
		if len(outputs)==1:
			output = outputs[0]
		else:
			ensemble_output = torch.cat(outputs, dim=1)
			output = F.linear(ensemble_output, self.fc_w, self.fc_b)
			#output = self.ffc(ensemble_output)
		if fin_feature is None:
			return output
		else:
			if 'ensemble_output' in dir():
				feature_out = ensemble_output
			else:
				feature_out = output
			return output, feature_out

class MultiTaskNet(ConvModule):
	def __init__(self, growth_rate=14, num_fin_growth=1, output_sizes=(2, 2), **kwargs):
		super(MultiTaskNet, self).__init__(growth_rate=growth_rate, num_fin_growth=num_fin_growth, **kwargs)
		self.stream = 1
		feature_size = growth_rate * num_fin_growth * int(self.end_feature_size)**3
		stdv = 1. / math.sqrt(feature_size)
		for os in range(len(output_sizes)):
			self.params['lr'+str(os+1)+'_fc_w'] = nn.Parameter(torch.Tensor(output_sizes[os], feature_size).uniform_(-stdv, stdv))
			self.params['lr'+str(os+1)+'_fc_b'] = nn.Parameter(torch.Tensor(output_sizes[os]).uniform_(-stdv, stdv))

	def setstream(self, stream):
		self.stream = stream

	def forward(self, input, fin_feature=None):
		if type(input) in (tuple, list):
			result = []
			for i in range(len(input)):
				fm = super(MultiTaskNet, self).forward(input[i])
				feature = fm.view(fm.size(0), -1)
				output = F.linear(feature, self.params['lr'+str(i+1)+'_fc_w'], self.params['lr'+str(i+1)+'_fc_b'])
				if fin_feature is None:
					result.append(output)
				else:
					if fin_feature == 'avg_pool':
						fm = F.avg_pool3d(fm, fm.shape[-3:])
						feature = fm.view(fm.size(0), -1)
					result.append((output, feature))
			return result
		else:
			fm = super(MultiTaskNet, self).forward(input)
			feature = fm.view(fm.size(0), -1)
			output = F.linear(feature, self.params['lr'+str(self.stream)+'_fc_w'], self.params['lr'+str(self.stream)+'_fc_b'])
			if fin_feature is None:
				return output
			else:
				if fin_feature == 'avg_pool':
					fm = F.avg_pool3d(fm, fm.shape[-3:])
					feature = fm.view(fm.size(0), -1)
				return output, feature
		'''
		fm1 = super(MultiTaskNet, self).forward(input1)
		fm2 = super(MultiTaskNet, self).forward(input2)
		feature1 = fm1.view(fm1.size(0), -1)
		feature2 = fm2.view(fm1.size(0), -1)
		output1 = F.linear(feature1, self.params['lr1_fc_w'], self.params['lr1_fc_b'])
		output2 = F.linear(feature2, self.params['lr2_fc_w'], self.params['lr2_fc_b'])
		if fin_feature is None:
			return output1, output2
		else:
			return (output1, feature1), (output2, feature2)
		'''

class RegularlySharedNet(SoftlySharedModule):
	def __init__(self, NetModule=BasicNet, normps=(2, 2), **kwargs):
		super(RegularlySharedNet, self).__init__(NetModule, **kwargs)
		#super(RegularlySharedNet, self).__init__()
		self.normps = normps
	'''
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
			#self.net2.load_state_dict(self.net1.state_dict())
			#self.net2.transfer(paramfix=False)
		elif self.stream==2:
			self.net2.load(path)
			self.net2.paramfix()
			#self.net1.load_state_dict(self.net2.state_dict())
			#self.net1.transfer(paramfix=False)
		else: super(RegularlySharedNet, self).load(path)

	def setstream(self, stream):
		self.stream = stream

	def sharing_parameters(self):
		sparams = OrderedDict()
		for pkey in self.net1.params.keys():
			if pkey.split('_')[0] != 'lr':
				sparams[pkey] = (self.net1.params[pkey] + self.net2.params[pkey]) / 2.0
		return sparams
	'''

	def norms(self, mode='difference', power=2):
		def norms_diff(self, power):
			#normp1 = 0
			#normp2 = 0
			norm_diff = 0
			for pkey in self.net1.params.keys():
				#norm1 = torch.norm(self.net1.params[pkey], self.normps[0])
				#if self.normps[0]>1 and power: norm1 = norm1.pow(self.normps[0])
				#normp1 += norm1
				#norm2 = torch.norm(self.net2.params[pkey], self.normps[1])
				#if self.normps[1]>1 and power: norm2 = norm2.pow(self.normps[1])
				#normp2 += norm2
				if pkey.split('_')[0] != 'lr':
					paramdiff = self.net1.params[pkey] - self.net2.params[pkey]
					normd = torch.norm(paramdiff, self.normps[0])
					if power:
						if isinstance(power, int): normd = normd.pow(power)/paramdiff.numel()
						else: normd = normd.pow(self.normps[0])
					norm_diff += normd
			#return normp1, normp2, norm_diff
			return norm_diff
		def norms_mean(self, power):
			sparams = self.sharing_parameters()
			norm1 = 0
			for pkey in self.net1.params.keys():
				if pkey in sparams.keys():
					norm = torch.norm(self.net1.params[pkey] - sparams[pkey], self.normps[0])
				#else:
				#	norm = torch.norm(self.net1.params[pkey], self.normps[0])
					if self.normps[0]>1 and power: norm = norm.pow(self.normps[0])
					norm1 += norm
			norm2 = 0
			for pkey in self.net2.params.keys():
				if pkey in sparams.keys():
					norm = torch.norm(self.net2.params[pkey] - sparams[pkey], self.normps[1])
				#else:
				#	norm = torch.norm(self.net2.params[pkey], self.normps[1])
					if self.normps[1]>1 and power: norm = norm.pow(self.normps[1])
					norm2 += norm
			#snorm = 0
			#for pkey in sparams.keys():
			#	snorm += torch.norm(sparams[pkey], self.normps[2]).pow(self.normps[2])
			return norm1, norm2
		def norms21(self):
			#the l2,1 norm
			params1 = []
			for pkey in self.net1.params.keys():
				params1.append(self.net1.params[pkey].view(-1))
			paramvector1 = torch.cat(params1)
			paramvector1 = paramvector1.reshape(len(paramvector1), 1)
			params2 = []
			for pkey in self.net2.params.keys():
				params2.append(self.net2.params[pkey].view(-1))
			paramvector2 = torch.cat(params2)
			paramvector2 = paramvector2.reshape(len(paramvector2), 1)
			parammatrix = torch.cat((paramvector1, paramvector2), dim=1)
			norm = parammatrix.norm(p=2, dim=1).norm(p=1)
			return norm
		if mode=='difference':
			return norms_diff(self, power)
		elif mode=='mean':
			return norms_mean(self, power)
		elif mode=='l21':
			return norms21(self)

	'''
	def forward(self, input, fin_feature=None):
		if type(input)==tuple:
			return self.net1(input[0], fin_feature), self.net2(input[1], fin_feature)
		else:
			if self.stream==2: return self.net2(input, fin_feature)
			else: return self.net1(input, fin_feature)
	'''

class BaseweightSharedNet(BasicModule):
	def __init__(self, NetModule=ParamDenseCropNet, baselayers=0, normps=(2, 2, 2), **kwargs):
		super(BaseweightSharedNet, self).__init__()
		self.NetModule = NetModule
		self.normps = normps
		self.kwargs = kwargs
		self.stream = 1

		net1 = NetModule(**kwargs)
		net2 = NetModule(**kwargs)
		self.sparams = nn.ParameterDict()
		self.params1 = nn.ParameterDict()
		self.params2 = nn.ParameterDict()
		for pkey in net1.params.keys():
			if pkey.split('_')[0] != 'lr':
				self.sparams[pkey] = nn.Parameter((net1.params[pkey] + net2.params[pkey]) / 2.0)
				layerid = int(pkey.split('_')[0][1:])
				if layerid >= baselayers:
					self.params1[pkey] = nn.Parameter(net1.params[pkey] - self.sparams[pkey])
					self.params2[pkey] = nn.Parameter(net2.params[pkey] - self.sparams[pkey])
			else:
				self.params1[pkey] = net1.params[pkey]
				self.params2[pkey] = net2.params[pkey]

	def setstream(self, stream):
		self.stream = stream

	def paramfix(self, fixindices=None):
		if fixindices is None:
			fix_parameters(self.sparams)
			fix_parameters(self.params1)
			fix_parameters(self.params2)
		else:
			for find in fixindices:
				if find==0:
					fix_parameters(self.sparams)
				elif find==1:
					fix_parameters(self.params1)
				elif find==2:
					fix_parameters(self.params2)
				else:
					print("unknown parameter index")

	def norms(self):
		paramvector1 = concatenate_parameters(self.params1)
		if isinstance(paramvector1, torch.Tensor):
			paramvector1 = paramvector1.reshape(len(paramvector1), 1)
			tnorm1 = paramvector1.norm(self.normps[0])
			if self.normps[0]>1: tnorm1 = tnorm1.pow(2)
		else:
			tnorm1 = 0
		paramvector2 = concatenate_parameters(self.params2)
		if isinstance(paramvector2, torch.Tensor):
			paramvector2 = paramvector2.reshape(len(paramvector2), 1)
			tnorm2 = paramvector2.norm(self.normps[1])
			if self.normps[1]>1: tnorm2 = tnorm2.pow(2)
		else:
			tnorm2 = 0
		sparamvector = concatenate_parameters(self.sparams)
		if isinstance(sparamvector, torch.Tensor):
			sparamvector = sparamvector.reshape(len(sparamvector), 1)
			snorm = sparamvector.norm(self.normps[2])
			if self.normps[2]>1: snorm = snorm.pow(2)
		else:
			snorm = 0
		'''
		tnorm = 0
		for pkey in self.params1.keys():
			pnorm = torch.norm(self.params1[pkey], self.normps[0])
			if self.normps[0]>1: pnorm = pnorm.pow(2)
			tnorm += pnorm
		for pkey in self.params2.keys():
			pnorm = torch.norm(self.params2[pkey], self.normps[1])
			if self.normps[1]>1: pnorm = pnorm.pow(2)
			tnorm += pnorm
		snorm = 0
		for pkey in self.sparams.keys():
			pnorm = torch.norm(self.sparams[pkey], self.normps[2])
			if self.normps[2]>1: pnorm = pnorm.pow(2)
			snorm += pnorm
		'''
		return tnorm1, tnorm2, snorm

	def forward(self, input, fin_feature=None, base=False):
		if type(input)==tuple:
			paramdict1 = OrderedDict()
			paramdict2 = OrderedDict()
			'''
			for key in self.params1.keys():
				if key in self.sparams.keys():
					if base:
						paramdict1[key] = self.sparams[key]
					else:
						paramdict1[key] = self.sparams[key] + self.params1[key]
				else:
					paramdict1[key] = self.params1[key]
			for key in self.params2.keys():
				if key in self.sparams.keys():
					if base:
						paramdict2[key] = self.sparams[key]
					else:
						paramdict2[key] = self.sparams[key] + self.params2[key]
				else:
					paramdict2[key] = self.params2[key]
			'''
			for key in self.sparams.keys():
				if not base and key in self.params1.keys():
					paramdict1[key] = self.sparams[key] + self.params1[key]
				else:
					paramdict1[key] = self.sparams[key]
				if not base and key in self.params2.keys():
					paramdict2[key] = self.sparams[key] + self.params2[key]
				else:
					paramdict2[key] = self.sparams[key]
			paramdict1['lr_fc_w'] = self.params1['lr_fc_w']
			paramdict2['lr_fc_w'] = self.params2['lr_fc_w']
			paramdict1['lr_fc_b'] = self.params1['lr_fc_b']
			paramdict2['lr_fc_b'] = self.params2['lr_fc_b']
			netc1 = self.NetModule(**self.kwargs, bare=True)
			netc2 = self.NetModule(**self.kwargs, bare=True)
			netc1.setparams(paramdict1)
			netc2.setparams(paramdict2)
			netc1.training = self.training
			netc2.training = self.training
			return netc1(input[0], fin_feature), netc2(input[1], fin_feature)
		else:
			if self.stream==1:
				params = self.params1
			elif self.stream==2:
				params = self.params2
			else:
				return None
			paramdict = OrderedDict()
			for key in self.sparams.keys():
				if not base and key in params.keys():
					paramdict[key] = self.sparams[key] + params[key]
				else:
					paramdict[key] = self.sparams[key]
			paramdict['lr_fc_w'] = params['lr_fc_w']
			paramdict['lr_fc_b'] = params['lr_fc_b']
			'''
			for key in params.keys():
				if key in self.sparams.keys():
					if base:
						paramdict[key] = self.sparams[key]
					else:
						paramdict[key] = self.sparams[key] + params[key]
				else:
					paramdict[key] = params[key]
			'''
			netc = self.NetModule(**self.kwargs, bare=True)
			netc.setparams(paramdict)
			netc.training = self.training
			return netc(input, fin_feature)

class BaseweightSharedBNNet(BasicModule):
	def __init__(self, NetModule=EnsembleNet_Paramwise, startbranchlayer='conv', normps=(2, 2, 2), **kwargs):
		super(BaseweightSharedBNNet, self).__init__()
		self.NetModule = NetModule
		self.startbranchlayer = startbranchlayer
		self.normps = normps
		self.stream = 1

		self.kwargs1 = {}
		self.kwargs2 = {}
		for akey in kwargs.keys():
			'''
			if akey[-1]=='2':
				kwargs2[akey] = kwargs[akey]
			elif akey+'2' in kwargs.keys():
				kwargs1[akey] = kwargs[akey]
			else:
				kwargs1[akey] = kwargs[akey]
				kwargs2[akey] = kwargs[akey]
			'''
			if akey[-1]!='2':
				self.kwargs1[akey] = kwargs[akey]
			if akey[-1]=='2':
				self.kwargs2[akey[:-1]] = kwargs[akey]
			elif akey+'2' not in kwargs.keys():
				self.kwargs2[akey] = kwargs[akey]
		model1 = NetModule(**self.kwargs1)
		model2 = NetModule(**self.kwargs2)
		self.sparams = nn.ParameterDict()
		self.params1 = nn.ParameterDict()
		self.params2 = nn.ParameterDict()
		#pkeys1 = model1.state_dict().keys()
		#pkeys2 = model2.state_dict().keys()
		#pkeys = set(pkeys1).union(set(pkeys2))
		parambranch = False
		if hasattr(model1, 'nets'):
			for n in range(len(model1.nets)):
				prefix = 'n%d_'%(n)
				for pkey, _ in model1.nets[n].named_parameters():
					if self.startbranchlayer in pkey: parambranch = True
					layer_parts = pkey.split('_')
					if (len(layer_parts)==2 and layer_parts[-2]=='fc') or n>=len(model2.nets):
						self.params1[prefix+pkey] = model1.nets[n]._parameters[pkey]
						if n<len(model2.nets):
							self.params2[prefix+pkey] = model2.nets[n]._parameters[pkey]
					else:
						self.sparams[prefix+pkey] = nn.Parameter((model1.nets[n]._parameters[pkey] + model2.nets[n]._parameters[pkey]) / 2.0)
						if parambranch:
							self.params1[prefix+pkey] = nn.Parameter(model1.nets[n]._parameters[pkey] - self.sparams[prefix+pkey])
							self.params2[prefix+pkey] = nn.Parameter(model2.nets[n]._parameters[pkey] - self.sparams[prefix+pkey])
				dictionary_extend(self._buffers, model1.nets[n]._buffers, 'm1_n%d'%(n))
				if n<len(model2.nets):
					dictionary_extend(self._buffers, model2.nets[n]._buffers, 'm2_n%d'%(n))
			for pkey in model1._parameters.keys():
				self.params1[pkey] = getattr(model1, pkey)
			for pkey in model2._parameters.keys():
				self.params2[pkey] = getattr(model1, pkey)
		else:
			for pkey, _ in model1.named_parameters():
				frame_parts = pkey.split('.')
				layer_parts = frame_parts[-1].split('_')
				if len(layer_parts)==2 and layer_parts[-2]=='fc':
					self.params1[pkey] = model1._parameters[pkey]
					self.params2[pkey] = model2._parameters[pkey]
				else:
					self.sparams[pkey] = nn.Parameter((model1._parameters[pkey] + model2._parameters[pkey]) / 2.0)
					if parambranch:
						self.params1[pkey] = nn.Parameter(model1._parameters[pkey] - self.sparams[pkey])
						self.params2[pkey] = nn.Parameter(model2._parameters[pkey] - self.sparams[pkey])
			dictionary_extend(self._buffers, model1._buffers, 'm1')
			dictionary_extend(self._buffers, model2._buffers, 'm2')

	def setstream(self, stream):
		self.stream = stream

	def paramfix(self, fixindices=None):
		if fixindices is None:
			fix_parameters(self.sparams)
			fix_parameters(self.params1)
			fix_parameters(self.params2)
		else:
			for find in fixindices:
				if find==0:
					fix_parameters(self.sparams)
				elif find==1:
					fix_parameters(self.params1)
				elif find==2:
					fix_parameters(self.params2)
				else:
					print("unknown parameter index")

	def norms(self):
		paramvector1 = concatenate_parameters(self.params1)
		if isinstance(paramvector1, torch.Tensor):
			paramvector1 = paramvector1.reshape(len(paramvector1), 1)
			tnorm1 = paramvector1.norm(self.normps[0])
			if self.normps[0]>1: tnorm1 = tnorm1.pow(2)
		else:
			tnorm1 = 0
		paramvector2 = concatenate_parameters(self.params2)
		if isinstance(paramvector2, torch.Tensor):
			paramvector2 = paramvector2.reshape(len(paramvector2), 1)
			tnorm2 = paramvector2.norm(self.normps[1])
			if self.normps[1]>1: tnorm2 = tnorm2.pow(2)
		else:
			tnorm2 = 0
		sparamvector = concatenate_parameters(self.sparams)
		if isinstance(sparamvector, torch.Tensor):
			sparamvector = sparamvector.reshape(len(sparamvector), 1)
			snorm = sparamvector.norm(self.normps[2])
			if self.normps[2]>1: snorm = snorm.pow(2)
		else:
			snorm = 0
		return tnorm1, tnorm2, snorm

	def forward(self, input, fin_feature=None, base=False):
		if type(input)==tuple:
			paramdict1 = OrderedDict()
			paramdict2 = OrderedDict()
			'''
			for key in self.sparams.keys():
				if not base and key in self.params1.keys():
					paramdict1[key] = self.sparams[key] + self.params1[key]
				else:
					paramdict1[key] = self.sparams[key]
				if not base and key in self.params2.keys():
					paramdict2[key] = self.sparams[key] + self.params2[key]
				else:
					paramdict2[key] = self.sparams[key]
			paramdict1['fc_w'] = self.params1['fc_w']
			paramdict2['fc_w'] = self.params2['fc_w']
			paramdict1['fc_b'] = self.params1['fc_b']
			paramdict2['fc_b'] = self.params2['fc_b']
			'''
			keys1 = set(self.params1.keys())
			keys1 = keys1.union(set(self.sparams.keys()))
			for key in keys1:
				if key in self.sparams.keys():
					if not base and key in self.params1.keys(): paramdict1[key] = self.sparams[key] + self.params1[key]
					else: paramdict1[key] = self.sparams[key]
				else:
					paramdict1[key] = self.params1[key]
			keys2 = set(self.params2.keys())
			keys2 = keys2.union(set(self.sparams.keys()))
			for key in keys2:
				if key in self.sparams.keys():
					if not base and key in self.params2.keys(): paramdict2[key] = self.sparams[key] + self.params2[key]
					else: paramdict2[key] = self.sparams[key]
				else:
					paramdict2[key] = self.params2[key]
			modelc1 = self.NetModule(**self.kwargs1, bare=True)
			modelc2 = self.NetModule(**self.kwargs2, bare=True)
			modelc1.setchildren(paramdict1, dictionary_extract(self._buffers, 'm1'))
			modelc2.setchildren(paramdict2, dictionary_extract(self._buffers, 'm2'))
			if not self.training:
				modelc1 = modelc1.eval()
				modelc2 = modelc2.eval()
			return modelc1(input[0], fin_feature), modelc2(input[1], fin_feature)
		else:
			if self.stream==1:
				params = self.params1
				buffers = dictionary_extract(self._buffers, 'm1')
				kwargs = self.kwargs1
			elif self.stream==2:
				params = self.params2
				buffers = dictionary_extract(self._buffers, 'm2')
				kwargs = self.kwargs2
			else:
				return None
			paramdict = OrderedDict()
			keys = set(params.keys())
			keys = keys.union(set(self.sparams.keys()))
			for key in keys:
				if key in self.sparams.keys():
					if not base and key in params.keys(): paramdict[key] = self.sparams[key] + params[key]
					else: paramdict[key] = self.sparams[key]
				else:
					paramdict[key] = params[key]
			'''
			for key in self.sparams.keys():
				if not base and key in params.keys():
					paramdict[key] = self.sparams[key] + params[key]
				else:
					paramdict[key] = self.sparams[key]
			paramdict['fc_w'] = params['fc_w']
			paramdict['fc_b'] = params['fc_b']
			'''
			modelc = self.NetModule(**kwargs, bare=True)
			modelc.setchildren(paramdict, buffers)
			if not self.training:
				modelc = modelc.eval()
			return modelc(input, fin_feature)

class ParameterShare(BasicModule):
	def __init__(self, NetModule, sharing_init='little', **kwargs):
		#the parameter of sharing_init lies in {'halfly', 'little', 'increment'}
		super(ParameterShare, self).__init__()
		self.NetModule = NetModule
		self.kwargs = kwargs
		self.stream = 1

		self.net1 = NetModule(**kwargs)
		self.net2 = NetModule(**kwargs)
		self.net2.load_state_dict(self.net1.state_dict())	#ensure the two architectures have equal initial parameters
		self.spd1 = nn.ParameterDict()
		self.spd2 = nn.ParameterDict()
		for i in range(self.net1.num_blocks):
			skey = 'l' + str(i)
			if sharing_init=='halfly':
				sharing = 0
			elif sharing_init=='little':
				sharing = 0.1 * (i + 1)
			elif sharing_init=='increment':
				sharing = -math.log(2*self.net1.num_blocks/(i+0.1+self.net1.num_blocks)-1)
			else:
				print("sharing initialization error. unknown configuration:{}" .format(sharing_init))
			self.spd1[skey] = nn.Parameter(torch.Tensor([sharing]))
			self.spd2[skey] = nn.Parameter(torch.Tensor([-sharing]))
	def setstream1(self):
		self.stream = 1
	def setstream2(self):
		self.stream = 2
	def setstream(self, stream):
		self.stream = stream
	def basic_parameters(self):
		for param1 in self.net1.parameters():
			yield param1
		for param2 in self.net2.parameters():
			yield param2
	def sharing_parameters(self):
		for skey1, sparam1 in self.spd1.items():
			yield sparam1
		for skey2, sparam2 in self.spd2.items():
			yield sparam2

class WeaklySharedNet(ParameterShare):
	def __init__(self, NetModule=BasicNet, **kwargs):
		super(WeaklySharedNet, self).__init__(NetModule, **kwargs)

	def forward(self, input, fin_feature=None):
		'''
		paramdict1 = OrderedDict()
		paramdict2 = OrderedDict()
		for key in self.net1.params.keys():
			skey = key.split('_')[0]
			paramdict1[key] = self.net1.params[key]*self.spd1[skey] + self.net2.params[key]*(1-self.spd1[skey])
			paramdict2[key] = self.net1.params[key]*self.spd2[skey] + self.net2.params[key]*(1-self.spd2[skey])
		netc1 = self.NetModule(bare=True)
		netc2 = self.NetModule(bare=True)
		netc1.setparams(self.kwargs['num_blocks'], paramdict1)
		netc2.setparams(self.kwargs['num_blocks'], paramdict2)
		return netc1(input, fin_feature), netc2(input, fin_feature)
		'''
		if type(input)==tuple:
			paramdict1 = OrderedDict()
			paramdict2 = OrderedDict()
			for key in self.net1.params.keys():
				#The last layers of 'lr' are independent between different tasks and should not be shared.
				skey = key.split('_')[0]
				if skey in self.spd1.keys():
					paramdict1[key] = self.net1.params[key]*torch.sigmoid(self.spd1[skey]) + self.net2.params[key]*(1-torch.sigmoid(self.spd1[skey]))
				else:
					paramdict1[key] = self.net1.params[key]
				if skey in self.spd2.keys():
					paramdict2[key] = self.net1.params[key]*torch.sigmoid(self.spd2[skey]) + self.net2.params[key]*(1-torch.sigmoid(self.spd2[skey]))
				else:
					paramdict2[key] = self.net2.params[key]
			netc1 = self.NetModule(**self.kwargs, bare=True)
			netc2 = self.NetModule(**self.kwargs, bare=True)
			netc1.setparams(paramdict1)
			netc2.setparams(paramdict2)
			netc1.training = self.training
			netc2.training = self.training
			return netc1(input[0], fin_feature), netc2(input[1], fin_feature)
		else:
			if self.stream==1:
				spd = self.spd1
				net = self.net1
			elif self.stream==2:
				spd = self.spd2
				net = self.net2
			else:
				return None
			paramdict = OrderedDict()
			for key in self.net1.params.keys():
				skey = key.split('_')[0]
				if skey in spd.keys():
					paramdict[key] = self.net1.params[key]*torch.sigmoid(spd[skey]) + self.net2.params[key]*(1-torch.sigmoid(spd[skey]))
				else:
					paramdict[key] = net.params[key]
			netc = self.NetModule(**self.kwargs, bare=True)
			netc.setparams(paramdict)
			netc.training = self.training
			return netc(input, fin_feature)

class WeaklySharedBNNet(ParameterShare):
	def __init__(self, NetModule=BNNet, **kwargs):
		super(WeaklySharedBNNet, self).__init__(NetModule, **kwargs)

	def forward(self, input, fin_feature=None):
		if type(input)==tuple:
			paramdict1 = OrderedDict()
			paramdict2 = OrderedDict()
			for key in self.net1._parameters.keys():
				#The last layers of 'lr' are independent between different tasks and should not be shared.
				ksplit = key.split('_')
				skey = ksplit[0]
				if skey in self.spd1.keys():
					paramdict1[key] = getattr(self.net1, key)*torch.sigmoid(self.spd1[skey]) + getattr(self.net2, key)*(1-torch.sigmoid(self.spd1[skey]))
				else:
					paramdict1[key] = getattr(self.net1, key)
				if skey in self.spd2.keys():
					paramdict2[key] = getattr(self.net1, key)*torch.sigmoid(self.spd2[skey]) + getattr(self.net2, key)*(1-torch.sigmoid(self.spd2[skey]))
				else:
					paramdict2[key] = getattr(self.net2, key)
			netc1 = self.NetModule(**self.kwargs, bare=True)
			netc2 = self.NetModule(**self.kwargs, bare=True)
			netc1.setparams(paramdict1, self.net1._buffers)
			netc2.setparams(paramdict2, self.net2._buffers)
			return netc1(input[0], fin_feature), netc2(input[1], fin_feature)
		else:
			if self.stream==1:
				spd = self.spd1
				net = self.net1
			elif self.stream==2:
				spd = self.spd2
				net = self.net2
			else:
				return None
			paramdict = OrderedDict()
			for key in self.net1._parameters.keys():
				ksplit = key.split('_')
				skey = ksplit[0]
				if skey in spd.keys():
					paramdict[key] = getattr(self.net1, key)*torch.sigmoid(spd[skey]) + getattr(self.net2, key)*(1-torch.sigmoid(spd[skey]))
				else:
					paramdict[key] = getattr(net, key)
			netc = self.NetModule(**self.kwargs, bare=True)
			netc.setparams(paramdict, net._buffers)
			return netc(input, fin_feature)

class WeaklySharedTest(BasicModule):
	def __init__(self, NetModule=ParamTest, input_size=64, use_gpu=True):
		super(WeaklySharedNet, self).__init__()
		self.NetModule = NetModule
		self.input_size = input_size
		if use_gpu:
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')
		self.net1 = NetModule(input_size=input_size)
		self.net2 = NetModule(input_size=input_size)
		self.sparamdict1 = nn.ParameterDict()
		self.sparamdict2 = nn.ParameterDict()
		for key in self.net1.params.keys():
			self.sparamdict1[key] = nn.Parameter(V(torch.rand(1, device=device), requires_grad=True))
			self.sparamdict2[key] = nn.Parameter(V(torch.rand(1, device=device), requires_grad=True))
		
	def forward(self, input, fin_feature=None):
		paramdict1 = OrderedDict()
		#paramdict2 = OrderedDict()
		for key in self.net1.params.keys():
			paramdict1[key] = self.net1.params[key]*self.sparamdict1[key] + self.net2.params[key]*self.sparamdict2[key]
			#paramdict2[key] = self.net1.params[key]*V(torch.randn(1, device=device), requires_grad=True) + self.net2.params[key]*V(torch.randn(1, device=device), requires_grad=True)
		netc1 = self.NetModule(input_size=self.input_size, bare=True)
		#netc2 = self.NetModule(input_size=self.input_size, bare=True)
		netc1.setparams(paramdict1)
		#netc2.setparams(paramdict2)
		return netc1(input, fin_feature)
