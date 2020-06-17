import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import time
from copy import copy, deepcopy
#from collections import OrderedDict
from .modules import BasicModule, SoftlySharedModule, crop, multi_crop_pooling, denseblock, simpleconv, conv_pool, doubconv_pool, doubconv_pool_pwise, densecropblock 
from .resnet import resnet50

'''
def crop(x):
	x_size = x.shape[2]
	size = int(x_size/2)
	begin = int(x_size/4)
	end = size + begin
	crop = x[:, :, begin:end, begin:end, begin:end]
	return crop

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

class simpleconv(BasicModule):
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
	def __init__(self, num_input_features, growth_rate, drop_rate):
		super(conv_pool, self).__init__()
		self.convpool = nn.Sequential(
		self.convpool = nn.Sequential(
			nn.BatchNorm3d(num_input_features),
			nn.ReLU(inplace = True),
			nn.Conv3d(num_input_features, growth_rate, kernel_size = 3, stride = 1, padding = 1),
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

class doubconv_pool_pwise(nn.Module):
	def __init__(self, num_input_features, growth_rate, drop_rate, final=False, inidev=0.01, use_gpu=True):
		super(doubconv_pool_pwise, self).__init__()
		if use_gpu:
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')
		self.params = nn.ParameterDict()
		self.params['conv1'] = nn.Parameter(V(torch.randn((growth_rate, num_input_features, 3, 3, 3), device=device)*inidev, requires_grad=True))
		self.params['conv2'] = nn.Parameter(V(torch.randn((growth_rate, growth_rate, 3, 3, 3), device=device)*inidev, requires_grad=True))
		if final: self.params['cbias2'] = nn.Parameter(V(torch.randn((growth_rate), device=device)*inidev, requires_grad=True))
		self.bn1 = nn.BatchNorm3d(num_input_features)
		self.bn2 = nn.BatchNorm3d(growth_rate)
		self.drop_rate = drop_rate
	def forward(self, x):
		bno1 = F.relu(self.bn1(x), inplace=True)
		convo1 = F.conv3d(bno1, self.params['conv1'], padding=1)
		bno2 = F.relu(self.bn2(convo1), inplace=True)
		if 'cbias2' in self.params.keys():
			convo2 = F.conv3d(bno2, self.params['conv2'], self.params['cbias2'], padding=1)
		else:
			convo2 = F.conv3d(bno2, self.params['conv2'], padding=1)
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
'''
			
class MulticropNet(BasicModule):
	def __init__(self, input_size=64, channels=[[16,16,16],[32,2]], poolings=[1,1,2], rrelu=(3,8)):
		super(MulticropNet, self).__init__()
		channels_layerwise = deepcopy(channels)
		poolings_conv = deepcopy(poolings)
		channels_layerwise[0].insert(0, 1)
		channels_layerwise[1].insert(0, int(input_size/2**sum(poolings))**3*channels[0][-1]*(poolings[-1]+1))
		poolings_conv.insert(0, 0)
		#self.rrelu = nn.RReLU(rrelu[0], rrelu[1])
		self.convs = []
		self.pools = []
		self.fcs = []
		for l in range(1, len(channels_layerwise[0])):
			convseq = nn.Sequential(
				nn.Conv3d(channels_layerwise[0][l-1]*(poolings_conv[l-1]+1), channels_layerwise[0][l], kernel_size=3, padding=1),
				nn.RReLU(1/float(rrelu[1]), 1/float(rrelu[0])))
			poolseq = multi_crop_pooling(poolings_conv[l])
			#self.convs.append(nn.Conv3d(channels_layerwise[0][l-1]*(poolings_conv[l-1]+1), channels_layerwise[0][l], kernel_size=3))
			self.convs.append(convseq)
			self.pools.append(poolseq)
			setattr(self, 'conv'+str(l), convseq)
			setattr(self, 'pool'+str(l), poolseq)
		for l in range(1, len(channels_layerwise[1])):
			if l < len(channels_layerwise[1])-1:
				fcseq = nn.Sequential(
					nn.Linear(channels_layerwise[1][l-1], channels_layerwise[1][l]),
					nn.RReLU(1/float(rrelu[1]), 1/float(rrelu[0])))
			else:
				fcseq = nn.Linear(channels_layerwise[1][l-1], channels_layerwise[1][l])
			#self.fcs.append(nn.Linear(channels_layerwise[1][l-1], channels_layerwise[1][l]))
			self.fcs.append(fcseq)
			setattr(self, 'fc'+str(l), fcseq)
	def forward(self, x, fin_feature=False):
		feature = x
		for c in range(len(self.convs)):
			feature_conved = self.convs[c].forward(feature)
			#feature_relued = self.rrelu.forward(feature_conved)
			feature = self.pools[c].forward(feature_conved)
		feature = feature.view(feature.size(0), -1)
		for f in range(len(self.fcs)-1):
			feature = self.fcs[f].forward(feature)
			#feature = self.rrelu.forward(feature_conved)
		out = self.fcs[-1].forward(feature)
		
		if fin_feature:
			return out, feature
		else:
			return out

class DenseNet_Iterative(BasicModule):
	def __init__(self, input_size=64, block_num=2, block_size=2, growth_rate=14, drop_rate=0.2, avgpool_size=0, output_size=2):
		super(DenseNet_Iterative, self).__init__()
		self.avgpool_size = avgpool_size
		self.iniconv = nn.Conv3d(1, growth_rate, kernel_size=3, stride=1, padding=1)
		self.denseblocks = []
		self.transitions = []
		for b in range(block_num):
			block = denseblock(block_size, growth_rate, drop_rate)
			self.denseblocks.append(block)
			setattr(self, 'denseblock'+str(b), block)
			transition = nn.MaxPool3d(2, 2)
			self.transitions.append(transition)
			setattr(self, 'transition'+str(b), transition)
		self.classifier = nn.Linear(growth_rate*int(input_size/(2**block_num*max(1,avgpool_size)))**3, output_size)
	def forward(self, x, fin_feature=False):
		fm = self.iniconv.forward(x)
		for b in range(len(self.denseblocks)):
			fm = self.denseblocks[b].forward(fm)
			fm = self.transitions[b].forward(fm)
		fm = F.relu(fm, inplace=False)
		if self.avgpool_size > 0:
			fm = F.avg_pool3d(fm, kernel_size=self.avgpool_size)
		fm = fm.view(fm.size(0), -1)
		out = self.classifier(fm)
		
		if fin_feature:
			return out, fm
		else:
			return out
		
class DensecropNet_Ini(BasicModule):
	def __init__(self,growth_rate=14,num_init_features=3, bn_size=5, drop_rate=0.2, avgpool_size=4, output_size=2):
		super(DensecropNet_Ini,self).__init__()
		self.avgpool_size = avgpool_size
		num_features = num_init_features
		self.conv = nn.Sequential(
		nn.BatchNorm3d(1),
		nn.ReLU(inplace = True),
		nn.Conv3d(1,num_init_features,kernel_size = 3,stride =1,padding = 1))
		self.pool1 = pooling(num_features,growth_rate,bn_size,drop_rate)
		self.pool2 = pooling(1+growth_rate, growth_rate, bn_size, drop_rate)
		self.pool3 = pooling(1+2*growth_rate, growth_rate, bn_size, drop_rate)
		self.pool4 = pooling(1+3*growth_rate, growth_rate, bn_size, drop_rate)
		self.classifier = nn.Linear(growth_rate, output_size)
	def forward(self,x):
		x = crop(x).contiguous()
		c0,p0 = crop(x),self.pool1.forward(self.conv(x))
		c1,c2 = crop(c0),crop(p0)
		p2 = self.pool2.forward(torch.cat([c0,p0],dim=1).contiguous())
		c3,c4,c5 = crop(c1),crop(c2),crop(p2)
		p5 = self.pool3.forward(torch.cat([c1,c2,p2],dim=1).contiguous())
		p6 = self.pool4.forward(torch.cat([c3,c4,c5,p5],dim=1).contiguous())
		out = F.relu(p6,inplace=True)	#problem
		out = F.avg_pool3d(out, kernel_size=self.avgpool_size).view(
						   out.size(0), -1)
		out = self.classifier(out)

		return out

class DensecropNet_Denser(BasicModule):
	def __init__(self,growth_rate=12,num_init_features=3, bn_size=5, drop_rate=0.2, avgpool_size=4,output_size=2):
		super(DensecropNet_Denser,self).__init__()
		self.avgpool_size = avgpool_size
		num_features = num_init_features
		self.conv = nn.Sequential(
			nn.BatchNorm3d(1),
			nn.ReLU(inplace=True),
			nn.Conv3d(1, num_init_features, kernel_size=3, stride=1, padding=1))
		self.pool0 = pooling(num_features,growth_rate,bn_size,drop_rate)
		self.pool1 = pooling(1, growth_rate, bn_size, drop_rate)
		self.pool2 = pooling(1+growth_rate, growth_rate, bn_size, drop_rate)
		self.pool3 = pooling(1, growth_rate, bn_size, drop_rate)
		self.pool4 = pooling(1+growth_rate, growth_rate, bn_size, drop_rate)
		self.pool5 = pooling(1+2*growth_rate, growth_rate, bn_size, drop_rate)
		self.pool6 = pooling(1+3*growth_rate,growth_rate,bn_size,drop_rate)
		self.pool7 = pooling(1+7*growth_rate,growth_rate,bn_size,drop_rate)

		self.classifier = nn.Linear(growth_rate, output_size)
	def forward(self,x):
		#x = crop(x).contiguous()
		c0,p0 = crop(x), self.pool0.forward(self.conv(x))
		c1,p1 = crop(c0), self.pool1.forward(c0.contiguous())
		c2 = crop(p0)
		p2 = self.pool2.forward(torch.cat([c0,p0],dim=1).contiguous())
		c3,p3 = crop(c1),self.pool3.forward(c1.contiguous())
		cat = torch.cat([c1, p1], dim=1).contiguous()
		c4,p4 = crop(p1),self.pool4.forward(cat)
		cat = torch.cat([cat, c2], dim=1).contiguous()
		c5,p5 = crop(c2),self.pool5.forward(cat)
		c6 = crop(p2)
		cat = torch.cat([cat, p2], dim=1).contiguous()
		p6 = self.pool6.forward(cat)
		p7 = self.pool7.forward(torch.cat([c3,p3,c4,p4,c5,p5,c6,p6],dim=1).contiguous())
		out = F.relu(p7,inplace=True)
		out = F.avg_pool3d( out, kernel_size=self.avgpool_size).view(
						   out.size(0), -1)
		out = self.classifier(out)

		return out

class DensecropNet_Iterative(BasicModule):
	def __init__(self, mode='singleline', input_size=64, transform=conv_pool, num_blocks=3, growth_rate=12, num_init_growth=1, num_fin_growth=1, avg_pool=False, drop_rate=0.2, output_size=2):
		#the parameter mode lies in {'singleline', 'fullline', 'fullconcat'}
		super(DensecropNet_Iterative, self).__init__()
		self.mode = mode
		#self.avgpool_size = avgpool_size
		if num_init_growth > 0:
			self.conv = nn.Sequential(
				nn.BatchNorm3d(1),
				nn.ReLU(inplace=True),
				nn.Conv3d(1, num_init_growth*growth_rate, kernel_size=3, stride=1, padding=1))
		inichan = max(1, num_init_growth*growth_rate)
		self.pools = []
		if self.mode=='singleline':
			for nbi in range(num_blocks):
				pool = transform(inichan+nbi*growth_rate, growth_rate, drop_rate)
				setattr(self, 'pool'+str(nbi), pool)
				poolings = [pool]
				self.pools.append(poolings)
			final_pooling = transform(inichan+num_blocks*growth_rate, num_fin_growth*growth_rate, drop_rate)
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
			final_pooling = transform(inichan+(2**num_blocks-1)*growth_rate, num_fin_growth*growth_rate, drop_rate)
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
			final_pooling = transform(inichan+(2**num_blocks-1)*growth_rate, num_fin_growth*growth_rate, drop_rate)
		else:
			print('Failed to build model:{} network mode incorrect.' .format(self.model_name))
			exit()
		setattr(self, 'final_pool', final_pooling)
		self.pools.append(final_pooling)
		if avg_pool:
			self.avg_pool = nn.AvgPool3d(int(input_size/2**(num_blocks+1)))
			#self.classifier0 = nn.Linear(num_fin_growth*growth_rate, num_fin_growth*growth_rate)
			self.classifier1 = nn.Linear(num_fin_growth*growth_rate, output_size)
		else:
			#self.classifier0 = nn.Linear(num_fin_growth*growth_rate*(input_size/2**(num_blocks+1))**3, num_fin_growth*growth_rate)
			self.classifier1 = nn.Linear(num_fin_growth*growth_rate*int(input_size/2**(num_blocks+1))**3, output_size)
	def forward(self, x, ini_crop=False, fin_feature=None):
		#the parameter fin_feature lies in {'rough', 'avg_pool'}
		if ini_crop:
			x = crop(x).contiguous()
		if hasattr(self, 'conv'):
			fm0 = self.conv(x)
		else:
			fm0 = x
		fms_pre = [fm0]
		for pi in range(len(self.pools)-1):
			poolings = self.pools[pi]
			fms = []
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
		if hasattr(self, 'avg_pool'):
			fmfinal = self.avg_pool.forward(fmfinal)
		ffc = F.relu(fmfinal, inplace=False).view(fmfinal.size(0), -1)
		#out = F.avg_pool3d(out, kernel_size=self.avgpool_size).view(out.size(0), -1)
		#ffc = self.classifier0(ffc)
		out = self.classifier1(ffc)

		if fin_feature is None:
			return out
		else:
			if fin_feature=='avg_pool': features = nn.AvgPool3d(fmfinal.shape[2:])(fmfinal).view(fmfinal.shape[0], -1)
			elif fin_feature=='rough': features = ffc
			else: features = None
			return out, features
		
class DensecropNet(BasicModule):
	def __init__(self, mode='singleline', input_size=64, transform=doubconv_pool, num_blocks=3, growth_rate=12, num_init_growth=1, num_fin_growth=1, avg_pool=False, drop_rate=0.2, output_size=2):
		#the parameter mode lies in {'noconnection', 'singleline', 'fullline', 'fullconcat'}
		super(DensecropNet, self).__init__()
		#self.mode = mode
		#self.avgpool_size = avgpool_size
		if num_init_growth > 0:
			#self.conv = nn.Sequential(
			#	nn.BatchNorm3d(1),
			#	nn.ReLU(inplace=True),
			#	nn.Conv3d(1, num_init_growth*growth_rate, kernel_size=3, stride=1, padding=1))
			self.conv = nn.Conv3d(1, num_init_growth*growth_rate, kernel_size=3, stride=1, padding=1)
		self.densecropblock = densecropblock(mode, transform, num_blocks, growth_rate, num_init_growth, num_fin_growth, drop_rate)
		if avg_pool:
			self.avg_pool = nn.AvgPool3d(int(input_size/2**(num_blocks+1)))
			#self.classifier0 = nn.Linear(num_fin_growth*growth_rate, num_fin_growth*growth_rate)
			self.classifier1 = nn.Linear(num_fin_growth*growth_rate, output_size)
		else:
			#self.classifier0 = nn.Linear(num_fin_growth*growth_rate*(input_size/2**(num_blocks+1))**3, num_fin_growth*growth_rate)
			self.classifier1 = nn.Linear(num_fin_growth*growth_rate*int(input_size/2**(num_blocks+1))**3, output_size)
	def forward(self, x, ini_crop=False, fin_feature=None):
		#the parameter fin_feature lies in {'rough', 'avg_pool'}
		if ini_crop:
			x = crop(x).contiguous()
		if hasattr(self, 'conv'):
			fm0 = self.conv(x)
		else:
			fm0 = x
		'''
		fms_pre = [fm0]
		for pi in range(len(self.pools)-1):
			poolings = self.pools[pi]
			fms = []
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
		'''
		fmfinal = self.densecropblock(fm0)
		if hasattr(self, 'avg_pool'):
			ffc = self.avg_pool.forward(fmfinal)
		else:
			ffc = fmfinal
		ffc = F.relu(ffc, inplace=False).view(ffc.size(0), -1)
		#out = F.avg_pool3d(out, kernel_size=self.avgpool_size).view(out.size(0), -1)
		#ffc = self.classifier0(ffc)
		out = self.classifier1(ffc)

		if fin_feature is None:
			return out
		else:
			if fin_feature=='avg_pool': features = nn.AvgPool3d(fmfinal.shape[2:])(fmfinal).view(fmfinal.shape[0], -1)
			elif fin_feature=='rough': features = fmfinal.view(fmfinal.size(0), -1)
			else: features = None
			return out, features
			
class DensecropNet_MultiTask(DensecropNet_Iterative):
	def __init__(self, mode='singleline', input_size=64, transform=simpleconv, num_blocks=3, growth_rate=12, num_init_growth=1, num_fin_growth=1, avg_pool=False, drop_rate=0.2, output_size=1):
		super(DensecropNet_MultiTask, self).__init__(mode, input_size, transform, num_blocks, growth_rate, num_init_growth, num_fin_growth, avg_pool, drop_rate, output_size)
		'''
		if num_init_growth > 0:
			self.conv = nn.Sequential(
				nn.BatchNorm3d(1),
				nn.ReLU(inplace=True),
				nn.Conv3d(1, num_init_growth*growth_rate, kernel_size=3, stride=1, padding=1))
		self.densecropblock = densecropblock(mode, transform, num_blocks, growth_rate, num_init_growth, num_fin_growth, drop_rate)
		if avg_pool:
			self.avg_pool = nn.AvgPool3d(input_size/2**(num_blocks+1))
			self.fcp1 = nn.Linear(num_fin_growth*growth_rate, output_size)
			self.fcp2 = nn.Linear(num_fin_growth*growth_rate*2, output_size)
		else:
			self.fcp1 = nn.Linear(num_fin_growth*growth_rate*(input_size/2**(num_blocks+1))**3, output_size)
			self.fcp2 = nn.Linear(num_fin_growth*growth_rate*(input_size/2**(num_blocks+1))**3*2, output_size)
		'''
		if avg_pool: self.relation = nn.Linear(num_fin_growth*growth_rate*2, output_size)
		else:
			self.relation = nn.Linear(num_fin_growth*growth_rate*int(input_size/2**(num_blocks+1))**3*2, output_size)
			#self.relation1 = nn.Linear(num_fin_growth*growth_rate*int(input_size/2**(num_blocks+1))**3*2, num_fin_growth*growth_rate)
			#self.relation2 = nn.Linear(num_fin_growth*growth_rate, output_size)
		
	def forward(self, x1, x2):
		out1, feature1 = super(DensecropNet_MultiTask, self).forward(x1, fin_feature='rough')
		out2, feature2 = super(DensecropNet_MultiTask, self).forward(x2, fin_feature='rough')
		featurecat = torch.cat((feature1, feature2), dim=1)
		outr = self.relation(featurecat)
		#outr1 = self.relation1(featurecat)
		#outr2 = self.relation2(outr1)
		return out1, out2, outr

class RegularlySharedNetCommon(SoftlySharedModule):
	def __init__(self, NetModule=MulticropNet, normps=(2,), **kwargs):
		super(RegularlySharedNetCommon, self).__init__(NetModule, **kwargs)
		self.normps = normps

	def norms(self, power):
		norm_diff = 0
		paramdict1 = self.net1.state_dict()
		paramdict2 = self.net2.state_dict()
		for p in range(len(paramdict1.keys())-2):
			pkey = paramdict1.keys()[p]
			paramdiff = paramdict1[pkey] - paramdict2[pkey]
			normd = torch.norm(paramdiff, self.normps[0])
			if power:
				if isinstance(power, int): normd = normd.pow(power)/paramdiff.numel()
				else: normd = normd.pow(self.normps[0])
			norm_diff += normd
		return norm_diff

class EnsembleNet(BasicModule):
	def __init__(self, NetModule=resnet50, pretrained=True, num_copies=3, output_size=2):
		super(EnsembleNet, self).__init__()
		self.nets = []
		for nc in range(num_copies):
			net = NetModule(pretrained=pretrained)
			#randomly downsample the parameters to fit to the real output size
			select_indices = torch.randint(net.fc.out_features, (output_size,), dtype=torch.long)
			net.fc.out_features = output_size
			net.fc.weight = nn.Parameter(torch.index_select(net.fc.weight, 0, select_indices))
			net.fc.bias = nn.Parameter(torch.index_select(net.fc.bias, 0, select_indices))
			self.nets.append(net)
			setattr(self, 'net'+str(nc+1), net)
		if num_copies>1:
			self.ffc = nn.Linear(num_copies*output_size, output_size)
	
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
			output = self.ffc(ensemble_output)
		if fin_feature is None:
			return output
		else:
			if 'ensemble_output' in dir():
				feature_out = ensemble_output
			else:
				feature_out = output
			return output, feature_out

class MultiViewNet(BasicModule):
	def __init__(self, NetModule, num_views, output_size=2, pretrained=True):
		super(MultiViewNet, self).__init__()

class BiasUndoingNet(BasicModule):
	def __init__(self, NetModule, num_tasks=2, normps=(2, 2, 2), **kwargs):
		super(BaseweightSharedNet, self).__init__()
		self.NetModule = NetModule
		self.normps = normps
		self.kwargs = kwargs
		self.stream = 1

		nets = []
		paramsets = []
		for t in range(num_tasks):
			nets.append(NetModule(**kwargs))
			paramsets.append(nn.ParameterDict())
			setattr(self, 'params'+str(t+1), paramsets[t])
		self.sparams = nn.ParameterDict()
		sparamsstat = OrderedDict()
		for pkey in nets[0].state_dict().keys():
			if pkey.split('.')[-1] in ('weight', 'bias'):
				for t in range(num_tasks):
					#self.paramsets[t][pkey] = nets[t][pkey]
					if pkey in sparamsstat.keys():
						sparamsstat[pkey].append(nets[t][pkey])
					else:
						sparamsstat[pkey] = [nets[t][pkey]]
				for pkey in sparamsstat.keys():
					self.sparams[pkey] = nn.Parameter(torch.stack(sparamsstat[pkey]).sum(dim=0))
					for t in range(num_tasks):
						self.paramsets[t][pkey] = nn.Parameter(nets[t][pkey] - self.sparams[pkey])

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

class SimpleNet(BasicModule):
	def __init__(self, input_size=64, num_blocks=3, growth_rate=14, drop_rate=0.2, output_size=2):
		super(SimpleNet, self).__init__()
		self.num_blocks = num_blocks
		num_channels = [max(1, nci*growth_rate) for nci in range(num_blocks)]
		convout = self.conv(x)
		convout_flatten = convout.view(convout.size(0), -1)
		fcout = self.fc(convout_flatten)
		if fin_feature is None:
			return fcout
		else:
			return fcout, convout

if __name__=='__main__':
	model = MulticropNet(input_size=64)
	para_dict = model.state_dict()
	total = 0
	for k,v in para_dict.items():
		size = v.size()
		length = len(size)
		num = 1
		for id in range(length):
			num = num*size[id]
		total += num
	print(total)
