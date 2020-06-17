from collections import OrderedDict
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from .modules import BasicModule, dictionary_extend, dictionary_extract

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck_Paramwise(nn.Module):
	expansion = 4

	def __init__(self, inplanes=0, planes=0, stride=1, downsample=False, bare=False):
		super(Bottleneck_Paramwise, self).__init__()
		self.stride = stride
		if not bare:
			self.params = OrderedDict()
			self.buffers = OrderedDict()
			self.params['l0_conv_w'] = nn.Parameter(torch.Tensor(planes, inplanes, 1, 1))
			nn.init.kaiming_uniform_(self.params['l0_conv_w'], a=math.sqrt(5))
			self.params['l0_bn_w'] = nn.Parameter(torch.ones(planes))
			self.params['l0_bn_b'] = nn.Parameter(torch.zeros(planes))
			self.buffers['l0_bn_running_mean'] = torch.zeros(planes)
			self.buffers['l0_bn_running_var'] = torch.zeros(planes)
			self.params['l1_conv_w'] = nn.Parameter(torch.Tensor(planes, planes, 3, 3))
			nn.init.kaiming_uniform_(self.params['l1_conv_w'], a=math.sqrt(5))
			self.params['l1_bn_w'] = nn.Parameter(torch.ones(planes))
			self.params['l1_bn_b'] = nn.Parameter(torch.zeros(planes))
			self.buffers['l1_bn_running_mean'] = torch.zeros(planes)
			self.buffers['l1_bn_running_var'] = torch.zeros(planes)
			self.params['l2_conv_w'] = nn.Parameter(torch.Tensor(planes*self.expansion, planes, 1, 1))
			nn.init.kaiming_uniform_(self.params['l2_conv_w'], a=math.sqrt(5))
			self.params['l2_bn_w'] = nn.Parameter(torch.ones(planes*self.expansion))
			self.params['l2_bn_b'] = nn.Parameter(torch.zeros(planes*self.expansion))
			self.buffers['l2_bn_running_mean'] = torch.zeros(planes*self.expansion)
			self.buffers['l2_bn_running_var'] = torch.zeros(planes*self.expansion)
			if downsample:
				self.params['ld_conv_w'] = nn.Parameter(torch.Tensor(planes*self.expansion, inplanes, 1, 1))
				nn.init.kaiming_uniform_(self.params['ld_conv_w'], a=math.sqrt(5))
				self.params['ld_bn_w'] = nn.Parameter(torch.ones(planes*self.expansion))
				self.params['ld_bn_b'] = nn.Parameter(torch.zeros(planes*self.expansion))
				self.buffers['ld_bn_running_mean'] = torch.zeros(planes*self.expansion)
				self.buffers['ld_bn_running_var'] = torch.zeros(planes*self.expansion)

	def setchildren(self, params, buffers):
		self.params = OrderedDict()
		self.buffers = OrderedDict()
		for pkey in params.keys():
			skey = '_'.join(pkey.split('_')[-3:])
			self.params[skey] = params[pkey]
		for bkey in buffers.keys():
			skey = '_'.join(bkey.split('_')[-3:])
			self.buffers[skey] = buffers[pkey]

	def forward(self, x):
		out = F.conv2d(x, self.params['l0_conv_w'])
		out = F.batch_norm(out, self.buffers['l0_bn_running_mean'], self.buffers['l0_bn_running_var'], self.params['l0_bn_w'], self.params['l0_bn_b'], training=self.training)
		out = F.relu(out, inplace=True)

		out = F.conv2d(out, self.params['l1_conv_w'], stride=self.stride, padding=1)
		out = F.batch_norm(out, self.buffers['l1_bn_running_mean'], self.buffers['l1_bn_running_var'], self.params['l1_bn_w'], self.params['l1_bn_b'], training=self.training)
		out = F.relu(out, inplace=True)

		out = F.conv2d(out, self.params['l2_conv_w'])
		out = F.batch_norm(out, self.buffers['l2_bn_running_mean'], self.buffers['l2_bn_running_var'], self.params['l2_bn_w'], self.params['l2_bn_b'], training=self.training)

		if 'ld_conv_w' in self.params.keys():
			identity = F.conv2d(x, self.params['ld_conv_w'], stride=self.stride)
			identity = F.batch_norm(identity, self.buffers['ld_bn_running_mean'], self.buffers['ld_bn_running_var'], self.params['ld_bn_w'], self.params['ld_bn_b'], training=self.training)
		else:
			identity = x

		out += identity
		out = F.relu(out, inplace=True)

		return out

class ResNet_Paramwise(BasicModule):
	def __init__(self, block, layer_nums, num_classes=2, bare=False, zero_init_residual=False):
		super(ResNet_Paramwise, self).__init__()
		self.bare = bare
		self.inplanes = 64
		self.num_block_groups = 4
		self.block = block
		self.layer_nums = layer_nums
		self.strides = [1, 2, 2, 2]

		if not bare:
			planes = [64, 128, 256, 512]
			#self.params = nn.ParameterDict()
			self.register_parameter('conv_w', nn.Parameter(torch.Tensor(64, 3, 7, 7)))
			nn.init.kaiming_uniform_(self.conv_w, a=math.sqrt(5))
			self.register_parameter('bn_w', nn.Parameter(torch.ones(64)))
			self.register_parameter('bn_b', nn.Parameter(torch.zeros(64)))
			self.register_buffer('bn_running_mean', torch.zeros(64))
			self.register_buffer('bn_running_var', torch.zeros(64))
			self.layers_group = []
			for bg in range(self.num_block_groups):
				blocks, params, buffer = self._make_layer(block, layer_nums[bg], planes[bg], stride=self.strides[bg])
				self.layers_group.append(blocks)
				dictionary_extend(self._parameters, params, 'g%d'%(bg))
				for bkey in buffer.keys():
					#self.register_buffer('g%d_%s'%(bg, bkey), buffer[bkey])
					self._buffers['g%d_%s'%(bg, bkey)] = buffer[bkey]
			stdv = 1. / (512 * block.expansion)
			self.register_parameter('fc_w', nn.Parameter(torch.Tensor(num_classes, 512 * block.expansion)))
			self.register_parameter('fc_b', nn.Parameter(torch.Tensor(num_classes).uniform_(-stdv, stdv)))
			nn.init.kaiming_uniform_(self.fc_w, a=math.sqrt(5))

			for m in self.modules():
				if isinstance(m, nn.ParameterDict):
					for pkey in m.keys():
						paraminfo = pkey.split('_')
						layertype = paraminfo[-2]
						paramtype = paraminfo[-1]
						if layertype=='conv' and paramtype=='w':
							nn.init.kaiming_normal_(m[pkey], mode='fan_out', nonlinearity='relu')
						elif layertype=='bn':
							if paramtype=='w':
								nn.init.constant_(m[pkey], 1)
							elif paramtype=='b':
								nn.init.constant_(m[pkey], 0)
			if zero_init_residual:
				for l in range(self.num_block_groups):
					#4 layers for blocks
					blocks = self.layers_group[l]
					for layer_block in blocks:
						if isinstance(layer_block, Bottleneck_Paramwise):
							nn.init.constant_(layer_block.params['l2_bn_w'], 0)

	def cuda(self):
		super(ResNet_Paramwise, self).cuda()
		self.assign_block_variables()
		return self

	def eval(self):
		super(ResNet_Paramwise, self).eval()
		for blocks in self.layers_group:
			for layer_block in blocks:
				layer_block.eval()
		#self.assign_block_variables()
		return self

	def assign_block_variables(self, setparams=False):
		for bg in range(len(self.layers_group)):
			for b in range(self.layer_nums[bg]):
				key_prefix = 'g%d_b%d' %(bg, b)
				if setparams:
					bparameters = dictionary_extract(self._parameters, key_prefix)
					self.layers_group[bg][b].params = bparameters
				bbuffers = dictionary_extract(self._buffers, key_prefix)
				self.layers_group[bg][b].buffers = bbuffers

	def setchildren(self, parameters, buffers):
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

	def _make_layer(self, block, num_blocks, planes=0, stride=1):
		blocks = []
		if self.bare:
			iniblock = block(stride=stride, bare=self.bare)
			blocks.append(iniblock)
			for b in range(1, num_blocks):
				appendblock = block(bare=self.bare)
				blocks.append(appendblock)
			return blocks
		else:
			parameters = nn.ParameterDict()
			buffers = OrderedDict()
			iniblock = block(self.inplanes, planes, stride, downsample=True, bare=self.bare)
			dictionary_extend(parameters, iniblock.params, 'b0')
			dictionary_extend(buffers, iniblock.buffers, 'b0')
			blocks.append(iniblock)
			self.inplanes = planes * block.expansion
			for b in range(1, num_blocks):
				appendblock = block(self.inplanes, planes, bare=self.bare)
				dictionary_extend(parameters, appendblock.params, 'b%d'%(b))
				dictionary_extend(buffers, appendblock.buffers, 'b%d'%(b))
				blocks.append(appendblock)
			return blocks, parameters, buffers

	def forward(self, x, fin_feature=None):
		#the parameter fin_feature lies in {'rough', 'avg_pool'}
		x = F.conv2d(x, self.conv_w, stride=2, padding=3)
		x = F.batch_norm(x, self.bn_running_mean, self.bn_running_var, self.bn_w, self.bn_b, training=self.training)
		x = F.relu(x, inplace=True)
		x = F.max_pool2d(x, 3, 2, 1)

		for blocks in self.layers_group:
			for layer_block in blocks:
				x = layer_block.forward(x)

		p = F.adaptive_avg_pool2d(x, (1, 1))
		f = p.view(p.size(0), -1)
		out = F.linear(f, self.fc_w, self.fc_b)

		if fin_feature is None:
			return out
		else:
			if fin_feature=='rough':
				feature = x.view(x.size(0), -1)
			elif fin_feature=='avg_pool':
				feature = f
			else:
				feature = None
			return out, feature

class ResNet(BasicModule):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
	model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
	return model


def ParamResNet50(pretrained=False, output_size=2, bare=False, **kwargs):
	"""Constructs a ResNet-50 model.

	Args:
	pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet_Paramwise(Bottleneck_Paramwise, [3, 4, 6, 3], output_size, bare, **kwargs)
	if not bare and pretrained:
		#Alter the public state dictionary to fit to the state of our network.
		replace_dict = {'weight':'w', 'bias':'b', 'running_mean':'running_mean', 'running_var':'running_var'}
		strnumre = re.compile("([a-zA-Z]*)([0-9]*)")
		pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
		adapted_state_dict = OrderedDict()
		for pkey in pretrained_state_dict.keys():
			pnests = pkey.split('.')
			anests = []
			if len(pnests)==2:
				#initial or final layers
				anests.append(strnumre.match(pnests[0]).group(1))
			else:
				#layers of block groups
				anests.append('g' + str(int(strnumre.match(pnests[0]).group(2))-1))
				anests.append('b' + pnests[1])
				if 'downsample' in pnests[2]:
					layer_replace = {'0':'conv', '1':'bn'}
					anests.append('ld')
					anests.append(layer_replace[pnests[3]])
				else:
					layername_match = strnumre.match(pnests[2])
					anests.append('l' + str(int(layername_match.group(2))-1))
					anests.append(layername_match.group(1))
			if 'running' in pnests[-1]:
				anests.append(pnests[-1])
			else:
				anests.append(replace_dict[pnests[-1]])
					
			akey = '_'.join(anests)
			adapted_state_dict[akey] = pretrained_state_dict[pkey]
		#Change the output size of the last fully connected layer.
		select_indices = torch.randint(adapted_state_dict['fc_b'].size(0), (output_size,), dtype=torch.long)
		adapted_state_dict['fc_w'] = nn.Parameter(torch.index_select(adapted_state_dict['fc_w'], 0, select_indices))
		adapted_state_dict['fc_b'] = nn.Parameter(torch.index_select(adapted_state_dict['fc_b'], 0, select_indices))
		#Load the state dictionary to the current model.
		model.load_state_dict(adapted_state_dict)
	return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
