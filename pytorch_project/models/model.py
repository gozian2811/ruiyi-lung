import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import time
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
        if path is not None:
            self.load_state_dict(torch.load(path))
        else:
            self.load_state_dict(torch.load(self.opt.load_model_path))
    def save(self,name = None):
        '''
        save model with given name(optional)
        ----
        @name: optional, if not define,save with default format(model_name+timepoint).
        '''
        if name is None:
            prefix = 'checkpoints/'+self.model_name+'_'
            name = time.strftime(prefix+'%m%d_%H:%M:%S.pkl')
        torch.save(self.state_dict(),name)
        return name

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.conv0 = nn.Sequential(
		nn.BatchNorm3d(num_input_features),
		nn.ReLU(inplace=True),
		nn.Conv3d(num_input_features,bn_size*growth_rate,kernel_size=1,stride =1,bias = False),
		nn.BatchNorm3d(bn_size*growth_rate),
		nn.ReLU(inplace = True),
		nn.Conv3d(bn_size*growth_rate,num_input_features,kernel_size=3,stride=1,padding=1,bias = False))
        self.conv1 = nn.Sequential(
		nn.BatchNorm3d(num_input_features),
		nn.ReLU(inplace = True),
		nn.Conv3d(num_input_features,growth_rate,kernel_size = 3,stride =1,padding =1))
        self.drop_rate = drop_rate

    def forward(self, x):
        x_out = self.conv0(x)
        new_features = self.conv1(x+x_out)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.MaxPool3d(kernel_size=2, stride=2))


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class DenseNet(BasicModule):

    def __init__(self, growth_rate=12, block_config=(3, 4, 4,3), compression=0.8,
                 num_init_features=3, bn_size=5, drop_rate=0.5, avgpool_size=4,
                 num_classes=2):
        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = avgpool_size

        # First convolution
        self.features = nn.Sequential(
		nn.BatchNorm3d(1),
		nn.ReLU(inplace = True),
		nn.Conv3d(1,num_init_features,kernel_size = 3, stride=1,padding=1,bias=False),
		nn.MaxPool3d(2,2))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features
                                                            * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.crop(x).contiguous()
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool3d(out, kernel_size=self.avgpool_size).view(
                           features.size(0), -1)
        out = self.classifier(out)
        return out

    @staticmethod
    def crop(x):
        x_size = x.size()[2]
        size = int(x_size / 2)
        begin = int(x_size / 4)
        end = size + begin
        crop = x[:, :, begin:end, begin:end, begin:end]
        return crop
