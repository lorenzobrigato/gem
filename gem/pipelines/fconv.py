import torch
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, BasicBlock
from torch import nn, Tensor
from typing import Callable

from gem.pipelines.common import BasicAugmentation
from gem.architectures import rn, rn_cifar, wrn_cifar


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1,padding=2):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlockFConv(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlockFConv, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.zero_pad_I = nn.ZeroPad2d(2) # padding on Identity
        self.zero_pad_Id = nn.ZeroPad2d(3) # padding on Identity while downsampling
   
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(self.zero_pad_Id(x))
            out += identity
        else:
            out += self.zero_pad_I(identity)
        out = self.relu(out)

        return out


class BottleneckFConv(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BottleneckFConv, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)	
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.down_pad = nn.ZeroPad2d(1)
        self.normal_pad = nn.ZeroPad2d(1)

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
            x = self.down_pad(x)
            identity = self.downsample(x)
            out += identity

        else:
            out += self.normal_pad(identity)

        out = self.relu(out)

        return out
    
    
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlockFConvRN(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlockFConvRN, self).__init__()
        
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.zero_pad_I = nn.ZeroPad2d(2) # padding on Identity
        self.zero_pad_Id = nn.ZeroPad2d(3) # padding on Identity while spatial downsampling (channel upsampling)
        
        self.shortcut = nn.Sequential()
        self.downsample = False
        
        if stride != 1 or in_planes != planes:
            self.downsample = True
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample:
            x = self.zero_pad_Id(x)
            
        else:
            x = self.zero_pad_I(x)
            
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
    
class BasicBlockFConvWRN(nn.Module):
    
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        
        super(BasicBlockFConvWRN, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        self.zero_pad_I = nn.ZeroPad2d(2) # padding on Identity
        self.zero_pad_Id = nn.ZeroPad2d(3) if ((not self.equalInOut) and (stride[0] > 1)) else nn.ZeroPad2d(2) # handle specific case of
                                                                                                               # no spatial downsampling but channel upsampling

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(self.zero_pad_I(x) if self.equalInOut else self.convShortcut(self.zero_pad_Id(x)), out)


class FConvModel(nn.Module):

    def __init__(self, base_model: nn.Module) -> None:

        super(FConvModel, self).__init__()
        self.base = base_model
        self.make_fconv_stem(self.base)
        self.make_fconv(self.base)

    
    def make_fconv_stem(self, model):
        
        keys = list(model._modules.keys()) 
        
        if 'conv1' in keys: # for imagenet/cifar rn
            model.conv1.padding = tuple([ks - 1 for ks in model._modules['conv1'].kernel_size])
        
        if 'maxpool' in keys: # for imagenet rn 
            model.maxpool.kernel_size = 2
            model.maxpool.stride = 2
            model.maxpool.padding= 0

        
    def make_fconv(self, model):

        for child_name, child in model.named_children(): 
            
            if isinstance(child, Bottleneck): # rn50/101 case

                new_attr = BottleneckFConv(inplanes=child.conv1.in_channels,
                                           planes=child.conv1.out_channels,
                                           stride=child.stride,
                                           downsample=child.downsample)
                setattr(model, child_name, new_attr)
                
            elif isinstance(child, BasicBlock): # rn18/20 case

                new_attr = BasicBlockFConv(inplanes=child.conv1.in_channels,
                                           planes=child.conv1.out_channels,
                                           stride=child.stride,
                                           downsample=child.downsample)
                setattr(model, child_name, new_attr)

            elif isinstance(child, wrn_cifar.BasicBlock): # wrn cifar case
                
                new_attr = BasicBlockFConvWRN(in_planes=child.conv1.in_channels,
                                              out_planes=child.conv1.out_channels,
                                              stride=child.conv1.stride,
                                              dropRate=child.droprate)
                setattr(model, child_name, new_attr)
                
            else:
                self.make_fconv(child)
                    
                
    def forward(self, imgs: Tensor) -> Tensor:

        logits = self.base(imgs)
        return logits


class FConvClassifier(BasicAugmentation):
    """ Classifier with full convolution (F-Conv).

    Paper: https://arxiv.org/pdf/2003.07064.pdf
    
    See `BasicAugmentation` for a documentation of the available hyper-parameters.
    """

    def create_model(self, arch: str, num_classes: int, input_channels: int, config: dict = {}) -> nn.Module:

        arch_class = self.get_arch_class(arch)
        
        if (arch_class == rn.ResNet) or (arch_class == rn_cifar.ResNet) or (arch_class == wrn_cifar.WideResNet):

            model = super(FConvClassifier, self).create_model(arch, num_classes, input_channels)
            model = FConvModel(model)
        
        else:
            raise ValueError(f'Architecture {arch} is not supported by {self.__class__.__name__}.')

        return model
    
    
    def get_loss_function(self) -> Callable:

        return nn.CrossEntropyLoss(reduction='mean')


    @staticmethod
    def get_pipe_name():

        return 'fconv'

