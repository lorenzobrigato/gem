import torch
import torch.nn.functional as F
import math
import numpy as np
from torch import nn, Tensor
from typing import Callable

from gem.pipelines.common import BasicAugmentation
from gem.architectures import rn, rn_cifar, wrn_cifar

    
def dct_filters(k=3, groups=1, expand_dim=1, level=None, DC=True, l1_norm=True):
    if level is None:
        nf = k**2 - int(not DC) 
    else:
       	if level <= k:
            nf = level*(level+1)//2 - int(not DC) 
       	else:
       	    r = 2*k-1 - level
       	    nf = k**2 - r*(r+1)//2 - int(not DC)
    filter_bank = np.zeros((nf, k, k), dtype=np.float32)
    m = 0
    for i in range(k):
        for j in range(k):
            if (not DC and i == 0 and j == 0) or (not level is None and i + j >= level):
                continue
            for x in range(k):
                for y in range(k):
                    filter_bank[m, x, y] = math.cos((math.pi * (x + .5) * i) / k) * math.cos((math.pi * (y + .5) * j) / k)
            if l1_norm:
                filter_bank[m, :, :] /= np.sum(np.abs(filter_bank[m, :, :]))
            else:
                ai = 1.0 if i > 0 else 1.0 / math.sqrt(2.0)
                aj = 1.0 if j > 0 else 1.0 / math.sqrt(2.0)
                filter_bank[m, :, :] *= (2.0 / k) * ai * aj
            m += 1
    filter_bank = np.tile(np.expand_dims(filter_bank, axis=expand_dim), (groups, 1, 1, 1))
    return torch.FloatTensor(filter_bank)


class Harm2d(nn.Module):

    def __init__(self, ni, no, kernel_size, stride=1, padding=0, bias=True, dilation=1, use_bn=False, level=None, DC=True, groups=1):
        super(Harm2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.dct = nn.Parameter(dct_filters(k=kernel_size, groups=ni if use_bn else 1, expand_dim=1 if use_bn else 0, level=level, DC=DC), requires_grad=False)
        
        nf = self.dct.shape[0] // ni if use_bn else self.dct.shape[1]
        if use_bn:
            self.bn = nn.BatchNorm2d(ni*nf, affine=False)
            self.weight = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(no, ni // self.groups * nf, 1, 1), mode='fan_out', nonlinearity='relu'))
        else:
            self.weight = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(no, ni // self.groups, nf, 1, 1), mode='fan_out', nonlinearity='relu'))
        self.bias = nn.Parameter(nn.init.zeros_(torch.Tensor(no))) if bias else None

    def forward(self, x):
        if not hasattr(self, 'bn'):
            filt = torch.sum(self.weight * self.dct, dim=2)
            x = F.conv2d(x, filt, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
            return x
        else:
            x = F.conv2d(x, self.dct, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=x.size(1))
            x = self.bn(x)
            x = F.conv2d(x, self.weight, bias=self.bias, padding=0, groups=self.groups)
            return x


class HarmonicModel(nn.Module):

    def __init__(self, base_model: nn.Module) -> None:

        super(HarmonicModel, self).__init__()
        self.base = base_model
        self.make_harmbn_1stlayer(self.base)
        self.make_harm(self.base)

    
    def make_harmbn_1stlayer(self, model):
        
        keys = list(model._modules.keys())
        if isinstance(model._modules[keys[0]], nn.Conv2d):
            
            child = model._modules[keys[0]]
            new_attr = Harm2d(child.in_channels, child.out_channels, child.kernel_size[0],
                                        stride=child.stride, padding=child.padding, bias=False, use_bn=True)
            setattr(model, keys[0], new_attr)
        else:
            print("First layer of the model is not convolutional. Skipping harmonic+bn block, just leaving harmonic")
            
        
    def make_harm(self, model):

        for child_name, child in model.named_children():
            
            if isinstance(child, nn.Conv2d):
                
                if child.kernel_size != (1,1):
                    
                    new_attr = Harm2d(child.in_channels, child.out_channels, child.kernel_size[0],
                                        stride=child.stride, padding=child.padding, bias=False, use_bn=False)
                    setattr(model, child_name, new_attr)
            else:
                self.make_harm(child)
                    
                
    def forward(self, imgs: Tensor) -> Tensor:

        logits = self.base(imgs)
        return logits


class HarmonicClassifier(BasicAugmentation):
    """ Harmonic classifier.

    Paper: https://arxiv.org/abs/1905.00135
    
    See `BasicAugmentation` for a documentation of the available hyper-parameters.
    """

    def create_model(self, arch: str, num_classes: int, input_channels: int, config: dict = {}) -> nn.Module:

        arch_class = self.get_arch_class(arch)
        
        if (arch_class == rn.ResNet) or (arch_class == rn_cifar.ResNet) or (arch_class == wrn_cifar.WideResNet):

            model = super(HarmonicClassifier, self).create_model(arch, num_classes, input_channels)
            model = HarmonicModel(model)
        
        else:
            raise ValueError(f'Architecture {arch} is not supported by {self.__class__.__name__}.')

        return model
    
    
    def get_loss_function(self) -> Callable:

        return nn.CrossEntropyLoss(reduction='mean')


    @staticmethod
    def get_pipe_name():

        return 'harmonic'

