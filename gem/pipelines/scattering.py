from subprocess import call

import torch.nn.functional as F
from torch import nn, Tensor, load
from typing import Callable

from gem.pipelines.common import BasicAugmentation
from gem.architectures import rn, rn_cifar, wrn_cifar


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
    

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
    
    
class ScatteringModel(nn.Module):

    def __init__(self, base_model: nn.Module, arch: str, target_size: int, J: int, input_channels: int) -> None:

        super(ScatteringModel, self).__init__()
        self.base = base_model
        self.input_channels = input_channels
        self.J = J
        self.target_size = target_size
        self.nspace = int(self.target_size / (2 ** self.J))
        self.nfscat = int((1 + 8 * self.J + 8 * 8 * self.J * (self.J - 1) / 2))
        
        self.make_scat(arch, base_model.__class__)
        
        
    def make_scat(self, arch, arch_class):

        try:
            from kymatio import Scattering2D
        except ModuleNotFoundError:
            print('Installing kymatio ...')
            rc = call(['bash', 'setup_kymatio.sh'])
            from kymatio import Scattering2D

        if arch_class == rn.ResNet:
            
            self.base.conv1 = nn.Identity()
            self.base.bn1 = nn.Identity()
            self.base.relu = nn.Identity()
            self.base.maxpool = nn.Identity()

            self.base.layer1 = Scattering2D(J=self.J, shape=(self.target_size, self.target_size), frontend='torch')
            
            child = self.base.layer3[0].conv1 
            
            if child.in_channels == 128: # rn18/rn34 case
                
                l3_ichannels = child.out_channels
                new_attr = nn.Conv2d(child.out_channels, child.out_channels, child.kernel_size[0],
                                        stride=(1,1), padding=child.padding, bias=child.bias)
                self.base.layer3[0].conv1 = new_attr
                self.base.layer3[0].downsample = nn.Identity()
                
            else: # rn50/rn101 case
                
                l3_ichannels = self.base.layer3[0].conv1.in_channels
                self.base.layer3[0].conv2.stride = (1,1)
                self.base.layer3[0].downsample[0].stride = (1,1)
                
            self.base.layer2 = nn.Sequential(View(shape=(-1, self.input_channels * self.nfscat, self.nspace, self.nspace)),
                                             nn.BatchNorm2d(self.input_channels * self.nfscat, eps=1e-5, momentum=0.9, affine=False),
                                             nn.Conv2d(self.input_channels * self.nfscat, l3_ichannels, kernel_size=3, padding=1),
                                             nn.BatchNorm2d(l3_ichannels),
                                             nn.ReLU(inplace=True),
                                             )

                        
        elif arch_class == rn_cifar.ResNet:
            
            self.base.conv1 = Scattering2D(J=self.J, shape=(self.target_size, self.target_size), frontend='torch')
            self.base.bn1 = nn.Identity()
            
            l2_ichannels = self.base.layer2[0].conv1.in_channels
            self.base.layer1 = nn.Sequential(View(shape=(-1, self.input_channels * self.nfscat, self.nspace, self.nspace)),
                                             nn.BatchNorm2d(self.input_channels * self.nfscat, eps=1e-5, affine=False),
                                             nn.Conv2d(self.input_channels * self.nfscat, l2_ichannels, kernel_size=3, stride=1, padding=1, bias=False),
                                             nn.BatchNorm2d(l2_ichannels),
                                             nn.ReLU(inplace=True)
                                             )
            
            self.base.layer2[0].conv1.stride = (1,1)
            out_channels2 = self.base.layer2[0].conv1.out_channels
            self.base.layer2[0].shortcut = LambdaLayer(lambda x:
                                                       F.pad(x, (0, 0, 0, 0, out_channels2//4, out_channels2//4), "constant", 0))
                
            self.base.layer3[0].conv1.stride = (1,1)
            out_channels3 = self.base.layer3[0].conv1.out_channels
            self.base.layer3[0].shortcut = LambdaLayer(lambda x:
                                                       F.pad(x, (0, 0, 0, 0, out_channels3//4, out_channels3//4), "constant", 0))
            
                
        elif arch_class == wrn_cifar.WideResNet:
            
            self.base.conv1 = Scattering2D(J=self.J, shape=(self.target_size, self.target_size), frontend='torch')
            
            l2_ichannels = self.base.block2.layer[0].conv1.in_channels
            self.base.block1 = nn.Sequential(View(shape=(-1, self.input_channels * self.nfscat, self.nspace, self.nspace)),
                                             nn.BatchNorm2d(self.input_channels * self.nfscat, eps=1e-5, affine=False),
                                             nn.Conv2d(self.input_channels * self.nfscat, l2_ichannels, kernel_size=3, stride=1, padding=1, bias=False),
                                             nn.BatchNorm2d(l2_ichannels),
                                             nn.ReLU(inplace=True)
                                             )
            
            self.base.block2.layer[0].conv1.stride = (1,1)
            self.base.block2.layer[0].convShortcut.stride = (1,1)
            
            self.base.block3.layer[0].conv1.stride = (1,1)
            self.base.block3.layer[0].convShortcut.stride = (1,1)
            
            
        else:
            raise ValueError(f'Architecture {arch} is not supported by {self.__class__.__name__}.')

                            
    def forward(self, imgs: Tensor) -> Tensor:
        
        logits = self.base(imgs)
        return logits


class ScatteringClassifier(BasicAugmentation):
    """ Scattering classifier.
    
    See `BasicAugmentation` for a documentation of the available hyper-parameters.
    """

    def create_model(self, arch: str, num_classes: int, input_channels: int = 3) -> nn.Module:

        model = super(ScatteringClassifier, self).create_model(arch, num_classes=num_classes, input_channels=input_channels)
        
        if not(isinstance(self.hparams['target_size'], int)):
            raise(TypeError("The input spatial dimension target_size should be specified as an integer"))
            
        model = ScatteringModel(model, arch, self.hparams['target_size'], self.hparams['J'], input_channels)
        return model
    

    def load_weights(self, model: nn.Module, path: str) -> nn.Module:
        
        model_dict = model.state_dict()
        
        loaded_dict = load(path)
        # filter out keys with name 'tensor' since are from Scattering2D layer (not-trainable)
        filtered_dict = {k: v for k, v in loaded_dict.items() if not('tensor' in k)}
        model_dict.update(filtered_dict)
        
        model.load_state_dict(model_dict)
        return model
        
    
    def get_loss_function(self) -> Callable:

        return nn.CrossEntropyLoss(reduction='mean')
    

    @staticmethod
    def get_pipe_name():

        return 'scattering'


    @staticmethod
    def default_hparams() -> dict:

        return {
            **super(ScatteringClassifier, ScatteringClassifier).default_hparams(),
            'target_size' : 224,
            'J' : 3
        }
