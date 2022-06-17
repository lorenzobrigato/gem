import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision.models.resnet import BasicBlock


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class PaddingDownsampling(nn.Module):

    def __init__(self, planes):
        super(PaddingDownsampling, self).__init__()

        self.planes = planes

    def forward(self, x):
        return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.planes//4, self.planes//4), "constant", 0)


class ResNet(nn.Module):
    """ The CIFAR variants of ResNet.

    Reference:
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

    Parameters
    ----------
    block : callable
        Factory/constructor creating the block to be used.
    layers : list of int
        Number of blocks in each layer.
    num_classes : int
        Number of output neurons.
    input_channels : int
        Number of input channels.
    shortcut_downsampling : {'pad', 'conv'}
        Downsampling mode for the shortcut.
        'pad' will subsample the input using strided slicing and pad the channels with zeros.
        'conv' will use a strided convolution instead.
    """

    def __init__(self, block, layers, num_classes=10, input_channels=3, shortcut_downsampling='pad', groups=1):
        super(ResNet, self).__init__()

        self.in_planes = 16

        if shortcut_downsampling not in ('pad', 'conv'):
            raise ValueError(f'Invalid value for argument shortcut_downsampling: {shortcut_downsampling} (expected one of: pad, conv).')
        self.shortcut_downsampling = shortcut_downsampling

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1, groups=groups)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, groups=groups)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, groups=groups)
        self.fc = nn.Linear(64, num_classes)

        self.apply(_weights_init)


    def _make_layer(self, block, planes, num_blocks, stride, groups=1):

        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if self.shortcut_downsampling == 'pad':
                downsample = PaddingDownsampling(block.expansion * planes)
            elif self.shortcut_downsampling == 'conv':
                downsample = nn.Sequential(
                     nn.Conv2d(self.in_planes, block.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(block.expansion * planes)
                )

        layers = [block(self.in_planes, planes, stride=stride, downsample=downsample, groups=groups)]
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, stride=1, groups=groups))

        return nn.Sequential(*layers)


    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        
        out = self.fc(out)

        return out


    @staticmethod
    def get_arch_names():
        return ['rn20', 'rn32', 'rn44', 'rn56', 'rn110', 'rn1202']
    

    @staticmethod
    def get_config():

        CIFAR_RESNET_CONFIG = {20    : { 'block' : BasicBlock, 'layers' : [3, 3, 3] },
                               32    : { 'block' : BasicBlock, 'layers' : [5, 5, 5] },
                               44    : { 'block' : BasicBlock, 'layers' : [7, 7, 7] },
                               56    : { 'block' : BasicBlock, 'layers' : [9, 9, 9] },
                               101   : { 'block' : BasicBlock, 'layers' : [18, 18, 18] },
                               1202  : { 'block' : BasicBlock, 'layers' : [200, 200, 200] },
                               }

        return CIFAR_RESNET_CONFIG


    @classmethod
    def build_architecture(cls, arch: str, num_classes: int, input_channels: int, config: dict = None):

        _, depth = arch.split('rn')
        
        if not(config):
            cls_instance = cls(**cls.get_config()[int(depth)], num_classes=num_classes, input_channels=input_channels)
        else:
            cls_instance = cls(**config[int(depth)], num_classes=num_classes, input_channels=input_channels)

        return cls_instance
