import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck
from gem.utils import is_notebook

if is_notebook():
    from tqdm.notebook import tqdm, trange
else:
    from tqdm import tqdm, trange

from gem.architectures import rn, rn_cifar, wrn_cifar


class BasicBlockOFD(BasicBlock):
    
    def __init__(self, *args, **kwargs):
        super(BasicBlockOFD, self).__init__(*args, **kwargs)

    def forward(self, x):
        x = F.relu(x)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        # removed last relu because of distillation approach (moved at the beginning of the block)
        out += residual

        return out


class BottleneckOFD(Bottleneck):
    
    def __init__(self, *args, **kwargs):
        super(BottleneckOFD, self).__init__(*args, **kwargs)

    def forward(self, x):
        x = F.relu(x)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        # removed last relu because of distillation approach (moved at the beginning of the block)
        out += residual

        return out


# OFD architecture for rn ImageNet
class ResNetOFD(rn.ResNet):

    def __init__(self, block, *args, **kwargs):
        super(ResNetOFD, self).__init__(block, *args, **kwargs)
        self.expansion = block.expansion
        
    def get_bn_before_relu(self):
        
        if isinstance(self.layer1[0], BottleneckOFD):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
            bn4 = self.layer4[-1].bn3
        elif isinstance(self.layer1[0], BasicBlockOFD):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
            bn4 = self.layer4[-1].bn2
        else:
            print('ResNet unknown block error !!!')

        return [bn1, bn2, bn3, bn4]


    def get_channel_num(self):
        
        return [64 * self.expansion, 128 * self.expansion, 256 * self.expansion, 512 * self.expansion]


    def extract_feature(self, x, preReLU=False):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)

        x = self.avgpool(F.relu(feat4))
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        if not preReLU:
            feat1 = F.relu(feat1)
            feat2 = F.relu(feat2)
            feat3 = F.relu(feat3)
            feat4 = F.relu(feat4)

        return [feat1, feat2, feat3, feat4], out
    

# OFD architecture for wrn cifar

class WideResNetOFD(wrn_cifar.WideResNet):
            
    def __init__(self, *args, widen_factor=1, **kwargs):
        super(WideResNetOFD, self).__init__( *args, widen_factor=widen_factor, **kwargs)
        self.nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        
    def get_bn_before_relu(self):
        bn1 = self.block2.layer[0].bn1
        bn2 = self.block3.layer[0].bn1
        bn3 = self.bn1

        return [bn1, bn2, bn3]

    def get_channel_num(self):

        return self.nChannels[1:]

    def extract_feature(self, x, preReLU=False):
        out = self.conv1(x)
        feat1 = self.block1(out)
        feat2 = self.block2(feat1)
        feat3 = self.block3(feat2)
        out = self.relu(self.bn1(feat3))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        out = self.fc(out)

        if preReLU:
            feat1 = self.block2.layer[0].bn1(feat1)
            feat2 = self.block3.layer[0].bn1(feat2)
            feat3 = self.bn1(feat3)

        return [feat1, feat2, feat3], out
    
    
# OFD architecture for rn cifar

class CifarResNetOFD(rn_cifar.ResNet):
    
    def __init__(self, *args, **kwargs):
         super(CifarResNetOFD, self).__init__( *args, **kwargs)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.fc(x)

        return x
    
    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], BottleneckOFD):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
        elif isinstance(self.layer1[0], BasicBlockOFD):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
        else:
            print('ResNet unknown block error !!!')

        return [bn1, bn2, bn3]

    def get_channel_num(self):

        return [16, 32, 64]

    def extract_feature(self, x, preReLU=False):

        x = self.conv1(x)
        x = self.bn1(x)

        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)

        x = F.relu(feat3)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        out = self.fc(x)

        if not preReLU:
            feat1 = F.relu(feat1)
            feat2 = F.relu(feat2)
            feat3 = F.relu(feat3)

        return [feat1, feat2, feat3], out

