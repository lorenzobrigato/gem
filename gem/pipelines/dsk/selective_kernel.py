import torch
from torch import nn, Tensor
from typing import Callable, Optional

from gem.pipelines.dsk.blurpool import BlurPool


def blur_downsampling(downsample: nn.Module) -> nn.Module:
    """ Replaces strided convolution on a shortcut connection with anti-aliased pooling. """

    if isinstance(downsample, nn.Conv2d) and (downsample.stride[0] > 1):
        stride = downsample.stride[0]
        downsample.stride = (1, 1)
        return nn.Sequential(
            downsample,
            BlurPool(downsample.out_channels, filt_size=3, stride=stride)
        )

    elif isinstance(downsample, nn.Sequential):
        for i in range(len(downsample)):
            if (isinstance(downsample[i], nn.Conv2d) and (downsample[i].stride[0] > 1)) or isinstance(downsample[i], nn.Sequential):
                downsample[i] = blur_downsampling(downsample[i])
    
    return downsample



class SKBlock(nn.Module):
    """ Selective Kernel Block
    
    Parameters
    ----------
    features: int
        Input channel dimensionality.
    M: int, default: 2
        The number of branches.
    G: int, default: 32
        Number of convolution groups.
    r: int, default: 16
        The ratio for compute d, the length of z.
    stride: int, default: 1
        Stride.
    L: int, default: 32
        The minimum dim of the vector z in paper, default 32.
    norm_layer: callable, default: nn.BatchNorm2d
        Constructor for normalization layer to be used (e.g., nn.BatchNorm2d).
    """

    def __init__(
        self,
        in_features: int,
        features: int,
        M: int = 2,
        G: int = 32,
        r: int = 16,
        stride: int = 1,
        L: int = 32,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d
    ) -> None:
        super(SKBlock, self).__init__()

        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            modules = [nn.Conv2d(in_features, features, kernel_size=3, stride=1, padding=1+i, dilation=1+i, groups=G, bias=False)]
            if stride > 1:
                modules.append(BlurPool(features, filt_size=3, stride=stride))
            modules.append(norm_layer(features))
            modules.append(nn.ReLU(inplace=False))
            self.convs.append(nn.Sequential(*modules))
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
                                norm_layer(d),
                                nn.ReLU(inplace=False))
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                 nn.Conv2d(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x: Tensor) -> Tensor:
        
        batch_size = x.shape[0]
        
        feats = [conv(x) for conv in self.convs]
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])
        
        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(feats*attention_vectors, dim=1)
        
        return feats_V



class SKBottleneck(nn.Module):
    """ Bottleneck with Selective Kernel Block

    Paper: https://arxiv.org/abs/1903.06586
    
    Parameters
    ----------
    in_features: int
        Input channel dimensionality.
    mid_features: int
        Bottleneck channel dimensionality.
        Will be multiplied with the number of branches.
    stride: int
        Stride.
    downsample: nn.Module, optional
        Downsampling module applied to the residual branch.
    groups: int
        Number of convolution groups.
    base_width: int
        Ignored.
    dilation: int
        Ignored.
    norm_layer: callable
        Constructor for normalization layer to be used (e.g., nn.BatchNorm2d).
    branches: int
        Number of branches.
    r: int
        The ratio for compute d, the length of z.
    L: int
        The minimum dim of the vector z in paper.
    """

    expansion: int = 4

    def __init__(
        self,
        in_features: int,
        mid_features: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 32,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        branches: int = 2,
        r: int = 16,
        L: int = 32
    ) -> None:
        super(SKBottleneck, self).__init__()

        out_features = mid_features * self.expansion
        mid_features *= branches
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1, bias=False),
            norm_layer(mid_features),
            nn.ReLU(inplace=True)
        )
        
        self.conv2_sk = SKBlock(mid_features, mid_features, M=branches, G=groups, r=r, stride=stride, L=L, norm_layer=norm_layer)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_features, out_features, 1, stride=1, bias=False),
            norm_layer(out_features)
        )

        self.downsample = nn.Sequential() if downsample is None else downsample

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x: Tensor) -> Tensor:
        
        out = self.conv1(x)
        out = self.conv2_sk(out)
        out = self.conv3(out)
        return self.relu(out + self.downsample(x))



class SKBasicBlock(nn.Module):
    """ Basic residual block for smaller ResNets with Selective Kernel Block
    
    Parameters
    ----------
    in_features: int
        Input channel dimensionality.
    out_features: int
        Output channel dimensionality.
    stride: int
        Stride.
    downsample: nn.Module, optional
        Downsampling module applied to the residual branch.
    groups: int
        Number of convolution groups.
    base_width: int
        Ignored.
    dilation: int
        Ignored.
    norm_layer: callable
        Constructor for normalization layer to be used (e.g., nn.BatchNorm2d).
    branches: int
        Number of branches.
    r: int
        The ratio for compute d, the length of z.
    L: int
        The minimum dim of the vector z in paper.
    """
    expansion = 1

    def __init__(
        self,
        in_features: int,
        out_features: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 32,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        branches: int = 2,
        r: int = 16,
        L: int = 32
    ) -> None:
        super(SKBasicBlock, self).__init__()

        self.conv1 = SKBlock(in_features, out_features, M=branches, G=groups, r=r, stride=stride, L=L, norm_layer=norm_layer)
        self.bn1 = norm_layer(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SKBlock(out_features, out_features, M=branches, G=groups, r=r, L=L, norm_layer=norm_layer)
        self.bn2 = norm_layer(out_features)
        self.downsample = nn.Sequential() if downsample is None else downsample
        self.stride = stride


    def forward(self, x):

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(out + x)



class SKWideBlock(nn.Module):
    """ Sequence of two Selective Kernel Blocks for use in Wide ResNets.
    
    Parameters
    ----------
    in_features: int
        Input channel dimensionality.
    out_features: int
        Output channel dimensionality.
    stride: int
        Stride.
    dropRate: float, default: 0
        Dropout rate.
    groups: int
        Number of convolution groups.
    norm_layer: callable
        Constructor for normalization layer to be used (e.g., nn.BatchNorm2d).
    branches: int
        Number of branches.
    r: int
        The ratio for compute d, the length of z.
    L: int
        The minimum dim of the vector z in paper.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        stride: int = 1,
        dropRate: float = 0.0,
        groups: int = 8,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        branches: int = 2,
        r: int = 16,
        L: int = 32
    ) -> None:
        super(SKWideBlock, self).__init__()

        self.bn1 = norm_layer(in_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = SKBlock(in_features, out_features, M=branches, G=groups, r=r, L=L, norm_layer=norm_layer, stride=stride)
        self.bn2 = norm_layer(out_features)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = SKBlock(out_features, out_features, M=branches, G=groups, r=r, L=L, norm_layer=norm_layer, stride=1)
        self.droprate = dropRate

    def forward(self, x):
        out = self.relu2(self.bn2(self.conv1(self.relu1(self.bn1(x)))))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return out



class DSKBottleneck(nn.Module):
    """ Dual Selective Kernel Block

    Paper: https://link.springer.com/chapter/10.1007/978-3-030-66096-3_35
    
    This block simply consists of two SKBottleneck blocks and selects the output
    of one of them at random during training, averaging outputs during inference.

    Parameters are the same as for SKBottleneck.
    """

    expansion: int = 4

    def __init__(self, in_features: int, mid_features: int, stride: int = 1, downsample: Optional[nn.Module] = None, *args, **kwargs) -> None:
        super(DSKBottleneck, self).__init__()

        self.downsample = nn.Sequential() if downsample is None else blur_downsampling(downsample)

        # We disable the residual connection on the individual SKBottleneck blocks by setting downsample
        # to constant 0, because the residual is added here in the dual block.
        self.branch1 = SKBottleneck(in_features, mid_features, stride, lambda x: 0, *args, **kwargs)
        self.branch2 = SKBottleneck(in_features, mid_features, stride, lambda x: 0, *args, **kwargs)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x: Tensor) -> Tensor:

        weight = torch.randint(0, 2, ()) if self.training else 0.5
        out1 = weight * self.branch1(x)
        out2 = (1 - weight) * self.branch2(x)
        return self.relu(out1 + out2 + self.downsample(x))



class DSKBasicBlock(nn.Module):
    """ Basic Dual Selective Kernel Block for smaller ResNets
    
    This block simply consists of two SKBasicBlock blocks and selects the output
    of one of them at random during training, averaging outputs during inference.

    Parameters are the same as for SKBasicBlock.
    """

    expansion: int = 1

    def __init__(self, in_features: int, out_features: int, stride: int = 1, downsample: Optional[nn.Module] = None, *args, **kwargs) -> None:
        super(DSKBasicBlock, self).__init__()

        self.downsample = nn.Sequential() if downsample is None else blur_downsampling(downsample)

        # We disable the residual connection on the individual SKBasicBlock blocks by setting downsample
        # to constant 0, because the residual is added here in the dual block.
        self.branch1 = SKBasicBlock(in_features, out_features, stride, lambda x: 0, *args, **kwargs)
        self.branch2 = SKBasicBlock(in_features, out_features, stride, lambda x: 0, *args, **kwargs)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x: Tensor) -> Tensor:

        weight = torch.randint(0, 2, ()) if self.training else 0.5
        out1 = weight * self.branch1(x)
        out2 = (1 - weight) * self.branch2(x)
        return self.relu(out1 + out2 + self.downsample(x))



class DSKWideBlock(nn.Module):
    """ Wide Dual Selective Kernel Block for Wide ResNets
    
    This block simply consists of two SKWideBlock blocks and a shortcut connection
    and selects the output of one of the two SK blocks at random during training,
    averaging outputs during inference.

    Parameters are the same as for SKWideBlock.
    """

    def __init__(self, in_features: int, out_features: int, stride: int = 1, *args, **kwargs) -> None:
        super(DSKWideBlock, self).__init__()

        if in_features == out_features and stride == 1:
            self.downsample = nn.Sequential()
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=1, stride=1, padding=0, bias=False),
                BlurPool(out_features, filt_size=3, stride=stride)
            )

        self.branch1 = SKWideBlock(in_features, out_features, stride, *args, **kwargs)
        self.branch2 = SKWideBlock(in_features, out_features, stride, *args, **kwargs)


    def forward(self, x: Tensor) -> Tensor:

        weight = torch.randint(0, 2, ()) if self.training else 0.5
        out1 = weight * self.branch1(x)
        out2 = (1 - weight) * self.branch2(x)
        return out1 + out2 + self.downsample(x)
