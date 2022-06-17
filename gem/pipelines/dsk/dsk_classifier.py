import torch
import torchvision
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Tuple, Callable

from gem.pipelines.dsk.selective_kernel import DSKBottleneck, DSKBasicBlock, DSKWideBlock
from gem.pipelines.dsk.center_loss import CenterLoss, SGDWithCenterLoss
from gem.pipelines.common import BasicAugmentation
from gem.architectures import rn, rn_cifar, wrn_cifar


class FeatureModelWrapper(nn.Module):

    def __init__(self, base_model: nn.Module, num_classes: int, feat_dim: int) -> None:

        super(FeatureModelWrapper, self).__init__()
        self.base = base_model
        self.base.fc = nn.Linear(self.base.fc.in_features, feat_dim)
        self.fc = nn.Linear(feat_dim, num_classes)
    

    def forward(self, input: Tensor) -> Tensor:

        feat = self.base(input)
        pred = self.fc(feat)
        return pred, feat


class PositiveClassLoss(nn.Module):

    def forward(self, pred: Tensor, target: torch.LongTensor) -> Tensor:

        # The definition of the positive class loss in the paper is a bit inscrutable.
        # Based on how the authors motivate it, we *guess* that they mean the positive
        # part of the binary cross-entropy loss.
        target_score = pred.gather(-1, target.view(-1, 1)).flatten()
        pcl = -F.logsigmoid(target_score)

        normed_pred = pred / torch.norm(pred, p=2, dim=-1, keepdim=True)
        cosine_loss = 1. - normed_pred.gather(-1, target.view(-1, 1)).flatten()

        return pcl.mean() + cosine_loss.mean()


class DSKLoss(nn.Module):

    def __init__(self, num_classes: int, feat_dim: int, center_loss_weight: float = 1.0) -> None:

        super(DSKLoss, self).__init__()
        self.pcl = PositiveClassLoss()
        self.center_loss = CenterLoss(num_classes, feat_dim)
        self.center_loss_weight = center_loss_weight


    def forward(self, pred: Tensor, feat: Tensor, target: torch.LongTensor) -> Tensor:

        return self.pcl(pred, target) + self.center_loss_weight * self.center_loss(feat, target)


class DSKClassifier(BasicAugmentation):
    """ Dual Selective Kernel Networks with a combination of cross-entropy, cosine, and center loss.

    Paper: https://link.springer.com/chapter/10.1007/978-3-030-66096-3_35

    Hyper-Parameters
    ----------------
    composite_loss : bool
        Whether to use the composite loss consisting of positive class loss, cosine loss,
        and center loss. If False, standard categorical cross-entropy loss will be used.
    center_feat_dim : int
        Feature dimensionality to project the model output to immediately before the
        final classification layer. This feature space will be used for computing
        the center loss. If set to None or 0 (the default), the feature dimensionality of
        the network will be used.
    centerloss_weight : float
        If set to a positive value, an additional feature projection layer will be
        added before the final classification layer and the center loss computed over
        these features will be added to the final loss, weighted with the value of
        this hyper-parameter.
    center_lr : float
        Learning rate for updating the centers used by center loss.
    
    See `BasicAugmentation` for a documentation of further hyper-parameters.
    """

    def __init__(self, **hparams):

        super(DSKClassifier, self).__init__(**hparams)
        self.num_classes = None
        self.feat_dim = None
        self.loss = None


    def create_model(self, arch: str, num_classes: int, input_channels: int, config: dict = {}) -> nn.Module:

        arch_class = self.get_arch_class(arch)
        arch_conf = arch_class.get_config()

        # Create network with Dual Selective Kernel Blocks
        if arch_class == rn.ResNet:

            orig_kwargs = arch_conf[int(arch[2:])]
            arch_conf[int(arch[2:])] = {**orig_kwargs,
                                        'block' : DSKBasicBlock if orig_kwargs['block'] == torchvision.models.resnet.BasicBlock else DSKBottleneck,
                                        'groups' : 32
            }

        elif arch_class == rn_cifar.ResNet:
            
            orig_kwargs = arch_conf[int(arch[2:])]
            arch_conf[int(arch[2:])] = {**orig_kwargs,
                                        'block' : DSKBasicBlock,
                                        'shortcut_downsampling' : 'conv',
                                        'groups' : 8
            }
        
        elif arch_class == wrn_cifar.WideResNet:

            arch_conf['block'] = DSKWideBlock
        
        else:
            raise ValueError(f'Architecture {arch} is not supported by {self.__class__.__name__}.')
        
        model = super(DSKClassifier, self).create_model(arch, num_classes, input_channels, config=arch_conf)

        # Store number of classes and features for use in get_loss_function
        self.num_classes = num_classes
        self.feat_dim = self.hparams['center_feat_dim'] or model.fc.in_features
        
        # Make model return features in addition to class predictions
        if self.hparams['composite_loss'] and (self.hparams['centerloss_weight'] > 0):
            model = FeatureModelWrapper(model, num_classes, self.feat_dim)

        return model


    def get_loss_function(self) -> Callable:

        if not self.hparams['composite_loss']:
            return nn.CrossEntropyLoss(reduction='mean')
        
        elif self.hparams['centerloss_weight'] > 0:
            if self.num_classes is None:
                raise RuntimeError('create_model must be called before get_loss_function when using center loss.')
            self.loss = DSKLoss(self.num_classes, self.feat_dim, self.hparams['centerloss_weight'])
            return self.loss
        
        else:
            return PositiveClassLoss()


    def get_optimizer(self, model: nn.Module, max_epochs: int, max_iter: int) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:

        if self.hparams['composite_loss'] and (self.hparams['centerloss_weight'] > 0):

            if self.loss is None:
                raise RuntimeError('get_loss_function must be called before get_optimizer when using center loss.')

            optimizer = SGDWithCenterLoss(
                model.parameters(), self.loss.parameters(),
                lr=self.hparams['lr'], center_lr=self.hparams['center_lr'],
                center_loss_weight=self.hparams['centerloss_weight'],
                momentum=0.9, weight_decay=self.hparams['weight_decay']
            )

        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=self.hparams['lr'], momentum=0.9, weight_decay=self.hparams['weight_decay'])
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter)
        
        return optimizer, scheduler


    @staticmethod
    def get_pipe_name():

        return 'dsk'


    @staticmethod
    def default_hparams() -> dict:

        return {
            **super(DSKClassifier, DSKClassifier).default_hparams(),
            'composite_loss' : True,
            'center_feat_dim' : None,
            'centerloss_weight' : 0.5,
            'center_lr' : 0.5,
        }
