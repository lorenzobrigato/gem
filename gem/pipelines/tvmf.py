import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data as datautil
import numpy as np
from typing import Callable, Optional

from gem.utils import is_notebook
if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
from gem.pipelines.common import BasicAugmentation, ClassificationMetrics
from gem.evaluation import balanced_accuracy_from_predictions
from gem.architectures import rn, rn_cifar, wrn_cifar


def predict_class_scores(model: nn.Module, data, transform: Optional[Callable] = None, batch_size: int = 10, softmax: bool = False) -> np.ndarray:


    if transform is not None:
        prev_transform = data.transform
        data.transform = transform

    loader = datautil.DataLoader(
        data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
    )

    model.eval()
    predictions = []
    
    with torch.no_grad():
        for X, y in loader:
            output, _ = model(X.cuda(), y.cuda()) # the model also performs the loss computation
            predictions.append(output.cpu().numpy())
    
    if transform is not None:
        data.transform = prev_transform

    return np.concatenate(predictions)


class ResNetFeat(rn.ResNet):

    def __init__(self, block, *args, **kwargs):
        super(ResNetFeat, self).__init__(block, *args, **kwargs)
        self.dim = 512 * block.expansion
    
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
        x = torch.flatten(x, 1)

        return x
    
    def get_dim(self):
        return self.dim
    

class WideResNetFeat(wrn_cifar.WideResNet):
            
    def __init__(self, *args, widen_factor=1, **kwargs):
        super(WideResNetFeat, self).__init__( *args, widen_factor=widen_factor, **kwargs)
        self.dim = 64 * widen_factor
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(self.bn1(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        
        return x
    
    def get_dim(self):
        return self.dim
    
    
class CifarResNetFeat(rn_cifar.ResNet):
    
    def __init__(self, *args, **kwargs):
         super(CifarResNetFeat, self).__init__( *args, **kwargs)
         self.dim = 64
         
    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)

        return x
    
    def get_dim(self):
        return self.dim


class CosLoss(nn.Linear):
    """
    Cosine Loss
    """
    def __init__(self, in_features, out_features, bias=False):
        super(CosLoss, self).__init__(in_features, out_features, bias)
        self.s_ = torch.nn.Parameter(torch.zeros(1))

    def loss(self, Z, target):
        s = F.softplus(self.s_).add(1.)
        l = F.cross_entropy(Z.mul(s), target, weight=None, ignore_index=-100, reduction='mean')
        return l
        
    def forward(self, input, target):
        logit = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1), self.bias) # [N x out_features]
        l = self.loss(logit, target)
        return logit, l


class tvMFLoss(CosLoss):
    """
    t-vMF Loss
    """
    def __init__(self, in_features, out_features, bias=False, kappa=16):
        super(tvMFLoss, self).__init__(in_features, out_features, bias)
        self.register_buffer('kappa', torch.Tensor([kappa]))

    def forward(self, input, target=None):
        assert target is not None
        cosine = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1), None) # [N x out_features]
        logit =  (1. + cosine).div(1. + (1.-cosine).mul(self.kappa)) - 1.

        if self.bias is not None:
            logit.add_(self.bias)

        l = self.loss(logit, target)
        return logit, l
    
    def extra_repr(self):
        return super(tvMFLoss, self).extra_repr() + ', kappa={}'.format(self.kappa)


class WrapperNet(nn.Module):
    
    def __init__(self, base_model, num_classes, **kwargs):
        super(WrapperNet, self).__init__()
        self.feat = base_model
        self.fc_loss = tvMFLoss(base_model.get_dim(), num_classes, **kwargs)
    
    def forward(self, x, target):
        x = self.feat(x)
        x, loss = self.fc_loss(x, target)
        return x, loss
    
    
class TvMFClassifier(BasicAugmentation):
    """ Classifier that implements the Student-t plus von Mises-Fisher
    distribution similarity as loss function.

    Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Kobayashi_T-vMF_Similarity_for_Regularizing_Intra-Class_Feature_Distribution_CVPR_2021_paper.html

    Hyper-Parameters
    ----------------
    kappa : float
        Parameter that controls the concentration of the distribution.
        If kappa is equal to zero, the original cosine similarity is implemented.
    
    See `BasicAugmentation` for a documentation of further hyper-parameters.
    """
    
    def create_model(self, arch: str, num_classes: int, input_channels: int = 3, config: dict = {}) -> nn.Module:

        # Create network which returns deep features depending on the requested architecture
        arch_class = self.get_arch_class(arch)

        # Create network which also returns deep features depending on the requested architecture
        if arch_class == rn.ResNet:

            model = ResNetFeat.build_architecture(arch, num_classes, input_channels)
        
        elif arch_class == rn_cifar.ResNet:

            model = CifarResNetFeat.build_architecture(arch, num_classes, input_channels)
        
        elif arch_class == wrn_cifar.WideResNet:

            model = WideResNetFeat.build_architecture(arch, num_classes, input_channels)
        
        else:
            raise ValueError(f'Architecture {arch} is not supported by {self.__class__.__name__}.')
        
        model = WrapperNet(model, num_classes, kappa=self.hparams['kappa'])
        
        return model
    
    
    def get_loss_function(self) -> None:
        """
        The loss function is embedded inside WrapperNet following the original code
        """
        return None
    

    def train_epoch(self,
                    model,
                    loader,
                    optimizer,
                    criterion,
                    scheduler=None,
                    regularizer=None,
                    show_progress=True):
        
        model.train()
        total_loss = total_acc = num_samples = 0
        
        for X, y in tqdm(loader, leave=False, disable=not show_progress):
            
            X, y = X.cuda(), y.cuda()
            optimizer.zero_grad(set_to_none=True)
            output, loss = model(X, y) # the model also performs the loss computation
            
            total_loss += loss.item() * len(X)
            total_acc += (output.argmax(dim=-1) == y).sum().item()
            num_samples += len(X)
    
            if regularizer is not None:
                loss = loss + regularizer(model)
            
            loss.backward()
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
        
        return ClassificationMetrics(total_loss / num_samples, total_acc / num_samples)


    def evaluate_epoch(self,
                       model,
                       loader,
                       criterion,
                       show_progress=True):
        
        model.eval()
        total_loss = total_acc = num_samples = 0
        
        with torch.no_grad():
            for X, y in tqdm(loader, leave=False, disable=not show_progress):
    
                X, y = X.cuda(), y.cuda()
                output, loss = model(X, y) # the model also performs the loss computation

                total_loss += loss.item() * len(X)
                total_acc += (output.argmax(dim=-1) == y).sum().item()
                num_samples += len(X)
        
        return ClassificationMetrics(total_loss / num_samples, total_acc / num_samples)


    def evaluate(self, model: nn.Module, test_data, batch_size: int = 10, print_metrics: bool = False):
        
        gt = np.asarray(test_data.targets)
        pred = predict_class_scores(model, test_data, batch_size=batch_size).argmax(axis=-1)

        acc = np.mean(gt == pred)
        acc_b = balanced_accuracy_from_predictions(test_data.targets, pred)
        
        if print_metrics:
            print('Accuracy: {:.2%}'.format(acc))
            print('Balanced accuracy: {:.2%}'.format(acc_b))
        
        return {
            'accuracy' : acc,
            'balanced_accuracy' : acc_b,
        }    


    @staticmethod
    def get_pipe_name():

        return 'tvmf'


    @staticmethod
    def default_hparams() -> dict:

        return {
            **super(TvMFClassifier, TvMFClassifier).default_hparams(),
            'kappa' : 16.0,
        }
        