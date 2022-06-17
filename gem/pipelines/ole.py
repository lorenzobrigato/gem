import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
import scipy as sp
import numpy as np
from typing import Callable

from gem.pipelines.common import BasicAugmentation
from gem.architectures import rn, rn_cifar, wrn_cifar


class ResNetOle(rn.ResNet):

    def __init__(self, *args, **kwargs):
        super(ResNetOle, self).__init__(*args, **kwargs)
    
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
        out = self.fc(x)

        return out, x # return also deep features
    

class WideResNetOle(wrn_cifar.WideResNet):
            
    def __init__(self, *args, widen_factor=1, **kwargs):
        super(WideResNetOle, self).__init__(*args, widen_factor=widen_factor, **kwargs)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(self.bn1(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        out = self.fc(x)
        
        return out, x # return also deep features
    
    
class CifarResNetOle(rn_cifar.ResNet):
    
    def __init__(self, *args, **kwargs):
         super(CifarResNetOle, self).__init__(*args, **kwargs)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        out = self.fc(x)

        return out, x # return also deep features
    
    
class OleLoss_(Function):
        
    @staticmethod
    def forward(ctx, X, y, lambda_):
        
        X = X.cpu()
        y = y.cpu()
        
        classes = np.unique(y)
        
        N, D = X.shape

        DELTA = 1.
        
        # gradients initialization
        Obj_c = 0
        dX_c = np.zeros((N, D))
        Obj_all = 0;
        dX_all = torch.zeros((N,D))

        eigThd = 1e-6 # threshold small eigenvalues for a better subgradient

        # compute objective and gradient for first term \sum ||TX_c||*
        for c in classes:
            A = X[y==c,:]

            # SVD
            U, S, V = sp.linalg.svd(A, full_matrices = False, lapack_driver='gesvd') 
            
            V = V.T
            nuclear = np.sum(S);

            ## L_c = max(DELTA, ||TY_c||_*)-DELTA
            
            if nuclear>DELTA:
              Obj_c += nuclear;
            
              # discard small singular values
              r = np.sum(S<eigThd)
              uprod = U[:,0:U.shape[1]-r].dot(V[:,0:V.shape[1]-r].T)
            
              dX_c[y==c,:] += uprod

            else:
              Obj_c+= DELTA
              
        # compute objective and gradient for secon term ||TX||*
        U, S, V = sp.linalg.svd(X, full_matrices = False, lapack_driver='gesvd')  # all classes
        V = V.T

        Obj_all = np.sum(S);

        r = np.sum(S<eigThd)

        uprod = U[:,0:U.shape[1]-r].dot(V[:,0:V.shape[1]-r].T)

        dX_all = uprod

        obj = (Obj_c  - Obj_all) / N * lambda_
        obj = torch.FloatTensor([float(obj)]).cuda()
        
        dX = (dX_c  - dX_all)/ N * lambda_
        dX = torch.FloatTensor(dX)
        
        # save inputs and gradients for backward pass
        ctx.save_for_backward(torch.FloatTensor(X).cuda(), dX.cuda())
        
        return obj
    
    @staticmethod
    def backward(ctx, grad_output):
        dX = ctx.saved_tensors[1] if ctx.needs_input_grad[0] else None 
        return dX, None, None
        

class OleLoss(nn.Module):
    
    def __init__(self, lambda_: float = 0.25):
        super(OleLoss, self).__init__()
        self.lambda_ = lambda_
        self.ole = OleLoss_.apply
        self.xe = nn.CrossEntropyLoss()
        
    def forward(self, out, feat, y):
        ole = self.ole(feat, y, self.lambda_)
        xe = self.xe(out, y)
        loss = ole + xe
        return loss


class OleClassifier(BasicAugmentation):
    """ Orthogonal Low-rank Embedding (OLE) loss.

    Paper: https://arxiv.org/abs/1712.01727
    
    Hyper-Parameters
    ----------------
    lambda : float
        Weight of the orthogonal low-rank embedding loss
    
    See `BasicAugmentation` for a documentation of the available hyper-parameters.
    """

    def create_model(self, arch: str, num_classes: int, input_channels: int, config: dict = {}) -> nn.Module:

        arch_class = self.get_arch_class(arch)

        # Create network which also returns deep features depending on the requested architecture
        if arch_class == rn.ResNet:

            model = ResNetOle.build_architecture(arch, num_classes, input_channels)
        
        elif arch_class == rn_cifar.ResNet:

            model = CifarResNetOle.build_architecture(arch, num_classes, input_channels)
        
        elif arch_class == wrn_cifar.WideResNet:

            model = WideResNetOle.build_architecture(arch, num_classes, input_channels)
        
        else:
            raise ValueError(f'Architecture {arch} is not supported by {self.__class__.__name__}.')
        
        return model
    
    
    def get_loss_function(self) -> Callable:

        return OleLoss()


    @staticmethod
    def get_pipe_name():

        return 'ole'


    @staticmethod
    def default_hparams() -> dict:

        return {
            **super(OleClassifier, OleClassifier).default_hparams(),
            'lambda' : 0.25,
        }
    
