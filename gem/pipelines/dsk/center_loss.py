""" Implementation of Center Loss from https://github.com/KaiyangZhou/pytorch-center-loss """

import torch
from torch import nn, Tensor


class CenterLoss(nn.Module):
    """ Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Parameters
    ----------
    num_classes : int
        Number of classes.
    feat_dim : int
        Dimensionality of feature vectors.
    """
    def __init__(self, num_classes: int, feat_dim: int, use_gpu: bool = True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))


    def forward(self, x: Tensor, labels: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : nn.Tensor
            Feature matrix with shape (batch_size, feat_dim).
        labels : nn.Tensor
            Ground truth labels with shape (batch_size,).
        """
        normed_feat = x / torch.clamp(torch.norm(x, dim=-1, p=2, keepdim=True), min=1e-8)
        normed_centers = self.centers / torch.clamp(torch.norm(self.centers, dim=-1, p=2, keepdim=True), min=1e-8)
        dist = 2 - 2 * torch.sum(normed_feat * normed_centers[labels], dim=-1)
        loss = dist.clamp(min=1e-12, max=1e+12).mean()
        return loss



class SGDWithCenterLoss(torch.optim.SGD):
    """ SGD optimizing both model parameters and center loss centers.

    Parameters
    ----------
    model_params : iterable
        Model parameters.
    centerloss_params : iterable
        Center loss parameters.
    lr : float
        Learning rate for optimizing model parameters.
    center_lr : float
        Learning rate for optimizing center loss parameters.
    center_loss_weight : float
        The factor/weight with wich the center loss is added to the total loss.
        Needed for correcting the influence of this weight on the gradients.
    **kwargs
        Further arguments supported by torch.optim.SGD. Will only be applied
        for optimizing model parameters.
    """

    def __init__(self, model_params, centerloss_params, lr: float, center_lr: float, center_loss_weight: float, **kwargs) -> None:

        super(SGDWithCenterLoss, self).__init__(model_params, lr=lr, **kwargs)
        self.centerloss_params = centerloss_params
        self.centerloss_optimizer = torch.optim.SGD(self.centerloss_params, lr=center_lr)
        self.center_loss_weight = center_loss_weight


    def zero_grad(self, set_to_none: bool = False):

        super(SGDWithCenterLoss, self).zero_grad(set_to_none=set_to_none)
        self.centerloss_optimizer.zero_grad(set_to_none=set_to_none)


    @torch.no_grad()
    def step(self, closure=None):

        # Remove effect of the weight of the center loss on updating centers
        for param in self.centerloss_params:
            param.grad.data *= 1. / self.center_loss_weight

        super(SGDWithCenterLoss, self).step()
        self.centerloss_optimizer.step()
