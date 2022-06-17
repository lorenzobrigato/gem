import torch
from torch import nn, Tensor
from torch.autograd import grad
from typing import Callable, Tuple


from gem.pipelines.common import BasicAugmentation

    
class XentAvgGradL2Loss(nn.Module):

    def __init__(self, model: nn.Module, gradl2_weight: float, reduction: str = 'mean', multi_gpu: bool = False) -> None:

        super(XentAvgGradL2Loss, self).__init__()
        self.model = model
        self.par_model = nn.DataParallel(model).cuda() if multi_gpu else model
        self.gradl2_weight = gradl2_weight
        self.reduction = reduction
        self.xent = nn.CrossEntropyLoss(reduction=reduction)
        self.xent_sum = nn.CrossEntropyLoss(reduction='sum')
        

    def forward(self, logits: Tensor, imgs: Tensor, target: torch.LongTensor) -> Tensor:
        
        xent_loss = self.xent(logits, target)

        if self.model.training:
            n = imgs.shape[0]
            imgsv = imgs.clone().requires_grad_()
        
            pred, _ = self.par_model(imgsv)
            l = self.xent_sum(pred, target)
            g, = grad(l, imgsv, create_graph=True)
            grad_pen = torch.sum(g ** 2) / n
        else:
            grad_pen = 0.0
        
        loss = xent_loss + (self.gradl2_weight * grad_pen)

        return loss


class GradL2Model(nn.Module):

    def __init__(self, base_model: nn.Module, bn: bool = False) -> None:

        super(GradL2Model, self).__init__()
        self.base = base_model
        if not bn:
            self.remove_bn(self.base)

        
    def remove_bn(self, model):
        for child_name, child in model.named_children():
            if isinstance(child, nn.BatchNorm2d):
                setattr(model, child_name, nn.Identity())
            else:
                self.remove_bn(child)
                    
                
    def forward(self, imgs: Tensor) -> Tuple[Tensor, Tensor]:

        logits = self.base(imgs)
        return logits, imgs


class GradL2LossClassifier(BasicAugmentation):
    """ Cross-entropy classifier with grad-l2 penalty.

    Paper: https://arxiv.org/abs/1810.00363

    Hyper-Parameters
    ----------------
    gradl2_weight : float
        Value that weights the contribution of the grad-l2 penalty.
    bn : bool, default: False
        Whether to use batch normalization.
    multi_gpu : bool, default: False
        Set this to True when parallelizing training across multiple GPUs.
        Otherwise, the gradients used for regularization will be computed on the main device only.
        This parameter only controls the parallelization of the regularizer computation, not of
        the normal forward pass.
    
    See `BasicAugmentation` for a documentation of further hyper-parameters.
    """

    def create_model(self, arch: str, num_classes: int, input_channels: int, config: dict = {}) -> nn.Module:

        model = super(GradL2LossClassifier, self).create_model(arch, num_classes, input_channels)
        model = GradL2Model(model, bn=self.hparams['bn'])
        self.model = model
        return self.model


    def get_loss_function(self) -> Callable:
        
        return XentAvgGradL2Loss(self.model, self.hparams['gradl2_weight'], multi_gpu=self.hparams['multi_gpu'])



    @staticmethod
    def get_pipe_name():

        return 'gradl2'


    @staticmethod
    def default_hparams() -> dict:

        return {
            **super(GradL2LossClassifier, GradL2LossClassifier).default_hparams(),
            'gradl2_weight' : 0.01,
            'bn' : True,
            'multi_gpu' : False,
        }
