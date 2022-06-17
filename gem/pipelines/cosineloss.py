import torch
from torch import nn, Tensor
from typing import Callable

from gem.pipelines.common import BasicAugmentation


class OneHotCosineLoss(nn.Module):

    def __init__(self, reduction: str = 'mean') -> None:

        super(OneHotCosineLoss, self).__init__()
        self.reduction = reduction


    def forward(self, input: Tensor, target: torch.LongTensor) -> Tensor:

        normed = input / torch.norm(input, dim=-1, keepdim=True)
        loss = 1. - normed.gather(-1, target[:,None])

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction != 'none':
            raise RuntimeError(f'Unknown reduction: {self.reduction}')
        
        return loss


class OneHotCosineXentLoss(nn.Module):

    def __init__(self, xent_weight: float, reduction: str = 'mean') -> None:

        super(OneHotCosineXentLoss, self).__init__()
        self.xent_weight = xent_weight
        self.reduction = reduction
        self.xent = nn.CrossEntropyLoss(reduction='none')


    def forward(self, logits: Tensor, embeddings: Tensor, target: torch.LongTensor) -> Tensor:

        cosine_loss = 1. - embeddings.gather(-1, target[:,None])
        xent_loss = self.xent(logits, target)
        loss = cosine_loss + self.xent_weight * xent_loss

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction != 'none':
            raise RuntimeError(f'Unknown reduction: {self.reduction}')
        
        return loss


class CosineXentModel(nn.Module):

    def __init__(self, base_model: nn.Module, num_classes: int) -> None:

        super(CosineXentModel, self).__init__()
        self.base = base_model
        self.bn = nn.BatchNorm1d(num_classes)
        self.fc = nn.Linear(num_classes, num_classes)
    

    def forward(self, input: Tensor) -> Tensor:

        embeddings = self.base(input)
        normed_embeddings = embeddings / torch.norm(embeddings, dim=-1, keepdim=True)
        feat = self.bn(torch.relu(normed_embeddings))
        logits = self.fc(feat)
        return logits, normed_embeddings


class CosineLossClassifier(BasicAugmentation):
    """ Cosine loss classification with one-hot targets.

    Paper: https://arxiv.org/abs/1901.09054

    Hyper-Parameters
    ----------------
    xent_weight : float
        If set to a positive value, an additional classification layer will be added
        on top of the embedding layer and the final loss will be the sum of the
        cosine loss on the embedding layer and the cross-entropy loss on the classification
        layer. This parameter specifies the weight of the cross-entropy loss.
    
    See `BasicAugmentation` for a documentation of further hyper-parameters.
    """

    def create_model(self, arch: str, num_classes: int, input_channels: int = 3) -> nn.Module:

        model = super(CosineLossClassifier, self).create_model(arch, num_classes=num_classes, input_channels=input_channels)
        if self.hparams['xent_weight'] > 0:
            model = CosineXentModel(model, num_classes)
        return model


    def get_loss_function(self) -> Callable:

        if self.hparams['xent_weight'] > 0:
            return OneHotCosineXentLoss(self.hparams['xent_weight'])
        else:
            return OneHotCosineLoss()


    @staticmethod
    def get_pipe_name():

        return 'cosine'


    @staticmethod
    def default_hparams() -> dict:

        return {
            **super(CosineLossClassifier, CosineLossClassifier).default_hparams(),
            'xent_weight' : 0,
        }
