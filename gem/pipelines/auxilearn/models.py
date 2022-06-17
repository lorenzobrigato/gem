import torch
import numpy as np
from torch import nn
from typing import Union, List

from gem.pipelines.auxilearn.hypernet import MonoLinearHyperNet, MonoNonlinearHyperNet


class MainNet(nn.Module):
    
    def __init__(self, main_net: nn.Module, psi: List, num_classes: int):
        super().__init__()
        self.main_net = main_net

        # main task classifier
        self.classifier1 = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

        # auxiliary task classifier
        self.classifier2 = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, int(np.sum(psi))),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.main_net(x)
        p1, p2 = self.classifier1(x), self.classifier2(x)
        return p1, p2


class AuxiliaryNet(nn.Module):
    
    def __init__(self, aux_net: nn.Module, psi: List):
        super().__init__()
        self.aux_net = aux_net

        self.class_nb = psi
        # generate label head
        self.classifier1 = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, int(np.sum(self.class_nb))),
        )

    def mask_softmax(self, x, mask, dim=1):
        z = x.max(dim=dim)[0]
        x = x - z.reshape(-1, 1)
        logits = torch.exp(x) * mask / (torch.sum(torch.exp(x) * mask, dim=dim, keepdim=True) + 1e-7)
        return logits

    def forward(self, x, y):
        device = x.device
        x = self.aux_net(x)

        # build a binary mask by psi, we add epsilon=1e-8 to avoid nans
        index = torch.zeros([len(self.class_nb), np.sum(self.class_nb)]) + 1e-8
        for i in range(len(self.class_nb)):
            index[i, int(np.sum(self.class_nb[:i])):np.sum(self.class_nb[:i + 1])] = 1
        mask = index[y].to(device)

        predict = self.classifier1(x.view(x.size(0), -1))
        label_pred = self.mask_softmax(predict, mask, dim=1)
        return label_pred
    
    
class WrapperModel(nn.Module):
    
    def __init__(self,
                 main_net: nn.Module,
                 gen_net: nn.Module,
                 comb_net: Union[MonoLinearHyperNet, MonoNonlinearHyperNet],
                 psi: List,
                 num_classes: int):
        
        super().__init__()
        self.main_net = MainNet(main_net, psi, num_classes)
        self.gen_net = AuxiliaryNet(gen_net, psi)
        self.comb_net = comb_net
        
    def forward(self, x):
        p1, p2 = self.main_net(x)
        return p1, p2
    