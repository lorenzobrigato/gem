import torch
from torch import nn
import torchvision
import numpy as np
import math
import os
from scipy.stats import norm
from typing import Callable, Tuple
from collections import OrderedDict

from gem.utils import is_notebook
if is_notebook():
    from tqdm.notebook import tqdm, trange
else:
    from tqdm import tqdm, trange

from gem.pipelines.distill_visual_priors.ofd_architectures import BasicBlockOFD, BottleneckOFD, ResNetOFD, CifarResNetOFD, WideResNetOFD  
from gem.evaluation import balanced_accuracy
from gem.pipelines.common import BasicAugmentation, ClassificationMetrics
from gem.architectures import rn, rn_cifar, wrn_cifar


def distillation_loss(source, target, margin):
    loss = ((source - margin)**2 * ((source > margin) & (target <= margin)).float() +
            (source - target)**2 * ((source > target) & (target > margin) & (target <= 0)).float() +
            (source - target)**2 * (target > 0).float())
    return torch.abs(loss).sum()


def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)


def get_margin_from_BN(bn):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for (s, m) in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)

    return torch.FloatTensor(margin).to(std.device)


class Distiller(nn.Module):
    
    def __init__(self, t_net: nn.Module, s_net: nn.Module):
        super(Distiller, self).__init__()

        t_channels = t_net.get_channel_num()
        s_channels = s_net.get_channel_num()
        
        self.Connectors = nn.ModuleList([build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])

        teacher_bns = t_net.get_bn_before_relu()
        margins = [get_margin_from_BN(bn) for bn in teacher_bns]
        for i, margin in enumerate(margins):
            self.register_buffer('margin%d' % (i+1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

        self.t_net = t_net
        self.s_net = s_net

    def forward(self, x):

        t_feats, t_out = self.t_net.extract_feature(x, preReLU=True)
        s_feats, s_out = self.s_net.extract_feature(x, preReLU=True)
        feat_num = len(t_feats)

        loss_distill = 0
        for i in range(feat_num):
            s_feats[i] = self.Connectors[i](s_feats[i])
            loss_distill += distillation_loss(s_feats[i], t_feats[i].detach(), getattr(self, 'margin%d' % (i+1))) \
                            / 2 ** (feat_num - i - 1)

        return s_out, loss_distill
    
    
class DistillClassifier(BasicAugmentation):
    
    """ Self-distillation model proposed in Distilling Visual Priors from Self-Supervised Learning.
        This class should be executed after the pre-training stage of the DistillPreTraining class that provides
        a self-supervised pre-trained network.
    
    Paper: https://arxiv.org/abs/2008.00261

    
    Hyper-Parameters
    ----------------
    distill_pretraining_path : string
        Path to a DistllPreTraining saved checkpoint.
    lambda : float
        Weight of the distillation loss.
    no_distill_epoch : int
        Number of epochs before adding the distillation loss to the total loss.
    
    See `ContrastiveAugmentation` for a documentation of further hyper-parameters.
    """


    def create_model(self, arch: str, num_classes: int, input_channels: int = 3) -> nn.Module:

        arch_class = self.get_arch_class(arch)
        arch_conf = arch_class.get_config()

        # Create teacher and student networks with the OFD required Blocks (when needed)
        if arch_class == rn.ResNet:

            # Change default residual block to OFD Block
            orig_kwargs = arch_conf[int(arch[2:])]
            arch_conf[int(arch[2:])] = {**orig_kwargs,
                                        'block' : BasicBlockOFD if orig_kwargs['block'] == torchvision.models.resnet.BasicBlock else BottleneckOFD
            }

            t_net = ResNetOFD.build_architecture(arch, num_classes, input_channels, config=arch_conf)
            s_net = ResNetOFD.build_architecture(arch, num_classes, input_channels, config=arch_conf)
        
        elif arch_class == rn_cifar.ResNet:

            orig_kwargs = arch_conf[int(arch[2:])]
            arch_conf[int(arch[2:])] = {**orig_kwargs,
                                        'block' : BasicBlockOFD
            }

            t_net = CifarResNetOFD.build_architecture(arch, num_classes, input_channels, config=arch_conf)
            s_net = CifarResNetOFD.build_architecture(arch, num_classes, input_channels, config=arch_conf)
        
        elif arch_class == wrn_cifar.WideResNet:

            t_net = WideResNetOFD.build_architecture(arch, num_classes, input_channels)
            s_net = WideResNetOFD.build_architecture(arch, num_classes, input_channels)
        
        else:
            raise ValueError(f'Architecture {arch} is not supported by {self.__class__.__name__}.')
        
        # Instantiate distiller
        model = Distiller(t_net, s_net)
        
        # Load the self-supervised pre-trained encoder  
        path = self.hparams['distill_pretraining_path']
        state_dict = torch.load(path)
        for k in list(state_dict.keys()):
            # retain only encoder_q
            if k.startswith('encoder_q'):
                # remove prefix
                state_dict[k[len("encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        model.t_net.load_state_dict(state_dict, strict=False)
        model.s_net.load_state_dict(state_dict, strict=False)
        
        return model


    def get_optimizer(self, model: nn.Module, max_epochs: int, max_iter: int) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        
        # only student and connectors parameters to the optimizer
        optimizer = torch.optim.SGD(list(model.s_net.parameters()) + list(model.Connectors.parameters()),
                                    lr=self.hparams['lr'], momentum=0.9, weight_decay=self.hparams['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter)
        return optimizer, scheduler
    

    def get_loss_function(self) -> Callable:
        
        return nn.CrossEntropyLoss()
    
    
    def train_epoch(self,
                    model,
                    loader,
                    optimizer,
                    criterion,
                    epoch,
                    scheduler=None,
                    regularizer=None,
                    show_progress=True):

        model.train()
        model.s_net.train()
        model.t_net.train()
        total_loss = total_acc = num_samples = 0
            
        for X, y in tqdm(loader, leave=False, disable=not show_progress):
            
            X, y = X.cuda(), y.cuda()
            optimizer.zero_grad(set_to_none=True)
            output, loss_distill = model(X)
            
            loss_CE = criterion(output, y)
            if epoch < self.hparams['no_distill_epoch']:
                loss = loss_CE
            else:
                loss = loss_CE + self.hparams['lambda'] * (loss_distill.sum() / len(X))
            
            total_loss += loss.item() * len(X)
            total_acc += (output.argmax(dim=-1) == y).sum().item()
            num_samples += len(X)
            
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
                output, _ = model(X)
    
                loss = criterion(output, y)
                total_loss += loss.item() * len(X)
                total_acc += (output.argmax(dim=-1) == y).sum().item()
                num_samples += len(X)
        
        return ClassificationMetrics(total_loss / num_samples, total_acc / num_samples)
    
    
    def train_model(self,
                    model,
                    train_loader,
                    val_loader,
                    optimizer,
                    criterion,
                    epochs,
                    evaluate=True,
                    train_args={},
                    eval_args={},
                    eval_interval=1,
                    show_progress=True,
                    report_tuner=False) -> OrderedDict:
        
        if report_tuner:
            from ray import tune

        metrics = OrderedDict()
        progbar = trange(epochs, disable=not show_progress)
        for ep in progbar:
            
            # Pass the epoch number to the epoch training loop
            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion, epoch=ep + 1, **train_args)
    
            if not (isinstance(train_metrics, dict) or isinstance(train_metrics, OrderedDict)):
                train_metrics = train_metrics._asdict()
            for key, value in train_metrics.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)
            
            if (evaluate is True) and (val_loader is not None) and (ep == 0 or (ep + 1) % eval_interval == 0):
                
                val_metrics = self.evaluate_epoch(model, val_loader, criterion, **eval_args)
                
                if not (isinstance(val_metrics, dict) or isinstance(val_metrics, OrderedDict)):
                    val_metrics = val_metrics._asdict()
                for key, value in val_metrics.items():
                    if 'val_' + key not in metrics:
                        metrics['val_' + key] = []
                    metrics['val_' + key].append(value)

                if report_tuner is True:
                    
                    tune.report(loss=metrics['val_loss'][-1],
                                loss_avg5=np.mean(metrics['val_loss'][-5:]),
                                accuracy=metrics['val_accuracy'][-1],
                                accuracy_avg5=np.mean(metrics['val_accuracy'][-5:])
                                )
                    
            if 'lr' not in metrics:
                metrics['lr'] = []
            metrics['lr'].append(optimizer.param_groups[0]['lr'])
            progbar.set_postfix(OrderedDict((key, values[-1]) for key, values in metrics.items()))
            
        return metrics
        

    @staticmethod
    def get_pipe_name():

        return 'dvp-distill'


    @staticmethod
    def default_hparams() -> dict:

        return {
            **super(DistillClassifier, DistillClassifier).default_hparams(),
            'distill_pretraining_path': 'distill_pretraining.pth',
            'lambda' : 1e-4,
            'no_distill_epoch': 5
        }
        
