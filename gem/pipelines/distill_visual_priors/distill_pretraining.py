import torch
from torch import nn, Tensor
import torch.utils.data as datautil
import numpy as np
from ray import tune
from typing import Callable, Tuple, Optional, Union
from collections import OrderedDict

from gem.utils import is_notebook
if is_notebook():
    from tqdm.notebook import tqdm, trange
else:
    from tqdm import tqdm, trange

from gem.pipelines.distill_visual_priors.contrastive_augmentation import ContrastiveAugmentation, TwoCropsTransform
from gem.pipelines.common import ClassificationMetrics
from gem.evaluation import balanced_accuracy
from gem.architectures import rn, rn_cifar, wrn_cifar


class EmbeddingHead(nn.Module):
    
    def __init__(self, in_features, dim: int=128, mlp: bool=False):
        super(EmbeddingHead, self).__init__()
        self.in_features = in_features
        self.fc = nn.Linear(in_features=in_features, out_features=dim)
        
        if mlp:  # hack: brute-force replacement
            dim_mlp = self.fc.weight.shape[1]
            self.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.fc)
    
    def forward(self, x):
        return self.fc(x)


class SplitBatchNorm(nn.BatchNorm2d):
    '''
    SplitBatchNorm: simulate multi-gpu behavior of BatchNorm in one gpu by splitting alone the batch dimension
    implementation taken from
    
    https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb#scrollTo=bNd_Q_Osi0SO
    
    adapted originally from 
    
    https://github.com/davidcpage/cifar10-fast/blob/master/torch_backend.py
    '''
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        
    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split, 
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                input, self.running_mean, self.running_var, 
                self.weight, self.bias, False, self.momentum, self.eps)
        
        
class MoCo(nn.Module):

    def __init__(self, encoder_q: nn.Module,
                       encoder_k: nn.Module,
                       dim: int=128,
                       K: int=65536,
                       m: float=0.999,
                       T: float=0.07,
                       num_splits: int=8):
        
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        

        
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        dim_mlp = encoder_q.fc.weight.shape[1]
        num_classes = encoder_q.fc.weight.shape[0]
        self.mlp_q = EmbeddingHead(dim_mlp, dim=dim, mlp=True) # for Moco V2
        self.mlp_k = EmbeddingHead(dim_mlp, dim=dim, mlp=True) # for Moco V2
        
        self.lh = EmbeddingHead(dim_mlp, dim=num_classes, mlp=False) 
                
        # brute-force replacement of original fc with identity layers
        # needed to easily evaluate the representations of pre-training with a linear head during hpo.
        self.encoder_k.fc = nn.Identity()
        self.encoder_q.fc = nn.Identity()
        
        # substitute all BatchNorm layers with SplitBatchNorm to simulate multi-gpu DistributedDataParallel training
        def sub_bn(model):
            for child_name, child in model.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    new_attr = SplitBatchNorm(child.num_features, num_splits=num_splits)
                    setattr(model, child_name, new_attr)
                else:
                    sub_bn(child)
                    
        sub_bn(self.encoder_q)
        sub_bn(self.encoder_k)
        
        
        
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            
        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue_single_gpu(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle
    
    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]
    
    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k) -> Tuple[Tensor, Tensor]:
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.mlp_q(self.encoder_q(im_q))  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

            k = self.mlp_k(self.encoder_k(im_k))  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue_single_gpu(k)

        return logits, labels


class MoCo_smallbank(MoCo):
    
    def __init__(self, encoder_q: nn.Module,
                       encoder_k: nn.Module,
                       dim: int=128,
                       K: int=4096,
                       m: float=0.999,
                       T: float=0.07,
                       num_splits: int=8,
                       margin: float=0.6):
        
        super(MoCo_smallbank, self).__init__(encoder_q, encoder_k, dim, K, m, T, num_splits)
        self.margin = margin

    def forward(self, im_q, im_k) -> Tuple[Tensor, Tensor]:
        
        q_enc = self.encoder_q(im_q)        
        q = self.mlp_q(q_enc)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)
        
        # detach the encoder features needed for linear evaluation and pass them through a linear head
        q_enc_copy = q_enc.clone().detach() 
        lin_eval_out = self.lh(q_enc_copy)
        
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            im_k, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)
            k = self.mlp_k(self.encoder_k(im_k))  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)

        # adding the margin
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        lb_views = labels.view(-1, 1)
        if lb_views.is_cuda: lb_views = lb_views.cpu()
        delt_costh = torch.zeros(logits.size()).scatter_(1, lb_views, self.margin)
        delt_costh = delt_costh.cuda()
        logits_m = logits - delt_costh

        logits_m /= self.T
        self._dequeue_and_enqueue_single_gpu(k)

        return logits_m, labels, lin_eval_out
        

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

        
class DistillPreTraining(ContrastiveAugmentation):
    
    """ Moco V2 pre-training model proposed in Distilling Visual Priors from Self-Supervised Learning.
        At the same time, this class performs unsupervised pre-training and supervised training of a linear head attached
        on top of the encoder to get accuracy values needed for evaluation (e.g. hpo).
    
    Paper: https://arxiv.org/abs/2008.00261

    
    Hyper-Parameters
    ----------------
    K : int
        Queue size, number of negative keys.
    m : float
        Moco momentum of updating key encoder.
    T : float
        Softmax temperature.
    
    See `ContrastiveAugmentation` for a documentation of further hyper-parameters.
    """

    def create_model(self, arch: str, num_classes: int, input_channels: int, config: dict = {}) -> nn.Module:

        arch_class = self.get_arch_class(arch)
        
        if (arch_class == rn.ResNet) or (arch_class == rn_cifar.ResNet) or (arch_class == wrn_cifar.WideResNet):

            encoder_q = super(DistillPreTraining, self).create_model(arch, num_classes, input_channels)
            encoder_k = super(DistillPreTraining, self).create_model(arch, num_classes, input_channels)

            model = MoCo_smallbank(encoder_q,
                                   encoder_k,
                                   dim=128,
                                   K=self.hparams['K'],
                                   m=self.hparams['m'],
                                   T=self.hparams['T'],
                                   margin=self.hparams['margin'])

        else:
            raise ValueError(f'Architecture {arch} is not supported by {self.__class__.__name__}.')
            
        return model


    def get_loss_function(self) -> Callable:
        
        return nn.CrossEntropyLoss()
    
            
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
            
            y = y.cuda()
            X[0] = X[0].cuda()
            X[1] = X[1].cuda()
            
            optimizer.zero_grad(set_to_none=True)
            
            # lin_eval_out are detached to prevent the backward flow of the linear evaluation gradients
            output, target, lin_eval_out = model(*X) 
            loss_selfsup = criterion(output, target)
            loss_sup = criterion(lin_eval_out, y)
            
            loss = loss_selfsup + loss_sup
            loss.backward()
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            # loss and accuracy refer to the linear head evaluation
            total_loss += loss_sup.item() * X[0].size(0)
            total_acc += (lin_eval_out.argmax(dim=-1) == y).sum().item()
            num_samples += X[0].size(0)
            
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
                lin_eval_out = model.lh(model.encoder_q(X))  # performing the linear evaluation protocol
    
                loss = criterion(lin_eval_out, y)
                total_loss += loss.item() * len(X)
                total_acc += (lin_eval_out.argmax(dim=-1) == y).sum().item()
                num_samples += len(X)
        
        return ClassificationMetrics(total_loss / num_samples, total_acc / num_samples)

    
    def train(self,
              train_data,
              val_data,
              batch_size: int,
              epochs: int,
              architecture: str = 'rn50',
              init_weights: Optional[str] = None,
              show_progress: bool = True,
              show_sub_progress: bool = False,
              eval_interval: int = 1,
              multi_gpu: bool = False,
              load_workers: int = 8,
              keep_transform: bool = False,
              report_tuner: bool = False) -> Tuple[nn.Module, OrderedDict]:
        
        if not keep_transform:
            train_transform, test_transform = self.get_data_transforms(train_data)
            if val_data is not None:
                val_data.transform = test_transform
        # apply the multi-crop data transformation
        train_data.transform = TwoCropsTransform(train_transform)
        
        # dropping last batch because of the queue building in MoCo, same as in the original code
        train_loader = datautil.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=load_workers, pin_memory=True, drop_last=True,
        )
        val_loader = datautil.DataLoader(
            val_data, batch_size=batch_size, shuffle=False, num_workers=load_workers, pin_memory=True
        ) if val_data is not None else None
    
        # Create model
        model = self.create_model(architecture, train_data.num_classes, train_data.num_input_channels).cuda()
        
        if init_weights is not None:
            model = self.load_weights(model, init_weights)
        
        par_model = nn.DataParallel(model).cuda() if multi_gpu else model
    
        iterations = len(train_loader) * epochs
        criterion = self.get_loss_function()
        optimizer, scheduler = self.get_optimizer(par_model, max_epochs=epochs, max_iter=iterations)
        regularizer = self.get_regularizer()  # pylint: disable=assignment-from-none
    
        metrics = self.train_model(
                        par_model, train_loader, val_loader, optimizer, criterion, epochs,
                        train_args={ 'scheduler' : scheduler, 'regularizer' : regularizer, 'show_progress' : show_sub_progress },
                        eval_args={ 'show_progress' : show_sub_progress },
                        eval_interval=eval_interval, show_progress=show_progress,
                        report_tuner=report_tuner
                        )
    
        return model, metrics


    def evaluate(self, model: nn.Module, test_data, batch_size: int = 10, print_metrics: bool = False):
        
        class MoCo_eval(nn.Module):
            def __init__(self, encoder, linear_head):
                super(MoCo_eval, self).__init__()
                self.enc = encoder
                self.lh = linear_head
                
            def forward(self, x):
                return self.lh(self.enc(x))

        if print_metrics:
            print('Following linear evaluation protocol:')

        model_eval = MoCo_eval(model.encoder_q, model.lh)
        return super(DistillPreTraining, self).evaluate(model_eval, test_data, batch_size=batch_size, print_metrics=print_metrics)


    @staticmethod
    def get_pipe_name():

        return 'dvp-pretrain'


    @staticmethod
    def default_hparams() -> dict:

        return {
            **super(DistillPreTraining, DistillPreTraining).default_hparams(),
            'K' : 4096,
            'm' : 0.999,
            'T' : 0.07,
            'margin': 0.6
        }
        