import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data as datautil
import numpy as np
from typing import Tuple, Callable, Optional
from collections import OrderedDict
from sklearn.model_selection import train_test_split

from gem.utils import is_notebook
if is_notebook():
    from tqdm.notebook import tqdm, trange
else:
    from tqdm import tqdm, trange
    
from gem.pipelines.auxilearn.models import WrapperModel
from gem.pipelines.auxilearn.hypernet import MonoLinearHyperNet, MonoNonlinearHyperNet
from gem.pipelines.auxilearn.optim import MetaOptimizer, OptimizerWrapper
from gem.pipelines.common import BasicAugmentation, ClassificationMetrics
from gem.architectures import rn, rn_cifar, wrn_cifar


class AuxiLearnClassifier(BasicAugmentation):
    """ AuxiLearn classifier that generates auxiliary tasks to improve performance on the main task of interest.

    Paper: https://arxiv.org/abs/2007.02693
    
    Hyper-Parameters
    ----------------
    num_aux_classes : int
        Number of classes for auxiliary tasks.
    aux_net : str
        Type of network that combines losses, Choose between linear and nonlinear.
    units_nonlinear : int
        Width of nonlinear layer of aux_net. Affecting only in case of nonlinear aux_net.
    depth_nonlinear : int
        Depth of aux_net. Affecting only in case of nonlinear aux_net.
    auxgrad_every : int
        Number of optimization steps between auxiliary parameters updates.    
    aux_scale : float
        Auxiliary task scale.
    aux_set_size : float
        Percentage of samples to allocate for auxiliary set (contained in (0, 1]). 
        
    See `BasicAugmentation` for a documentation of further hyper-parameters.
    """
    
    def create_model(self, arch: str, num_classes: int, input_channels: int, config: dict = {}) -> nn.Module:
        
        arch_class = self.get_arch_class(arch)
        
        if (arch_class == rn.ResNet) or (arch_class == rn_cifar.ResNet) or (arch_class == wrn_cifar.WideResNet):

            # keeping the 1000D linear layer for the main and task-generative networks as in the original code
            main_net = super(AuxiLearnClassifier, self).create_model(arch, 1000, input_channels)
            aux_gen_net = super(AuxiLearnClassifier, self).create_model(arch, 1000, input_channels)

            aux_comb_mapping = dict(
                linear=MonoLinearHyperNet,
                nonlinear=MonoNonlinearHyperNet,
            )
            
            aux_comb_config = dict(input_dim=2, main_task=0, weight_normalization=False)
            
            # setting default parameters
            if self.hparams['aux_net'] == 'nonlinear':
                aux_comb_config['hidden_sizes'] = [self.hparams['units_nonlinear']] * self.hparams['depth_nonlinear']
                aux_comb_config['init_upper'] = 0.2
            elif self.hparams['aux_net'] == 'linear':
                aux_comb_config['skip_connection'] = True
            else:
                ValueError('aux_net parameter {} is not expected. Choose among linear or nonlinear.'.format(self.hparams["aux_net"]))
            
            aux_comb_net = aux_comb_mapping[self.hparams['aux_net']](**aux_comb_config)

            psi = [self.hparams['num_aux_classes']] * num_classes
            model = WrapperModel(main_net, aux_gen_net, aux_comb_net, psi, num_classes)
        
        else:
            raise ValueError(f'Architecture {arch} is not supported by {self.__class__.__name__}.')
        
        return model
    

    def get_optimizer(self, model: WrapperModel, max_epochs: int, max_iter: int) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:

        optimizer = torch.optim.SGD(model.main_net.parameters(),
                                    lr=self.hparams['lr'],
                                    momentum=0.9,
                                    weight_decay=self.hparams['weight_decay'])
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter)
        
        auxiliary_params = list(model.gen_net.parameters())
        auxiliary_params += list(model.comb_net.parameters())
        
        
        meta_opt = torch.optim.SGD(auxiliary_params,
                                   lr=self.hparams['lr'],
                                   momentum=0.9,
                                   weight_decay=self.hparams['weight_decay'])
            
        meta_optimizer = MetaOptimizer(
            meta_optimizer=meta_opt, hpo_lr=self.hparams['lr'], truncate_iter=3, max_grad_norm=50
        )
        
        optim_wrap = OptimizerWrapper(optimizer, meta_optimizer)
        
        return optim_wrap, scheduler
    
    
    def get_loss_function(self) -> Callable:
        
        def calc_loss(x_pred, x_output, num_output, pri=True):
            """Focal loss
            :param x_pred:  prediction of primary network (either main or auxiliary)
            :param x_output: label
            :param pri: is primary task output?
            :param num_output: number of classes
            :return: loss per sample
            """
            if not pri:
                # generated auxiliary label is a soft-assignment vector (no need to change into one-hot vector)
                x_output_onehot = x_output
            else:
                # convert a single label into a one-hot vector
                x_output_onehot = torch.zeros((len(x_output), num_output)).to(x_pred.device)
                x_output_onehot.scatter_(1, x_output.unsqueeze(1), 1)
                x_pred = F.softmax(x_pred, dim=1)
        
            loss = torch.sum(- x_output_onehot * (1 - x_pred) ** 2 * torch.log(x_pred + 1e-12), dim=1)
        
            return loss
        
        return calc_loss


    def train_epoch(self,
                    model,
                    loader,
                    aux_loader,
                    optimizer,
                    criterion,
                    step,
                    scheduler=None,
                    regularizer=None,
                    show_progress=True):
        
        model.main_net.train()
        total_loss = total_acc = num_samples = 0
        
        for X, y in tqdm(loader, leave=False, disable=not show_progress):
            
            step += 1
            X, y = X.cuda(), y.cuda()
            optimizer.optimizer.zero_grad(set_to_none=True)
            
            main_pred, aux_pred = model(X)
            aux_label = model.gen_net(X, y)
            
            train_loss_main = criterion(main_pred, y, pri=True, num_output=loader.dataset.dataset.num_classes)
            train_loss_aux = criterion(aux_pred, aux_label, pri=False,
                                       num_output=loader.dataset.dataset.num_classes * self.hparams['num_aux_classes']) * self.hparams['aux_scale']
            
            loss = model.comb_net(torch.stack((train_loss_main, train_loss_aux)).t())
            
            main_loss_data = train_loss_main.mean()
        
            total_loss += main_loss_data.item() * len(X)
            total_acc += (main_pred.argmax(dim=-1) == y).sum().item()
            num_samples += len(X)
    
            if regularizer is not None:
                loss = loss + regularizer(model)
            
            loss.backward()
            optimizer.optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
                
            if step % self.hparams['auxgrad_every'] == 0:
                
                # hyperstep
                meta_val_loss = .0
                
                for data, clf_labels in aux_loader:
                    
                    data, clf_labels = data.cuda(), clf_labels.cuda()
            
                    val_pred, val_labels = model.main_net(data)
            
                    val_loss = criterion(val_pred, clf_labels, pri=True, num_output=loader.dataset.dataset.num_classes).mean()
                    meta_val_loss += val_loss
                    break
            
                inner_loop_end_train_loss = 0.
                for train_data, train_target in loader:
                    # to device and take only first val_batch_size
                    train_data, train_target = train_data.cuda()[:len(train_data), ], train_target.cuda()[:len(train_data), ]
                    
                    train_main_pred, train_aux_pred = model.main_net(train_data)
                    train_aux_target = model.gen_net(train_data, train_target)
            
                    inner_train_loss_main = criterion(train_main_pred, train_target, pri=True,
                                                      num_output=loader.dataset.dataset.num_classes)
                    inner_train_loss_aux = criterion(train_aux_pred, train_aux_target, pri=False,
                                                     num_output=loader.dataset.dataset.num_classes * self.hparams['num_aux_classes']) * self.hparams['aux_scale']
            
                    inner_loop_end_train_loss += model.comb_net(
                        torch.stack((inner_train_loss_main, inner_train_loss_aux)).t()
                    )
                    break
            
                phi = list(model.gen_net.parameters())
                phi += list(model.comb_net.parameters())
                W = [p for n, p in model.main_net.named_parameters() if 'classifier2' not in n]
            
                optimizer.meta_optimizer.step(
                    val_loss=meta_val_loss,
                    train_loss=inner_loop_end_train_loss,
                    aux_params=phi,
                    parameters=W,
                    return_grads=False
                )
                
        return ClassificationMetrics(total_loss / num_samples, total_acc / num_samples), step
    

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
                main_pred, _ = model(X)
                
                loss = criterion(main_pred, y, pri=True, num_output=loader.dataset.num_classes)
                loss = loss.mean()
                total_loss += loss.item() * len(X)
                total_acc += (main_pred.argmax(dim=-1) == y).sum().item()
                num_samples += len(X)
        
        return ClassificationMetrics(total_loss / num_samples, total_acc / num_samples)
    
    
    def train_model(self,
                    model,
                    train_loader,
                    aux_loader,
                    val_loader,
                    optimizer,
                    criterion,
                    epochs,
                    evaluate=True,
                    train_args={},
                    eval_args={},
                    eval_interval=1,
                    show_progress=True,
                    report_tuner=False):

        if report_tuner:
            from ray import tune
        
        metrics = OrderedDict()
        progbar = trange(epochs, disable=not show_progress)
        step = 0 # enumerate steps needed for updating the aux_grad
        
        for ep in progbar:
            
            # pass and return the variable step that keeps the count of all the updates
            train_metrics, step = self.train_epoch(model, train_loader, aux_loader, optimizer, criterion, step, **train_args)
    
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
            metrics['lr'].append(optimizer.optimizer.param_groups[0]['lr'])
            progbar.set_postfix(OrderedDict((key, values[-1]) for key, values in metrics.items()))
            
        return metrics


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
              report_tuner: bool = False):
                
        # Set data transforms
        if not keep_transform:
            train_transform, test_transform = self.get_data_transforms(train_data)
            train_data.transform = train_transform
            if val_data is not None:
                val_data.transform = test_transform
        
        # Create auxiliary split
        test_size = self.hparams['aux_set_size'] if self.hparams['aux_set_size'] * len(train_data) >= train_data.num_classes else train_data.num_classes
        
        train_indices, aux_indices = train_test_split(
            range(len(train_data)), test_size=test_size, random_state=42,
            stratify=train_data.targets
        )
        
        meta_data = datautil.Subset(train_data, aux_indices)
        train_data = datautil.Subset(train_data, train_indices)
        
        # Create data loaders
        train_loader = datautil.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=load_workers, pin_memory=True
        )
        val_loader = datautil.DataLoader(
            val_data, batch_size=batch_size, shuffle=False, num_workers=load_workers, pin_memory=True
        ) if val_data is not None else None
        
        aux_loader = datautil.DataLoader(
            meta_data, batch_size=batch_size, shuffle=True, num_workers=load_workers, pin_memory=True
        )
        
        # Create model
        model = self.create_model(architecture, train_data.dataset.num_classes, train_data.dataset.num_input_channels).cuda()
        
        if init_weights is not None:
            model = self.load_weights(model, init_weights)
        
        par_model = nn.DataParallel(model).cuda() if multi_gpu else model
    
        # Get optimizer, LR schedule, regularizer, and loss function
        iterations = len(train_loader) * epochs
        criterion = self.get_loss_function()
        optimizer, scheduler = self.get_optimizer(par_model, max_epochs=epochs, max_iter=iterations)
        regularizer = self.get_regularizer()  # pylint: disable=assignment-from-none
        
        # Train model
        metrics = self.train_model(
                        par_model, train_loader, aux_loader, val_loader, optimizer, criterion, epochs,
                        train_args={ 'scheduler' : scheduler, 'regularizer' : regularizer, 'show_progress' : show_sub_progress },
                        eval_args={ 'show_progress' : show_sub_progress },
                        eval_interval=eval_interval, show_progress=show_progress,
                        report_tuner=report_tuner
                        )

        metrics = {}
    
        return model, metrics
    

    @staticmethod
    def get_pipe_name():

        return 'auxilearn'


    @staticmethod
    def default_hparams() -> dict:

        return {
            **super(AuxiLearnClassifier, AuxiLearnClassifier).default_hparams(),
            'num_aux_classes' : 5,
            'aux_net' : 'linear',
            'units_nonlinear' : 10,
            'depth_nonlinear' : 2,
            'auxgrad_every' : 15,
            'aux_scale' : 0.25,
            'aux_set_size' : 0.015
        }
    