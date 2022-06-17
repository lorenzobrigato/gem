import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

        
        
class MetaOptimizer:

    def __init__(self, meta_optimizer, hpo_lr, truncate_iter=3, max_grad_norm=10):
        """Auxiliary parameters optimizer wrapper
        :param meta_optimizer: optimizer for auxiliary parameters
        :param hpo_lr: learning rate to scale the terms in the Neumann series
        :param truncate_iter: number of terms in the Neumann series
        :param max_grad_norm: max norm for grad clipping
        """
        self.meta_optimizer = meta_optimizer
        self.hypergrad = Hypergrad(learning_rate=hpo_lr, truncate_iter=truncate_iter)
        self.max_grad_norm = max_grad_norm

    def step(self, train_loss, val_loss, parameters, aux_params, return_grads=False):
        """
        :param train_loss: train loader
        :param val_loss:
        :param parameters: parameters (main net)
        :param aux_params: auxiliary parameters
        :param return_grads: whether to return gradients
        :return:
        """
        # zero grad
        self.zero_grad()

        # validation loss
        hyper_gards = self.hypergrad.grad(
            loss_val=val_loss,
            loss_train=train_loss,
            aux_params=aux_params,
            params=parameters
        )

        for p, g in zip(aux_params, hyper_gards):
            p.grad = g

        # grad clipping
        if self.max_grad_norm is not None:
            clip_grad_norm_(aux_params, max_norm=self.max_grad_norm)

        # meta step
        self.meta_optimizer.step()
        if return_grads:
            return hyper_gards

    def zero_grad(self):
        self.meta_optimizer.zero_grad()
        
        
class Hypergrad:
    """Implicit differentiation for auxiliary parameters.
    This implementation follows the Algs. in "Optimizing Millions of Hyperparameters by Implicit Differentiation"
    (https://arxiv.org/pdf/1911.02590.pdf), with small differences.
    """

    def __init__(self, learning_rate=.1, truncate_iter=3):
        self.learning_rate = learning_rate
        self.truncate_iter = truncate_iter

    def grad(self, loss_val, loss_train, aux_params, params):
        """Calculates the gradients w.r.t \phi dloss_aux/dphi, see paper for details
        :param loss_val:
        :param loss_train:
        :param aux_params:
        :param params:
        :return:
        """
        dloss_val_dparams = torch.autograd.grad(
            loss_val,
            params,
            retain_graph=True,
            allow_unused=True
        )

        dloss_train_dparams = torch.autograd.grad(
                loss_train,
                params,
                allow_unused=True,
                create_graph=True,
        )

        v2 = self._approx_inverse_hvp(dloss_val_dparams, dloss_train_dparams, params)

        v3 = torch.autograd.grad(
            dloss_train_dparams,
            aux_params,
            grad_outputs=v2,
            allow_unused=True
        )

        # note we omit dL_v/d_lambda since it is zero in our settings
        return list(-g for g in v3)

    def _approx_inverse_hvp(self, dloss_val_dparams, dloss_train_dparams, params):
        """
        :param dloss_val_dparams: dL_val/dW
        :param dloss_train_dparams: dL_train/dW
        :param params: weights W
        :return: dl_val/dW * dW/dphi
        """
        p = v = dloss_val_dparams

        for _ in range(self.truncate_iter):
            grad = torch.autograd.grad(
                    dloss_train_dparams,
                    params,
                    grad_outputs=v,
                    retain_graph=True,
                    allow_unused=True
                )

            grad = [g * self.learning_rate for g in grad]  # scale: this a is key for convergence

            v = [curr_v - curr_g for (curr_v, curr_g) in zip(v, grad)]
            # note: different than the pseudo code in the paper
            p = [curr_p + curr_v for (curr_p, curr_v) in zip(p, v)]

        return list(pp for pp in p)


class OptimizerWrapper():
    
    def __init__(self, optimizer: torch.optim.Optimizer, meta_optimizer: MetaOptimizer):
        super().__init__()
        self.optimizer = optimizer
        self.meta_optimizer = meta_optimizer
        
        