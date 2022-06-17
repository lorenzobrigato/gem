import numpy as np
import torch
import torch.utils.data as datautil
import torchvision.transforms as tf
from torch import nn
from PIL import Image

from abc import ABC, abstractmethod
from typing import Tuple, Callable, Optional, Union
from collections import OrderedDict, namedtuple

from gem.loader import InstanceLoader
from gem.utils import is_notebook
from gem.evaluation import predict_class_scores, balanced_accuracy_from_predictions

if is_notebook():
    from tqdm.notebook import tqdm, trange
else:
    from tqdm import tqdm, trange


ClassificationMetrics = namedtuple('Metrics', ['loss', 'accuracy'])


class LearningMethod(ABC):
    """ Base class for all training methods (or "pipelines").

    A training pipeline consists of the following parts:
    - a data transformation,
    - a model to be trained,
    - an optimizer and learning rate scheduler,
    - a loss function,
    - a training procedure.

    A class derived from this class should provide all these parts.
    This allows for convenient use of different training techniques by simply calling
    the method `train` of this class.

    Parameters
    ----------
    **hparams
        Any keyword arguments will be considered as method-specific hyper-parameters
        and are stored in the `hparams` attribute.
        A list of available hyper-parameters and their defaults can be obtained from
        `default_hparams()`.
    
    Hyper-Parameters
    ----------------
    lr : float
        Learning rate.
    weight_decay : float
        Weight decay.
    """

    def __init__(self, **hparams):

        super(LearningMethod, self).__init__()
        self.hparams = { **self.__class__.default_hparams() }
        self.set_hparams(**hparams)


    def train_epoch(self,
                    model,
                    loader,
                    optimizer,
                    criterion,
                    scheduler=None,
                    regularizer=None,
                    show_progress=True):
        
        """ Performs one epoch of training a model.
    
        Parameters
        ----------
        model : torch.nn.Module
            The model to be trained.
            Might yield multiple outputs, but the first ones will be considered to
            be class scores for accuracy computation.
        loader : iterable
            The data loader, yielding batches of samples and labels.
        optimizer : torch.optim.Optimizer
            The optimizer to be used for the backward pass and model update.
        criterion : callable
            The loss function.
            All outputs of the model will be passed as argument, followed by
            the class labels.
        scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            A learning rate scheduler to be called after every iteration.
        regularizer : callable(torch.nn.Module), optional
            A function taking the model as argument and returning a regularization
            loss as scalar tensor that will be added to the total loss function.
        show_progress : bool, default: True
            Whether to show a tqdm progress bar updated after every iteration.
        
        Returns
        -------
        loss : float
        accuracy : float
        """
        
        model.train()
        total_loss = total_acc = num_samples = 0
        
        for X, y in tqdm(loader, leave=False, disable=not show_progress):
            
            X, y = X.cuda(), y.cuda()
            optimizer.zero_grad(set_to_none=True)
            output = model(X)
            if not isinstance(output, tuple):
                output = (output,)
            
            loss = criterion(*output, y)
            total_loss += loss.item() * len(X)
            total_acc += (output[0].argmax(dim=-1) == y).sum().item()
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
        
        """ Evaluates a classification model on validation/test data.
    
        Parameters
        ----------
        model : torch.nn.Module
            The trained model.
            Might yield multiple outputs, but the first ones will be considered to
            be class scores for accuracy computation.
        loader : iterable
            The data loader, yielding batches of samples and labels.
        criterion : callable
            The loss function.
            All outputs of the model will be passed as argument, followed by
            the class labels.
        show_progress : bool, default: True
            Whether to show a tqdm progress bar updated after every batch.
        
        Returns
        -------
        loss : float
        accuracy : float
        """
        
        model.eval()
        total_loss = total_acc = num_samples = 0
        
        with torch.no_grad():
            for X, y in tqdm(loader, leave=False, disable=not show_progress):
    
                X, y = X.cuda(), y.cuda()
                output = model(X)
                if not isinstance(output, tuple):
                    output = (output,)
    
                loss = criterion(*output, y)
                total_loss += loss.item() * len(X)
                total_acc += (output[0].argmax(dim=-1) == y).sum().item()
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
        
        """ Trains a classification model.
    
        Parameters
        ----------
        model : torch.nn.Module
            The model to be trained.
            Might yield multiple outputs, but the first ones will be considered to
            be class scores for accuracy computation.
        train_loader : iterable
            The training data loader, yielding batches of samples and labels.
        val_loader : iterable, optional
            The validation data loader, yielding batches of samples and labels.
        optimizer : torch.optim.Optimizer
            The optimizer to be used for the backward pass and model update.
        criterion : callable
            The loss function.
            All outputs of the model will be passed as argument, followed by
            the class labels.
        epochs : int
            Number of training epochs.
        evaluate : bool, optional, default: True
            The function to be called for evaluation. The first three arguments must
            be `model`, `val_loader`, and `criterion`.
        train_args : dict, optional
            Dictionary with additional keyword arguments passed to `train_func`.
        eval_args : dict, optional
            Dictionary with additional keyword arguments passed to `eval_func`.
        eval_interval : int, default: 1
            Number of epochs after which evaluation will be performed.
        show_progress : bool, default: True
            Whether to show a tqdm progress bar updated after every epoch.
        report_tuner : bool, default False
            Whether to call the tune.report function for hpo.
            
        Returns
        -------
        metrics : dict
            Dictionary with training and evaluation metrics for all epochs.
            Evaluation metrics will be prefixed with 'val_'.
            The additional key 'lr' specifies the learning rate at the end
            of the respective epoch.
            The training history can be visualized using
            `viz_utils.plot_training_history`.
        """

        if report_tuner:
            from ray import tune
        
        metrics = OrderedDict()
        progbar = trange(epochs, disable=not show_progress)
        for ep in progbar:
            
            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion, **train_args)
    
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
        
        """ Trains the deep learning pipeline on a given dataset.
    
        Parameters
        ----------
        train_data : datasets.common.ImageClassificationDataset
            Training data.
            The `transform` attribute will be set to the data transforms obtained from
            `get_data_transforms` unless `keep_transform` is True.
        val_data : datasets.common.ImageClassificationDataset, optional
            Validation data.
            The `transform` attribute will be set to the data transforms obtained from
            `get_data_transforms` unless `keep_transform` is True.
        batch_size : int
            The batch size.
        epochs : int
            Total number of training epochs.
        architecture : str, default: 'rn50'
            The model architecture to be trained. Note that the pipeline might need to make
            modifications to the standard architecture which are done inside the create_model method.
        init_weights : str, default: None
            The path of the state_dict of the saved model to resume (e.g. /ubuntu/saved_model.pth).
        show_progress : bool, default: True
            Whether to show a tqdm progress bar updated after every epoch.
        show_sub_progress : bool, default: False
            Whether to show a second tqdm progress bar updated after every batch.
        eval_interval : int, default: 1
            Number of epochs after which evaluation will be performed.
        multi_gpu : bool, default: False
            If `True`, model training will be parallelized across all available GPUs.
        load_workers : int, default: 8
            Number of parallel processes used for data loading and pre-processing.
        keep_transform : bool, default: False
            If True, the `transform` attribute of `train_data` and `val_data` will not be modified.
        report_tuner : bool, default: False
            Whether to call the tune.report function inside train_model loop for hpo.
        
        Returns
        -------
        model : torch.nn.Module
            The trained model.
        metrics : dict
            Dictionary with training and evaluation metrics for all epochs.
            Evaluation metrics will be prefixed with 'val_'.
            The additional key 'lr' specifies the learning rate at the end
            of the respective epoch.
            The training history can be visualized using
            `viz_utils.plot_training_history`.
        """
        
        # Set data transforms
        if not keep_transform:
            train_transform, test_transform = self.get_data_transforms(train_data)
            train_data.transform = train_transform
            if val_data is not None:
                val_data.transform = test_transform
    
        # Create data loaders
        train_loader = datautil.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=load_workers, pin_memory=True
        )
        val_loader = datautil.DataLoader(
            val_data, batch_size=batch_size, shuffle=False, num_workers=load_workers, pin_memory=True
        ) if val_data is not None else None
    
        # Create model
        model = self.create_model(architecture, train_data.num_classes, train_data.num_input_channels).cuda()
        
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
                        par_model, train_loader, val_loader, optimizer, criterion, epochs,
                        train_args={ 'scheduler' : scheduler, 'regularizer' : regularizer, 'show_progress' : show_sub_progress },
                        eval_args={ 'show_progress' : show_sub_progress },
                        eval_interval=eval_interval, show_progress=show_progress,
                        report_tuner=report_tuner
                        )
    
        return model, metrics


    def evaluate(self, model: nn.Module, test_data, batch_size: int = 10, print_metrics: bool = False):
        """ Evaluates the model on the given test data

        In contrast to `evaluate_epoch`, this first generates predictions for the
        entire dataset before computing evaluation metrics. This allows for computing
        evaluation metrics that cannot be accumulated as a running average.

        Parameters
        ----------
        model : nn.Module
            Model to be evaluated.
        test_data : gem.datasets.common.ImageClassificationDataset
            The dataset on which the classifier will be evaluated.
        batch_size : int, default: 10
            Batch size to be used for prediction.
        print_metrics : bool, default: False
            Whether to print the computed metrics on the command line.

        Returns
        -------
        metrics : dict
            A dictionary with the following keys:
            - accuracy
            - balanced_accuracy
        """
        
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


    def set_hparams(self, **hparams):
        """ Updates all hyper-parameters given as keyword arguments. """

        self.hparams.update(hparams)


    def get_arch_class(self, arch: str) -> Callable:
        """ Utility function that returns the class of the input architecture. 

        Parameters
        ----------
        arch : str
            The name of the network architecture.
            A list of supported architectures can be obtained from
            `gem.loader.InstanceLoader('architectures').available_instances`.

        Returns
        -------
        callable(*args, **kwargs)
            A callable child class of nn.Module implementing a set of architectures.
        """
        
        return InstanceLoader('architectures').get_class(arch)


    def create_model(self, arch: str, num_classes: int, input_channels: int, config: dict = {}) -> nn.Module:
        """ Instantiates a given neural network architecture for use with this method.

        Parameters
        ----------
        arch : str
            The name of the network architecture.
            A list of supported architectures can be obtained from
            `gem.loader.InstanceLoader('architectures').available_instances`.
        num_classes : int
            The number of classes to be distinguished, i.e., the number of output neurons.
        input_channels : int, default: 3
            The number of input channels.
        config : dict
            A dictionary containing configurations by means of key-word arguments accepted
            by the constructor of the architecture.
        
        Returns
        -------
        torch.nn.Module
            The model to be trained.
            Might yield multiple outputs, which will all be passed to the loss function
            as arguments before the target labels. The first output tensor will be
            considered to contain class scores for accuracy computation.
        """

        return InstanceLoader('architectures').build_instance(arch, num_classes, input_channels, config=config)


    def load_weights(self, model: nn.Module, path: str) -> nn.Module:
        """ Overwrites the input model instance with a saved instance loadable from path.

        Parameters
        ----------
        model : torch.nn.Module
            An instance of a neural network model obtained from `create_model`.
        path : str
            The path of the state_dict of the saved model (e.g. /ubuntu/saved_model.pth)
        
        Returns
        -------
        torch.nn.Module
            The loaded model.
        """
        
        loaded_dict = torch.load(path)
        model.load_state_dict(loaded_dict)
        return model
    
    
    def get_optimizer(self, model: nn.Module, max_epochs: int, max_iter: int) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """ Instantiates an optimizer and learning rate schedule.

        Parameters
        ----------
        model : nn.Module
            The model to be trained.
        max_epochs : int
            The total number of epochs.
        max_iter : int
            The total number of iterations (epochs * batches_per_epoch).
        
        Returns
        -------
        optimizer : torch.optim.Optimizer
        lr_schedule : torch.optim.lr_scheduler._LRScheduler
        """

        optimizer = torch.optim.SGD(model.parameters(), lr=self.hparams['lr'], momentum=0.9, weight_decay=self.hparams['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter)
        return optimizer, scheduler


    def get_regularizer(self) -> Optional[Callable]:
        """ Returns a function for computing a regularization loss over model weights.

        Returns
        -------
        callable(model)
            A function taking the model as sole argument and returning a scalar tensor
            with the regularization loss.
            If this method does not apply regularization, `None` is returned.
        """

        return None


    @abstractmethod
    def get_loss_function(self) -> Callable:
        """ Returns a loss function.
        
        Returns
        -------
        callable(*outputs, targets)
            A loss function taking as arguments all model outputs followed by
            target class labels and returning a single loss value.
        """

        return NotImplemented


    @abstractmethod
    def get_data_transforms(self, dataset) -> Tuple[Callable, Callable]:
        """ Creates data transforms for training and test data.

        Parameters
        ----------
        dataset : gem.datasets.ImageClassificationDataset
            The dataset for which the transform will be used.
        
        Returns
        -------
        train_transform
            The transformation pipeline to be used for training data.
        test_transform
            The transformation pipeline to be used for test/validation data.
        """

        return NotImplemented


    @staticmethod
    @abstractmethod
    def get_pipe_name() -> str:
        """ Returns a string specifying the name of the pipeline. """

        return NotImplemented


    @staticmethod
    def default_hparams() -> dict:
        """ Returns a dictionary specifying the default values for all hyper-parameters supported by this pipeline. """

        return { 'lr' : 0.01, 'weight_decay' : 0.001 }


    def __repr__(self):

        return self.__class__.__name__ + '(' + ', '.join(
            f'{param_name}={param_val}' for param_name, param_val in self.hparams.items()
        ) + ')'


    def __str__(self):

        max_name_len = max(map(len, self.hparams.keys()))

        desc = f'{self.__class__.__name__} with the following hyper-parameters:'
        for param_name, param_value in self.hparams.items():
            desc += f'\n{param_name:>{max_name_len}s}: {param_value}'
        return desc



class BasicAugmentation(LearningMethod):
    """ Extends `LearningMethod` with basic data augmentation.
    
    Hyper-Parameters
    ----------------
    lr : float
        Learning rate.
    weight_decay : float
        Weight decay.
    normalize : bool
        Whether to normalize images using channel mean and standard deviation.
    target_size : int
        Target size for resizing and cropping. If `None`, image size will remain unmodified.
    min_scale : float
        Minimum scaling ratio for scale agumentation.
    max_scale : float
        Maximum scaling ratio for scale agumentation.
    rand_shift : int
        Maximum amount of random translation in pixels.
        This is mutually exclusive with `min_scale` and `max_scale`.
    hflip : bool
        Whether to randomly flip images horizontally.
    vflip : bool
        Whether to randomly flip images vertically.
    """

    def get_data_transforms(self, dataset) -> Tuple[Callable, Callable]:

        transforms = []
        test_transforms = []

        # Check whether data type is PIL image or NumPy array
        # (arrays must be converted to tensors before anything else)
        pil_data = isinstance(dataset[0][0], Image.Image)
        if not pil_data:
            transforms.append(tf.ToTensor())
            test_transforms.append(tf.ToTensor())
        
        # Resize/Crop/Shift
        if self.hparams['target_size'] is not None:
            if self.hparams['rand_shift'] > 0:
                transforms.append(tf.Resize(self.hparams['target_size'] + 2 * self.hparams['rand_shift']))
                transforms.append(tf.RandomCrop(self.hparams['target_size']))
            else:
                transforms.append(tf.RandomResizedCrop(self.hparams['target_size'], scale=(self.hparams['min_scale'], self.hparams['max_scale'])))
            test_transforms.append(tf.Resize(self.hparams['target_size'] + 2 * self.hparams['rand_shift']))
            test_transforms.append(tf.CenterCrop(self.hparams['target_size']))
        elif self.hparams['rand_shift'] > 0:
            img_size = np.asarray(dataset[0][0]).shape[:2]
            transforms.append(tf.RandomCrop(img_size, padding=self.hparams['rand_shift'], padding_mode='reflect'))
        
        # Horizontal/Vertical Flip
        if self.hparams['hflip']:
            transforms.append(tf.RandomHorizontalFlip())
        if self.hparams['vflip']:
            transforms.append(tf.RandomVerticalFlip())
        
        # Convert PIL image to tensor
        if pil_data:
            transforms.append(tf.ToTensor())
            test_transforms.append(tf.ToTensor())
        
        # Channel-wise normalization
        if self.hparams['normalize']:
            channel_mean, channel_std = self.compute_normalization_statistics(dataset)
            norm_trans = tf.Normalize(channel_mean, channel_std)
            transforms.append(norm_trans)
            test_transforms.append(norm_trans)
        
        return tf.Compose(transforms), tf.Compose(test_transforms)


    def compute_normalization_statistics(self, dataset, show_progress=False) -> Tuple[np.ndarray, np.ndarray]:
        """ Computes channel-wise mean and standard deviation for a given dataset.

        Parameters
        ----------
        dataset : gem.datasets.ImageClassificationDataset
            The dataset.
        show_progress : bool
            Whether to show a tqdm progress bar.
        
        Returns
        -------
        mean : np.ndarray
        std : np.ndarray
        """

        # Check whether data type is PIL image or NumPy array
        pil_data = isinstance(dataset[0][0], Image.Image)

        # Create data loader with resize and center crop transform
        transforms = []
        if not pil_data:
            transforms.append(tf.ToTensor())
        if self.hparams['target_size']:
            transforms.append(tf.Resize(self.hparams['target_size']))
            transforms.append(tf.CenterCrop(self.hparams['target_size']))
        if pil_data:
            transforms.append(tf.ToTensor())
        prev_transform = dataset.transform
        dataset.transform = tf.Compose(transforms)
        data_loader = datautil.DataLoader(dataset, batch_size=1000, shuffle=False)
        
        # Compute mean
        num_samples = 0
        channel_mean = 0
        for batch, _ in tqdm(data_loader, desc='Computing mean', disable=not show_progress):
            channel_mean = channel_mean + batch.sum(axis=(0,2,3))
            num_samples += batch.shape[0] * batch.shape[2] * batch.shape[3]
        channel_mean /= num_samples

        # Compute standard deviation
        channel_std = 0
        for batch, _ in tqdm(data_loader, desc='Computing std', disable=not show_progress):
            batch -= channel_mean[None,:,None,None]
            channel_std = channel_std + (batch * batch).sum(axis=(0,2,3))
        channel_std = torch.sqrt(channel_std / num_samples)

        # Restore saved transform
        dataset.transform = prev_transform

        return channel_mean.numpy().copy(), channel_std.numpy().copy()


    @staticmethod
    def default_hparams() -> dict:

        return {
            **super(BasicAugmentation, BasicAugmentation).default_hparams(),
            'normalize' : True,
            'target_size' : None,
            'min_scale' : 1.0,
            'max_scale' : 1.0,
            'rand_shift' : 0,
            'hflip' : True,
            'vflip' : False
        }
