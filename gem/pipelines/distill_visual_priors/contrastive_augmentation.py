import numpy as np
import torch
import torch.utils.data as datautil
import torchvision.transforms as tf
from PIL import Image, ImageFilter
import random

from typing import Tuple, Callable

from gem.utils import is_notebook

if is_notebook():
    from tqdm.notebook import tqdm, trange
else:
    from tqdm import tqdm, trange
    
from gem.pipelines.common import LearningMethod

    
        
class ContrastiveAugmentation(LearningMethod):
    """ Extends `LearningMethod` with contrastive data augmentation.
        In particular, MoCo v2's augmentation.
    
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
    prob_jitter: float
        Probability of applying random jitter.
    prob_grayscale: float
        Probability of transforming images to grayscale.
    prob_gaussblur: float
        Probability of applying Gaussian blur.
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
        
        # Can not use full contrastive augmentation on multi-spectral images
        if dataset.num_input_channels == 1 or dataset.num_input_channels == 3:
            # Color Jitter
            transforms.append(tf.RandomApply([
                    tf.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=self.hparams['prob_jitter']))
            
            # Grayscale
            transforms.append(tf.RandomGrayscale(p=self.hparams['prob_grayscale']))
            
            # Gaussian Blur
            transforms.append(tf.RandomApply([GaussianBlur([.1, 2.])], p=self.hparams['prob_gaussblur']))
        
        
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
        dataset : small_data.datasets.ImageClassificationDataset
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
            **super(ContrastiveAugmentation, ContrastiveAugmentation).default_hparams(),
            'normalize' : True,
            'target_size' : None,
            'min_scale' : 0.2,
            'max_scale' : 1.0,
            'rand_shift' : 0,
            'hflip' : True,
            'vflip' : False,
            'prob_jitter': 0.8,
            'prob_grayscale': 0.2,
            'prob_gaussblur': 0.5
        }


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
