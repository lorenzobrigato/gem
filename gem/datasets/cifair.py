""" ciFAIR data loaders for PyTorch.

Version: 1.0

https://cvjena.github.io/cifair/
"""

import numpy as np
import torchvision.datasets


class ciFAIR10(torchvision.datasets.CIFAR10):
    """ ciFAIR10 dataset.

    Dataset: https://cvjena.github.io/cifair/
    Paper: https://arxiv.org/abs/1902.00423

    Parameters
    ----------
    root : str
        Root directory of the dataset.
    split : str
        One of the following values specifying the dataset split ({i} represents the split id) to be loaded:
            - 'train{i}': Subsampled training data (30 images per class).
            - 'val{i}': Validation subset of the original training data (20 images per class).
            - 'trainval{i}': 'train' and 'val' combined (50 images per class).
            - 'fulltrain': The original training set (50,000 images in total).
            - 'test0': The original test set (10,000 images in total).
    
    transform : callback, optional
        A function/transform that takes in a PIL image and returns
        a transformed version. E.g, `torchvision.transforms.RandomCrop`.
    target_transform : callback, optional
        A function/transform that takes in the target and transforms it.
    
    Attributes
    ----------
    num_classes : int
        Number of different classes in the dataset.
    num_input_channels : int
        Number of input channels.
    classes : list of str
        Class names.
    class_to_idx : dict
        Dictionary mapping class names to consecutive numeric indices.
    data : np.ndarray
        N x 3 x 32 x 32 array with image data.
    targets : list of int
        List of the class indices of all samples.
    """
    base_folder = 'ciFAIR-10'
    url = 'https://github.com/cvjena/cifair/releases/download/v1.0/ciFAIR-10.zip'
    filename = 'ciFAIR-10.zip'
    tgz_md5 = 'ca08fd390f0839693d3fc45c4e49585f'
    test_list = [
        ['test_batch', '01290e6b622a1977a000eff13650aca2'],
    ]

    def __init__(self, root, split, transform=None, target_transform=None, download=True):

        super(ciFAIR10, self).__init__(
            root,
            train=(split != 'test0'),
            transform=transform,
            target_transform=target_transform,
            download=download
        )

        if split in ('train0', 'val0', 'trainval0'):
            class_members = { i : [] for i in range(len(self.classes)) }
            for idx, lbl in enumerate(self.targets):
                class_members[lbl].append(idx)
            start = 0 if split.startswith('train') else 30
            end = 30 if split == 'train0' else 50
            indices = np.concatenate([mem[start:end] for mem in class_members.values()])
            self.data = self.data[indices]
            self.targets = np.asarray(self.targets)[indices]

        elif split in ('train1', 'val1', 'trainval1'):
            self.indistrib_lbls = list(range(len(self.classes)))
            class_members = { i : [] for i in range(len(self.classes)) }
            for idx, lbl in enumerate(self.targets):
                class_members[lbl].append(idx)
            start = 50 if split.startswith('train') else 80
            end = 80 if split == 'train1' else 100
            indices = np.concatenate([mem[start:end] for mem in class_members.values()])
            self.data = self.data[indices]
            self.targets = np.asarray(self.targets)[indices]
            
        elif split in ('train2', 'val2', 'trainval2'):
            self.indistrib_lbls = list(range(len(self.classes)))
            class_members = { i : [] for i in range(len(self.classes)) }
            for idx, lbl in enumerate(self.targets):
                class_members[lbl].append(idx)
            start = 100 if split.startswith('train') else 130
            end = 130 if split == 'train2' else 150
            indices = np.concatenate([mem[start:end] for mem in class_members.values()])
            self.data = self.data[indices]
            self.targets = np.asarray(self.targets)[indices]


    @property
    def num_classes(self):
        """ The number of different classes in the dataset. """

        return len(self.classes)


    @property
    def num_input_channels(self):
        """ The number of input channels for this dataset. """

        return 3


    @staticmethod
    def get_ds_name():
        
        return 'cifair10'
