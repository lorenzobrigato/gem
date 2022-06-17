import numpy as np
import tifffile

from gem.datasets.common import ImageClassificationDataset


def tif_loader(fn):
    return tifffile.imread(fn).astype(np.float32)


class EuroSATRGBDataset(ImageClassificationDataset):
    """ RGB images of the EuroSAT dataset for land cover classification.

    Dataset: https://github.com/phelber/eurosat
    Paper: https://arxiv.org/abs/1709.00029

    See `ImageClassificationDataset` for documentation.
    """
    
    def __init__(self, root, split, img_dir='RGB',
                 transform=None, target_transform=None):

        super(EuroSATRGBDataset, self).__init__(
            root=root,
            split=split,
            img_dir='RGB' if img_dir is None else img_dir,
            file_ext='.jpg',
            transform=transform,
            target_transform=target_transform
        )


    @staticmethod
    def get_ds_name():
        
        return 'eurosat_rgb'


class EuroSATMultispectralDataset(ImageClassificationDataset):
    """ Multispectral images of the EuroSAT dataset for land cover classification.

    Dataset: https://github.com/phelber/eurosat
    Paper: https://arxiv.org/abs/1709.00029

    See `ImageClassificationDataset` for documentation.
    """
    
    def __init__(self, root, split, img_dir='allBands',
                 transform=None, target_transform=None):
        
        super(EuroSATMultispectralDataset, self).__init__(
            root=root,
            split=split,
            img_dir='allBands' if img_dir is None else img_dir,
            file_ext='.tif',
            loader=tif_loader,
            transform=transform,
            target_transform=target_transform
        )


    @property
    def num_input_channels(self):

        return 13


    @staticmethod
    def get_ds_name():

        return 'eurosat'
