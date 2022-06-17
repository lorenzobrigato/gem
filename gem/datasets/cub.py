from gem.datasets.common import ImageClassificationDataset


class CUBDataset(ImageClassificationDataset):
    """ Caltech-UCSD Birds-200-2011 dataset.

    Dataset: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
    Paper: https://authors.library.caltech.edu/27452/1/CUB_200_2011.pdf

    See `ImageClassificationDataset` for documentation.
    """
    
    def __init__(self, root, split, img_dir='CUB_200_2011/images',
                 transform=None, target_transform=None):

        super(CUBDataset, self).__init__(
            root=root,
            split=split,
            img_dir='CUB_200_2011/images' if img_dir is None else img_dir,
            transform=transform,
            target_transform=target_transform
        )


    def _get_class_name(self, filename, class_idx):

        return filename.split('/')[0].split('.', maxsplit=1)[1].replace('_', ' ')

        
    @staticmethod
    def get_ds_name():
        return 'cub'
