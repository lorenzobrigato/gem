from PIL import Image
from gem.datasets.common import ImageClassificationDataset


def grayscale_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')


class CLaMMDataset(ImageClassificationDataset):
    """ CLaMM Classification of Latin Medieval Manuscripts.

    Dataset: https://clamm.irht.cnrs.fr/icfhr2016-clamm/data-set/
    Paper: https://journal.digitalmedievalist.org/articles/10.16995/dm.61/

    See `ImageClassificationDataset` for documentation.
    """

    CLASSNAMES = ['Caroline', 'Cursiva', 'Half_uncial', 'Humanistic',
                  'Humanistic_Cursive', 'Hybrida', 'Praegothica',
                  'Semihybrida', 'Semitextualis', 'Southern_Textualis',
                  'Textualis', 'Uncial']
    
    def __init__(self, root, split, img_dir='images',
                 transform=None, target_transform=None):

        super(CLaMMDataset, self).__init__(
            root=root,
            split=split,
            img_dir='images' if img_dir is None else img_dir,
            file_ext='.tif',
            loader=grayscale_loader,
            transform=transform,
            target_transform=target_transform
        )


    @property
    def num_input_channels(self):

        return 1


    @staticmethod
    def get_ds_name():
        return 'clamm'
