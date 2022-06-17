import os.path
import numpy as np
import torch
import torchvision


class ImageClassificationDataset(torchvision.datasets.vision.VisionDataset):
    """ Generic superclass for classification datasets.

    Parameters
    ----------
    root : str
        Root directory of the dataset.
    split : str
        Path to a textfile listing the files to include in this dataset
        as well as their labels. Relative paths will be considered relative
        to `root`. If no file extension is given, ".txt" will be appended.
        The file should list one image per file, without the file extension,
        and the index of its label, separated by a space.
        Relative paths will be considered relative to `img_dir`.
    img_dir : str, default=''
        The directory where the images reside. Relative paths will be
        considered relative to `root`.
    file_ext : str, default: '.jpg'
        The file extension that will be appended to all image paths listed in
        `split`.
    loader : callback, optional
        Custom function for loading images given their path.
    transform : callback, optional
        A function/transform that takes in a PIL image or NumPy array and returns
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
    samples : list of (str, int) tuples
        List of all image filenames and their class index.
    targets : list of int
        List of the class indices of all samples.
    """
    
    def __init__(self, root, split, img_dir='',
                 file_ext='.jpg', loader=torchvision.datasets.folder.pil_loader,
                 transform=None, target_transform=None):

        super(ImageClassificationDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.loader = loader
        
        if len(os.path.splitext(split)[1]) == 0:
            split += '.txt'
        split = split if os.path.isabs(split) else os.path.join(self.root, split)
        img_dir = img_dir if os.path.isabs(img_dir) else os.path.join(self.root, img_dir)
        
        self._load_sample_list(split, img_dir, file_ext)

        self.targets = [s[1] for s in self.samples]


    def _load_sample_list(self, split_file, img_dir, file_ext):
        """ Loads the list of samples in this dataset and their label from a split file.
        
        This method initializes `self.samples`, `self.classes`, and `self.class_to_idx`.
        It is called by the constructor and can be overridden in sub-classes to handle
        split files with a custom file format.
        
        Arguments
        ---------
        split : str
            Path to a textfile listing the files to include in this dataset
            as well as their labels.
            The file should list one image per file, without the file extension,
            and the index of its label, separated by a space.
            Relative paths will be considered relative to `img_dir`.
        img_dir : str
            The directory where the images reside.
        file_ext : str
            The file extension that will be appended to all image paths listed in
            `split_file`.
        """

        samples = []
        classes = []
        class_to_idx = {}

        with open(split_file) as f:
            for line in f:
                if line.strip() != '':
                    fn, lbl = line.strip().split()
                    lbl = int(lbl)
                    samples.append((
                        os.path.join(img_dir, fn if (file_ext is None) or fn.endswith(file_ext) else fn + file_ext),
                        lbl
                    ))
                    classname = self._get_class_name(fn, lbl)
                    class_to_idx[classname] = lbl
                    if len(classes) < lbl + 1:
                        classes += [None] * (lbl + 1 - len(classes))
                    classes[lbl] = classname

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples


    def __getitem__(self, index):
        """ Loads and returns the sample with the given index and its label. """

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


    def __len__(self):
        """ The number of samples in the dataset. """

        return len(self.samples)


    @property
    def num_classes(self):
        """ The number of different classes in the dataset. """

        return len(self.classes)

    @property
    def num_input_channels(self):
        """ The number of input channels for this dataset. """

        return 3


    def _get_class_name(self, filename, class_idx):
        """ Determines the name of the class for a given sample.

        By default, this function uses the class attribute `CLASSNAMES`
        to get the name for the given `class_idx`, if that attribute exists.
        Otherwise, it returns the beginning of the filename
        until the first slash. Subclasses may override this to implement
        dataset-specific behaviour or a fixed list of classnames.

        Parameters
        ----------
        filename : str
            The path to the file.
        class_idx : int
            The index of the class.

        Returns
        -------
        str
        """

        try:
            return self.CLASSNAMES[class_idx]
        except:
            return filename.split('/')[0]
