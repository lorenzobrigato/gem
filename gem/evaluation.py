import numpy as np
import torch
from torch import nn, Tensor
from torch.utils import data as datautil
from multiprocessing import Pool

from typing import Tuple, List, Callable, Union, Optional


def predict_class_scores(model: nn.Module, data, transform: Optional[Callable] = None, batch_size: int = 10, softmax: bool = False) -> np.ndarray:
    """ Predict classes on given data using a given model.

    Parameters
    ----------
    model : torch.nn.Module
        The trained classification model. If the model has multiple outputs, the first
        one will be interpreted as class scores.
    data : small_data.datasets.common.ImageClassificationDataset
        The data to make predictions for.
    transform : callable, optional
        Data transform to be used.
        If `None`, the existing transform of `data` will be used.
    batch_size : int, default: 10
        Batch size to be used for prediction.
    softmax : bool, default: False
        Wether to cast the raw class scores to pseudo-probabilities using the softmax function.
    
    Returns
    -------
    class_scores : np.ndarray
        An N x C numpy array, where N is the number of samples and C the number of classes.
    """

    if transform is not None:
        prev_transform = data.transform
        data.transform = transform

    loader = datautil.DataLoader(
        data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
    )

    model.eval()
    predictions = []
    
    with torch.no_grad():
        for X, _ in loader:
            output = model(X.cuda())
            if isinstance(output, tuple):
                output = output[0]
            if softmax:
                output = torch.softmax(output, 1)
            predictions.append(output.cpu().numpy())
    
    if transform is not None:
        data.transform = prev_transform

    return np.concatenate(predictions)


def confusion_matrix(model: nn.Module, test_data, batch_size: int = 10, norm: bool = False) -> np.ndarray:
    """ Computes the confusion matrix for a given model on given test data.

    Parameters
    ----------
    model : torch.nn.Module
        The trained classification model. If the model has multiple outputs, the first
        one will be interpreted as class scores.
    test_data : small_data.datasets.common.ImageClassificationDataset
        The dataset on which the classifier will be evaluated.
    batch_size : int, default: 10
        Batch size to be used for prediction.
    norm : bool, default: False
        If `True`, the confusion counts will be normalized so that the entries
        in each *row* sum up to 1.
    
    Returns
    -------
    confusion_matrix : np.ndarray
        An C x C array, where C is the number of classes.
    """

    gt = np.asarray(test_data.targets)
    pred = predict_class_scores(model, test_data, batch_size=batch_size).argmax(axis=-1)
    cf = np.zeros((test_data.num_classes, test_data.num_classes))
    np.add.at(cf, (gt, pred), 1)
    if norm:
        cf /= cf.sum(axis=1, keepdims=True)
    return cf


def balanced_accuracy(model: nn.Module, test_data, batch_size: int = 10) -> float:
    """ Evaluates the balanced classification accuracy of a given model on given data.

    Balanced classification accuracy is defined as the average per-class accuracy,
    i.e., the average of the diagonal in the confusion matrix.

    Parameters
    ----------
    model : torch.nn.Module
        The trained classification model. If the model has multiple outputs, the first
        one will be interpreted as class scores.
    test_data : small_data.datasets.common.ImageClassificationDataset
        The dataset on which the classifier will be evaluated.
    batch_size : int, default: 10
        Batch size to be used for prediction.
    
    Returns
    -------
    balanced_accuracy : float
    """

    cf = confusion_matrix(model, test_data, batch_size=batch_size, norm=True)
    return np.diag(cf).mean()


def balanced_accuracy_from_predictions(y_true: Union[List[int], np.ndarray], y_pred: Union[List[int], np.ndarray]) -> float:
    """ Evaluates the balanced classification accuracy of given predictions.

    Balanced classification accuracy is defined as the average per-class accuracy,
    i.e., the average of the diagonal in the confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray or list of int
        True class indices.
    y_pred : np.ndarray or list of int
        Predicted class indices.
    
    Returns
    -------
    balanced_accuracy : float
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    class_sizes = np.bincount(y_true)
    tp_per_class = np.bincount(y_pred[y_pred == y_true], minlength=len(class_sizes))
    return np.mean(tp_per_class / class_sizes)


def bootstrap_significance_test(
        model1: nn.Module,
        model2: nn.Module,
        test_data,
        transform1: Optional[Callable] = None,
        transform2: Optional[Callable] = None,
        alpha: float = 0.05,
        num_samples: int = 1000000,
        batch_size: int = 10
    ) -> Tuple[bool, Tuple[float, float]]:
    """ Tests whether the performance difference between two classification models is significant using the bootstrap.

    This implementation follows the procedure for testing significance employed by
    the PASCAL VOC challenge and described here:
    http://host.robots.ox.ac.uk/pascal/VOC/pubs/bootstrap_note.pdf

    Classification performance is measured in terms of balanced accuracy.

    Parameters
    ----------
    model1 : torch.nn.Module
        The reference model. If the model has multiple outputs, the first
        one will be interpreted as class scores.
    model2 : torch.nn.Module
        The model to be compared with the reference. If the model has
        multiple outputs, the first one will be interpreted as class scores.
    test_data : small_data.datasets.common.ImageClassificationDataset
        The dataset on which the classifiers will be evaluated.
    transform1 : callable, optional
        Data transform to be used for the first model.
        If `None`, the existing transform of `test_data` will be used.
    transform2 : callable, optional
        Data transform to be used for the second model.
        If `None`, the existing transform of `test_data` will be used.
    alpha : float, default: 0.05
        The significance level to be tested.
    num_samples : int, default: 1000000
        The number of bootstrap samples. Higher values increase robustness
        but also computation time.
    batch_size : int, default: 10
        Batch size to be used for prediction.

    Returns
    -------
    is_significant: bool
        True if the performance difference between the reference and the comparison
        model is significant on a level of `alpha`, i.e., 0 is not contained in
        the `1 - alpha` confidence interval.
    confidence_interval: (float, float) tuple
        The bootstrapped `1 - alpha` confidence interval of
        `balanced_accuracy(model1) - balanced_accuracy(model2)`.
    accuracies: (float, float) tuple
        Balanced accuracies of `model1` and `model2`.
    """

    # Obtain model predictions
    pred1 = predict_class_scores(model1, test_data, transform=transform1, batch_size=batch_size).argmax(axis=-1)
    pred2 = predict_class_scores(model2, test_data, transform=transform2, batch_size=batch_size).argmax(axis=-1)
    targets = np.asarray(test_data.targets)

    # Draw bootstrap samples of balanced accuracy
    with Pool(initializer=_init_bootstrap_pool, initargs=(pred1, pred2, targets)) as pool:
        differences = pool.map(_get_bootstrap_sample, range(num_samples))
    
    # Calculate confidence interval and check significance
    low, high = np.quantile(differences, [alpha / 2, 1 - alpha / 2])
    is_significant = (low <= 0 <= high)

    # Compute balanced accuracy on the original dataset
    acc1 = balanced_accuracy_from_predictions(targets, pred1)
    acc2 = balanced_accuracy_from_predictions(targets, pred2)

    return is_significant, (low, high), (acc1, acc2)


def _init_bootstrap_pool(_pred1, _pred2, _targets):
    global pred1, pred2, targets
    pred1 = _pred1
    pred2 = _pred2
    targets = _targets
    np.random.seed()


def _get_bootstrap_sample(i):
    indices = np.random.choice(len(pred1), len(pred1), replace=True)
    acc1 = balanced_accuracy_from_predictions(targets[indices], pred1[indices])
    acc2 = balanced_accuracy_from_predictions(targets[indices], pred2[indices])
    return acc1 - acc2
