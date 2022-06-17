import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from collections.abc import Sequence


def plot_training_history(metrics, eval_interval=1, smooth=15, val_smooth=15):
    """ Plots the training history of a model.

    Parameters
    ----------
    metrics : dict
        Dictionary with training and evaluation metrics as returned by `LearningMethod.train`.
    eval_interval : int, default: 1
        Number of epochs between two evaluation runs.
    smooth : int, default: 15
        Window size for smoothing training metrics.
    val_smooth : int, default: 15
        Window size for smoothing validation metrics.
    """

    metric_names = [key for key, values in metrics.items() if key != 'lr' and (not key.startswith('val_')) and isinstance(values, Sequence)]

    fig, axes = plt.subplots(1, len(metric_names), figsize=(12, 6))

    for metric_name, ax in zip(metric_names, axes):
        epochs = np.arange(1, len(metrics[metric_name])+1)
        if eval_interval <= 1:
            epochs_val = epochs
        else:
            epochs_val = np.arange(0, len(metrics[metric_name])+1, eval_interval)
            epochs_val[0] = 1

        if smooth > 2:
            ax.plot(epochs, metrics[metric_name], color='#91aec2', linewidth=1, label=metric_name)
        if val_smooth > 2:
            ax.plot(epochs_val, metrics['val_' + metric_name], color='#ffc898', linewidth=1, label='val_' + metric_name)
        if smooth > 2:
            ax.plot(epochs, scipy.ndimage.convolve1d(metrics[metric_name], np.ones(smooth) / smooth), color='#1f77b4', label=metric_name + ' (smoothed)')
        else:
            ax.plot(epochs, metrics[metric_name], color='#1f77b4', label=metric_name)
        if val_smooth > 2:
            ax.plot(epochs_val, scipy.ndimage.convolve1d(metrics['val_' + metric_name], np.ones(val_smooth) / val_smooth), color='#ff7f0e', label='val_' + metric_name + ' (smoothed)')
        else:
            ax.plot(epochs_val, metrics['val_' + metric_name], color='#ff7f0e', label='val_' + metric_name)
        
        if metric_name == 'loss':
            ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid()
    
    fig.tight_layout()
    plt.show()
