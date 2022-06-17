import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import warnings
import torch
import time
import json
from collections import OrderedDict

from gem.loader import InstanceLoader
from train import save_history_csv


def update_fname(filename, pattern):
    """ Function which updtaes a filename in the general format /{BASE_PATH}/{NAME}.{EXTENSION} into
        /{BASE_PATH}/{NAME}_{PATTERN}.{EXTENSION}
    
    Parameters
    ----------
    filename : string
        The name of the file.
    pattern : string
        Pattern to be added to the file name.
    """
    
    bname = os.path.basename(filename)
    dirname = os.path.dirname(filename)
    
    newbname = bname.split('.')[0] + '_' + pattern + '.' + bname.split('.')[1]
    
    return os.path.join(dirname, newbname)
    

def save_history_json(fn, metrics, config, hparams):

    with open(fn, 'w') as f:
        json.dump(OrderedDict((
            ('dataset', config['dataset']),
            ('method', config['pipeline']),
            ('architecture', config['architecture']),
            ('hparams', {'epochs' : config['epochs'], 
                         'batch_size' : config['batch_size'],
                         'lr' : config['lr'],
                         'weight_decay' : config['decay'],
                         **hparams }),
            ('metrics', metrics)
        )), f, indent=4)
        
        
def ray_train_classifier(config):
    """ Training function to be passed to `ray.tune.run`.

    Parameters
    ----------
    config : dict
        Dictionary specifying the configuration of this training run. Contains the following items:
            - 'dataset': The name of the dataset to be passed to `InstanceLoader('datasets').build_instance`.
            - 'data_root': The root directory of the dataset.
            - 'img_dir' (optional): The directory containing the images.
            - 'train_split' (optional): File specifying the subset of the data to be used for training.
                                        Default: 'train'
            - 'test_split' (optional): File specifying the subset of the data to be used for evaluation.
                                       Default: 'val'
            - 'pipeline' (optional): Name of the training pipeline. A list of supported values
                                     can be obtained from `InstanceLoader('pipelines').available_instances`.
                                     Default: 'CrossEntropyClassifier'
            - 'architecture' (optional): Name of the base architecture. A list of supported values
                              can be obtained from `InstanceLoader('architectures').available_instances`.
                              Default: 'rn50'
            - 'init_weights' (optional): Path to a checkpoint to initialize the model with.
            - 'hparams' (optional): A dictionary with additional fixed pipeline-specific hyper-parameters.
            - 'transform' (optional): Custom training data transform to be used instead of the one
                                      provided by the pipeline.
            - 'test_transform' (optional): Custom validation data transform to be used instead of the one
                                           provided by the pipeline.
            - 'load_workers' (optional): Number of parallel data loading processes.
            - 'batch_size': The batch size.
            - 'epochs': The maximum number of training epochs.
            - 'lr': The initial learning rate.
            - 'decay': The weight decay.
            - 'eval_interval' (optional): The evaluation interval in terms of epochs.
                                          Default: 1
            - 'save' (optional): Filename under which the final model weights will be stored.
                                 Notice that a time pattern is added to the filename. 
            - 'history' (optional): Path to a CSV or JSON file where metrics will be written to.
                                    If the file is JSON, a final evaluation using 
                                    `small_data.methods.common.LearningMethod.evaluate` will be performed.
    """

    warnings.filterwarnings('ignore', category=UserWarning)

    # Instantiate pipeline
    pipe_loader = InstanceLoader('pipelines')
    pipeline = pipe_loader.build_instance(
        config.get('pipeline', 'xent'),
        lr=config['lr'],
        weight_decay=config['decay'],
        **config.get('hparams', {})
    )

    # Load dataset
    data_kwargs = {}
    if 'img_dir' in config:
        data_kwargs['img_dir'] = config['img_dir']
    ds_loader = InstanceLoader('datasets')
    train_data = ds_loader.build_instance(config['dataset'], config['data_root'], config.get('train_split', 'train'), **data_kwargs)
    val_data = ds_loader.build_instance(config['dataset'], config['data_root'], config.get('test_split', 'val'), **data_kwargs)

    # Set custom data transforms
    if (config.get('transform') is not None) or (config.get('test_transform') is not None):
        if (config.get('transform') is None) or (config.get('test_transform') is None):
            train_transform, test_transform = pipeline.get_data_transforms(train_data)
        if config.get('transform') is not None:
            train_transform = config['transform']
        if config.get('test_transform') is not None:
            test_transform = config['test_transform']
        train_data.transform = train_transform
        val_data.transform = test_transform
        custom_transforms = True
    else:
        custom_transforms = False

    # Train model
    model, metrics = pipeline.train(
        train_data, val_data,
        architecture=config.get('architecture', 'rn50'),
        init_weights=config.get('init_weights'),
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        show_progress=False,
        eval_interval=config.get('eval_interval', 1),
        load_workers=config.get('load_workers', 8),
        keep_transform=custom_transforms,
        report_tuner=True
    )

    # Set time pattern for updating the file name of model/history
    if (config.get('save') is not None) or (config.get('history') is not None):
        t = time.time()
        pattern = str(t).split('.')[0] + str(t).split('.')[1]
    
    # Save model
    if config.get('save') is not None:
        save = update_fname(config['save'], pattern)
        torch.save(model.state_dict(), save)
    
    # Save history
    if config.get('history') is not None:
        history = update_fname(config['history'], pattern)
        if config['history'].lower().endswith('.json'):
            final_metrics = pipeline.evaluate(model, val_data, print_metrics=False)
            metrics['balanced_accuracy'] = final_metrics['balanced_accuracy']
            save_history_json(history, metrics, config, config.get('hparams', {}))
        else:
            save_history_csv(history, metrics, config.get('eval_interval', 1))
            