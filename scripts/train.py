import sys, os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import torch
from collections import OrderedDict

from gem.loader import InstanceLoader


class NegateAction(argparse.Action):
    def __call__(self, parser, ns, values, option):
        setattr(ns, self.dest, option[2:5] != 'no-')


def get_pipeline_hparams(method):
    return InstanceLoader('pipelines').get_class(method).default_hparams()


def get_hparams(method, args):

    hparam_names = set(get_pipeline_hparams(method).keys())
    
    # Standard parameters with dedicated CLI arguments
    hparams = {
        param_name : param_val
        for param_name, param_val in vars(args).items()
        if param_name in hparam_names
    }

    # Additional parameters specified via --param
    if args.param is not None:
        for param_name, *param_val in args.param:
            if param_name not in hparam_names:
                print(f'Hyper-parameter "{param_name}" is not supported by method "{args.method}".')
                exit(-1)
            hparams[param_name] = [str2any(val) for val in param_val] if len(param_val) > 1 else str2any(param_val[0])
    
    return hparams


def str2any(val):

    if val.lower() in ('yes', 'true'):
        return True
    elif val.lower() in ('no', 'false'):
        return False
    else:
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                return val


def list_hparams(method, exclude=[]):

    defaults = get_pipeline_hparams(method)
    max_name_len = max(len(param_name) for param_name in defaults.keys() if param_name not in exclude)

    print(f'Hyper-Parameters for method {method}:\n')
    print(f'{"Parameter":>{max_name_len}s}  Default')
    print(f'{"":-<{max_name_len}s}  -------')
    for param_name, param_default in defaults.items():
        if param_name not in exclude:
            print(f'{param_name:>{max_name_len}s}  {param_default}')


def save_history_csv(fn, metrics, eval_interval=1):

    with open(fn, 'w') as f:
        f.write('epoch,' + ','.join(metrics.keys()) + '\n')
        for ep in range(len(metrics['loss'])):
            f.write(str(ep+1) + ',' + ','.join(
                str(values[(ep+1)//eval_interval if eval_interval > 1 and key.startswith('val_') else ep])
                for key, values in metrics.items()
            ) + '\n')


def save_history_json(fn, metrics, args, hparams):

    with open(fn, 'w') as f:
        json.dump(OrderedDict((
            ('dataset', args.dataset),
            ('method', args.method),
            ('architecture', args.architecture),
            ('hparams', { 'epochs' : args.epochs, 'batch_size' : args.batch_size, **hparams }),
            ('metrics', metrics)
        )), f, indent=4)


if __name__ == '__main__':

    ds_loader = InstanceLoader('datasets')
    pipe_loader = InstanceLoader('pipelines')

    # Parse Arguments
    parser = argparse.ArgumentParser(
        description='Trains a ResNet baseline model on a given dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    group = parser.add_argument_group('Data')
    group.add_argument('dataset', type=str, choices=ds_loader.available_instances(),
                       help='Name of the dataset.')
    group.add_argument('--data-root', type=str, default=None,
                       help='Dataset root directory containing the split files. '\
                            'Defaults to "./datasets/<dataset>".')
    group.add_argument('--img-dir', type=str, default=None,
                       help='The directory containg the images, if different from the default.')
    group.add_argument('--train-split', type=str, default='trainval0',
                       help='File specifying the subset of the data to be used for training.')
    group.add_argument('--test-split', type=str, default='test0',
                       help='File specifying the subset of the data to be used for evaluation.')
    group = parser.add_argument_group('Model')
    group.add_argument('--method', type=str, choices=pipe_loader.available_instances(), default='xent',
                       help='The learning method/pipeline.')
    group.add_argument('--architecture', type=str, choices=InstanceLoader('architectures').available_instances(), default='rn50',
                       help='Network architecture.')
    group.add_argument('--init-weights', type=str, default=None,
                       help='Path to a file containing model weights to be used as initialization.')
    group = parser.add_argument_group('Augmentation')
    group.add_argument('--normalize', '--no-normalize', dest='normalize', action=NegateAction, nargs=0, default=True,
                       help='Normalize images using channel mean and standard deviation.')
    group.add_argument('--target-size', type=int, default=None,
                       help='Target size for resizing and cropping. If not given, image size will remain unmodified.')
    group.add_argument('--min-scale', type=float, default=1.0,
                       help='Minimum scaling ratio for scale agumentation.')
    group.add_argument('--max-scale', type=float, default=1.0,
                       help='Maximum scaling ratio for scale agumentation.')
    group.add_argument('--rand-shift', type=int, default=0,
                       help='Maximum amount of random translation in pixels.')
    group.add_argument('--hflip', '--no-hflip', dest='hflip', action=NegateAction, nargs=0, default=True,
                       help='Toggle random horizontal flipping on/off.')
    group.add_argument('--vflip', '--no-vflip', dest='vflip', action=NegateAction, nargs=0, default=False,
                       help='Toggle random vertical flipping on/off.')
    group = parser.add_argument_group('Training')
    group.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs.')
    group.add_argument('--batch-size', type=int, default=25,
                       help='Batch size.')
    group.add_argument('--multi-gpu', action='store_true', default=False,
                       help='Train on all available GPUs in parallel.')
    group.add_argument('--eval-interval', type=int, default=1,
                       help='Number of epochs between evaluation runs.')
    group.add_argument('--load-workers', type=int, default=8,
                       help='Number of parallel data loading processes.')
    group = parser.add_argument_group('Method Hyper-Parameters')
    group.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate.')
    group.add_argument('--weight-decay', type=float, default=0.001,
                       help='Weight decay.')
    group.add_argument('--param', type=str, action='append', nargs='+', metavar=('NAME', 'VALUE'),
                       help='Name and value of an additional method-specific hyper-parameter. '\
                            'May be specified multiple times. VALUE may be a list of items.'
    )
    group.add_argument('--list-params', action='store_true', default=False,
                       help='Print a list of additional hyper-parameters supported by the given method and exit.')
    group = parser.add_argument_group('Output')
    group.add_argument('--save', type=str, default=None,
                       help='Filename under which the final model weights will be stored.')
    group.add_argument('--history', type=str, default=None,
                       help='Path to a CSV or JSON file where metrics for each training epoch will be written to. '\
                            'If the filename ends on ".json", the file will also contain the hyper-parameters used for training.')
    args = parser.parse_args()

    if args.data_root is None:
        args.data_root = os.path.join('datasets', 'cifair' if args.dataset.startswith('cifair') else args.dataset)
    
    if (args.min_scale != 1 or args.max_scale != 1) and (args.target_size is None):
        print('Specifying --target-size is required when --min-scale or --max-scale are given.')
        exit()

    if args.rand_shift > 0 and (args.min_scale != 1 or args.max_scale != 1):
        print('--rand-shift and --min-scale/--max-scale are mutually exclusive.')
        exit()

    # Set up hyper-parameters
    if args.list_params:
        list_hparams(args.method, set(vars(args).keys()))
        exit()
    hparams = get_hparams(args.method, args)
    print(f'Method: {args.method}')
    print(f'Hyper-parameters: {hparams}')

    # Load dataset
    data_kwargs= {}
    if args.img_dir is not None:
        data_kwargs['img_dir'] = args.img_dir

    train_data = ds_loader.build_instance(args.dataset, args.data_root, args.train_split, **data_kwargs)
    test_data = ds_loader.build_instance(args.dataset, args.data_root, args.test_split, **data_kwargs)

    # Instantiate training pipeline
    pipeline = pipe_loader.build_instance(args.method, **hparams)

    # Train model
    model, metrics = pipeline.train(
        train_data, test_data,
        architecture=args.architecture,
        init_weights=args.init_weights,
        batch_size=args.batch_size,
        epochs=args.epochs,
        eval_interval=args.eval_interval,
        multi_gpu=args.multi_gpu,
        load_workers=args.load_workers
    )

    # Evaluate the model
    final_metrics = pipeline.evaluate(model, test_data, print_metrics=True)

    # Save model
    if args.save:
        torch.save(model.state_dict(), args.save)
    
    # Save history
    if args.history:
        if args.history.lower().endswith('.json'):
            metrics['balanced_accuracy'] = final_metrics['balanced_accuracy']
            save_history_json(args.history, metrics, args, hparams)
        else:
            save_history_csv(args.history, metrics, args.eval_interval)
