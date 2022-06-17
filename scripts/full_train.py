import sys, os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
from ray import tune

from gem.ray_classifier import ray_train_classifier
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

    # Additional fixed parameters specified via --param
    if args.param is not None:
        for param_name, *param_val in args.param:
            if param_name not in hparam_names:
                print(f'Hyper-parameter "{param_name}" is not supported by method "{args.method}".')
                exit(-1)
            hparams[param_name] = [str2any(val) for val in param_val] if len(param_val) > 1 else str2any(param_val[0])
    
    # Additional tuned parameters specified via --param-uniform
    if args.param_uniform is not None:
        for param_name, lower_bound, upper_bound in args.param_uniform:
            if param_name not in hparam_names:
                print(f'Hyper-parameter "{param_name}" is not supported by method "{args.method}".')
                exit(-1)
            hparams[param_name] = tune.uniform(float(lower_bound), float(upper_bound))
    
    # Additional tuned parameters specified via --param-loguniform
    if args.param_loguniform is not None:
        for param_name, lower_bound, upper_bound in args.param_loguniform:
            if param_name not in hparam_names:
                print(f'Hyper-parameter "{param_name}" is not supported by method "{args.method}".')
                exit(-1)
            hparams[param_name] = tune.loguniform(float(lower_bound), float(upper_bound))
    
    # Additional tuned parameters specified via --param-choice
    if args.param_choice is not None:
        for param_name, *param_vals in args.param_choice:
            if param_name not in hparam_names:
                print(f'Hyper-parameter "{param_name}" is not supported by method "{args.method}".')
                exit(-1)
            hparams[param_name] = tune.choice([str2any(val) for val in param_vals])
    
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


if __name__ == '__main__':

    # Parse Arguments
    parser = argparse.ArgumentParser(
        description='Performs hyper-parameter tuning using asynchronous successive halving and final training with the found configuration.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    group = parser.add_argument_group('Data')
    group.add_argument('dataset', type=str, choices=InstanceLoader('datasets').available_instances(),
                       help='Name of the dataset.')
    group.add_argument('--data-root', type=str, default=None,
                       help='Dataset root directory containing the split files. '\
                            'Defaults to "./datasets/<dataset>".')
    group.add_argument('--img-dir', type=str, default=None,
                       help='The directory containg the images, if different from the default.')
    group.add_argument('--train-split', type=str, default='train0',
                       help='File specifying the subset of the data to be used for training.')
    group.add_argument('--test-split', type=str, default='val0',
                       help='File specifying the subset of the data to be used for evaluation.')
    group = parser.add_argument_group('Model')
    group.add_argument('--method', type=str, choices=InstanceLoader('pipelines').available_instances(), default='xent',
                       help='The learning method/pipeline.')
    group.add_argument('--architecture', type=str, default='rn50',
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
    group = parser.add_argument_group('HPO Training')
    group.add_argument('--epochs', type=int, default=200,
                       help='Maximum number of epochs.')
    group.add_argument('--batch-sizes', type=int, nargs='+', default=[10, 25, 50],
                       help='List of batch sizes to try.')
    group.add_argument('--min-lr', type=float, default=1e-4,
                       help='Lower bound for learning rate.')
    group.add_argument('--max-lr', type=float, default=0.1,
                       help='Upper bound for learning rate.')
    group.add_argument('--min-decay', type=float, default=1e-5,
                       help='Lower bound for weight decay.')
    group.add_argument('--max-decay', type=float, default=0.1,
                       help='Upper bound for weight decay.')
    group.add_argument('--metric', type=str, choices=['loss', 'loss_avg5', 'accuracy', 'accuracy_avg5'], default='accuracy_avg5',
                       help='The metric to be optimized.')
    group.add_argument('--num-trials', type=int, default=100,
                       help='Number of trial runs with different hyper-parameters.')
    group.add_argument('--grace-period', type=int, default=10,
                       help='Minimum number of epochs before stopping a trial.')
    group.add_argument('--reduction-factor', type=int, default=2,
                       help='Halving rate and amount.')
    group.add_argument('--cpus-per-trial', type=float, default=8,
                       help='Number of CPUs to be allocated to each trial.')
    group.add_argument('--gpus-per-trial', type=float, default=1.0,
                       help='Number of GPUs to be allocated to each trial. '\
                            'Note that fractional GPUs (e.g., 0.25) are fine, as long as you have sufficient memory.')
    group = parser.add_argument_group('Final Training')
    group.add_argument('--train-split-f', type=str, default='trainval0',
                       help='File specifying the subset of the data to be used for final training.')
    group.add_argument('--test-split-f', type=str, default='test0',
                       help='File specifying the subset of the data to be used for final evaluation.')
    group.add_argument('--num-trials-f', type=int, default=10,
                       help='Number of trials repeated for final training.'\
                       'Each trial is performed using the parameters configuration found with HPO.')
    group.add_argument('--epochs-f', type=int, default=None,
                       help='Number of epochs for the final training.'\
                       'If not specified, the script will execute the maximum epochs specified for HPO training.')
    group.add_argument('--eval-interval-f', type=int, default=1,
                       help='Number of epochs between evaluation runs in the final training.')
    group.add_argument('--cpus-per-trial-f', type=int, default=None,
                       help='Number of CPUs to be allocated to each trial.'\
                       'If not specified, the script will use the same number of CPUs used for HPO training.')
    group.add_argument('--gpus-per-trial-f', type=int, default=None,
                       help='Number of GPUs to be allocated to each trial.'\
                       'If not specified, the script will use the same number of GPUs used for HPO training.')
    group.add_argument('--save-f', type=str, default=None,
                       help='Filename under which the the weights of the final models will be stored. '\
                            'Because of multi-processing, a pattern representing the unique timestamp will be added rigth before the file extensions.'\
                            'E.g., with input string: /home/model.pth, the corresponding name will be: /home/model_{TIME}.pth')
    group.add_argument('--history-f', type=str, default=None,
                       help='Path to a CSV or JSON file where metrics for each training epoch will be written to. '\
                            'Because of multi-processing, a pattern representing the unique timestamp will be added rigth before the file extensions.'\
                            'E.g., with input string: /home/history.json, the corresponding name will be: /home/history_{TIME}.json')
    group = parser.add_argument_group('Method Hyper-Parameters')
    group.add_argument('--param', type=str, action='append', nargs='+', metavar=('NAME', 'VALUE'),
                       help='Name and value of an additional method-specific hyper-parameter. '\
                            'May be specified multiple times. VALUE may be a list of items.')
    group.add_argument('--param-uniform', type=str, action='append', nargs=3, metavar=('NAME', 'LOWER', 'UPPER'),
                       help='Name, lower and upper bound of an additional, method-specific hyper-parameter that will be tuned on a uniform scale. '\
                            'May be specified multiple times.')
    group.add_argument('--param-loguniform', type=str, action='append', nargs=3, metavar=('NAME', 'LOWER', 'UPPER'),
                       help='Name, lower and upper bound of an additional, method-specific hyper-parameter that will be tuned on a loguniform scale. '\
                            'May be specified multiple times.')
    group.add_argument('--param-choice', type=str, action='append', nargs='+', metavar=('NAME', 'VALUE'),
                       help='Name and possible values of an additional, method-specific hyper-parameter that will be tuned. '\
                            'May be specified multiple times.')
    group = parser.add_argument_group('Output')
    group.add_argument('--log-dir', type=str, default=None,
                       help='Directory where trial logs will be stored. Defaults to ~/ray_results.')
    group.add_argument('--exp-name', type=str, default=None,
                       help='The name of the experiment. Default uses date and other information.')
    group.add_argument('--checkpoint-at-end', type=bool, default=False,
                       help='Whether to save checkpoints at the end of HPO training. Needed for eventual restoring/re-run of trials. Default to False.')
    group.add_argument('--checkpoint-at-end-f', type=bool, default=False,
                       help='Whether to save checkpoints at the end of final training. Needed for eventual restoring/re-run of trials. Default to False.')
    group.add_argument('--resume', type=str, choices=['LOCAL', 'REMOTE', 'PROMPT', 'ERRORED_ONLY'], default=None,
                       help='Whether to resume the HPO training. The checkpoint is loaded from the local_checkpoint_dir which is '\
                       'determined by exp-name and log-dir. Default to None.')
    group.add_argument('--resume-f', type=str, choices=['LOCAL', 'REMOTE', 'PROMPT', 'ERRORED_ONLY'], default=None,
                       help='Whether to resume a training experiment. The checkpoint is loaded from the local_checkpoint_dir which is '\
                       'determined by exp-name and log-dir. Default to None.')
    args = parser.parse_args()

    if args.data_root is None:
        args.data_root = os.path.join('datasets', 'cifair' if args.dataset.startswith('cifair') else args.dataset)
    
    if args.epochs_f is None:
        args.epochs_f = args.epochs
    
    if args.cpus_per_trial_f is None:
        args.cpus_per_trial_f = args.cpus_per_trial
        
    if args.gpus_per_trial_f is None:
        args.gpus_per_trial_f = args.gpus_per_trial
        
    if (args.min_scale != 1 or args.max_scale != 1) and (args.target_size is None):
        print('Specifying --target-size is required when --min-scale or --max-scale are given.')
        exit()

    if args.rand_shift > 0 and (args.min_scale != 1 or args.max_scale != 1):
        print('--rand-shift and --min-scale/--max-scale are mutually exclusive.')
        exit()
    
        
    # Set up HPO config
    config = {
        'lr': tune.loguniform(args.min_lr, args.max_lr),
        'decay': tune.loguniform(args.min_decay, args.max_decay) if args.max_decay > 0 else 0,
        'batch_size': tune.choice(args.batch_sizes),
        'epochs': args.epochs,
        
        'pipeline': args.method,
        'architecture': args.architecture,
        'init_weights': args.init_weights,
        'hparams': get_hparams(args.method, args),

        'dataset': args.dataset,
        'data_root': os.path.abspath(args.data_root),
        'train_split': args.train_split,
        'test_split': args.test_split,

        'load_workers': max(1, int(args.cpus_per_trial)),
    }
    if args.img_dir is not None:
        config['img_dir'] = args.img_dir
    metric_mode = 'max' if args.metric.startswith('accuracy') else 'min'

    # Set up scheduler
    scheduler = tune.schedulers.ASHAScheduler(
        metric=args.metric,
        mode=metric_mode,
        max_t=args.epochs,
        grace_period=args.grace_period,
        reduction_factor=args.reduction_factor
    )

    # Set up reporter
    hparam_names = { 'lr' : 'lr', 'decay' : 'decay' }
    if len(args.batch_sizes) > 1:
        hparam_names['batch_size'] = 'batch_size'
    if args.param_uniform is not None:
        for param_name, *_ in args.param_uniform:
            hparam_names['hparams/' + param_name] = param_name
    if args.param_loguniform is not None:
        for param_name, *_ in args.param_loguniform:
            hparam_names['hparams/' + param_name] = param_name
    if args.param_choice is not None:
        for param_name, *_ in args.param_choice:
            hparam_names['hparams/' + param_name] = param_name
    reporter = tune.CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"],
        parameter_columns=hparam_names,
        print_intermediate_tables=True,
        metric=args.metric,
        mode=metric_mode
    )

    # Run HPO
    result = tune.run(
        ray_train_classifier,
        name=args.exp_name,
        resources_per_trial={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
        config=config,
        num_samples=args.num_trials,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=args.log_dir,
        checkpoint_at_end=args.checkpoint_at_end,
        resume=args.resume,
        verbose=1,
        queue_trials=True
    )

    # Print best result
    best_trial = result.get_best_trial(args.metric, metric_mode, 'last')
    print("Best trial config:")
    for key, name in hparam_names.items():
        val = best_trial.config
        for subkey in key.split('/'):
            val = val[subkey]
        print('    {}: {}'.format(name, val))
    print()
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {:.2%}".format(best_trial.last_result["accuracy"]))


    # Set up final training config
    config = best_trial.config
    config['train_split'] = args.train_split_f
    config['test_split'] = args.test_split_f
    config['epochs'] = args.epochs_f
    config['load_workers'] = max(1, int(args.cpus_per_trial_f))
    config['eval_interval'] = args.eval_interval_f
    config['save'] = args.save_f
    config['history'] = args.history_f


    # Set up reporter
    hparam_names = { 'lr' : 'lr', 'decay' : 'decay',  'batch_size' : 'batch_size'}

    if args.param_uniform is not None:
        for param_name, *_ in args.param_uniform:
            hparam_names['hparams/' + param_name] = param_name
    if args.param_loguniform is not None:
        for param_name, *_ in args.param_loguniform:
            hparam_names['hparams/' + param_name] = param_name
    if args.param_choice is not None:
        for param_name, *_ in args.param_choice:
            hparam_names['hparams/' + param_name] = param_name
    reporter = tune.CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"],
        parameter_columns=hparam_names,
        print_intermediate_tables=True,
        metric=args.metric,
        mode=metric_mode
    )
    
    # Run final training
    result = tune.run(
        ray_train_classifier,
        name=args.exp_name,
        resources_per_trial={"cpu": args.cpus_per_trial_f, "gpu": args.gpus_per_trial_f},
        config=config,
        num_samples=args.num_trials_f,
        progress_reporter=reporter,
        local_dir=args.log_dir,
        checkpoint_at_end=args.checkpoint_at_end_f,
        resume=args.resume_f,
        verbose=1,
        queue_trials=True
    )
    