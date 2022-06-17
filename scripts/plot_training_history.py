import sys, os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

from gem.viz_utils import plot_training_history


def read_metrics_csv(csv_file, eval_interval=1):

    metrics = {}
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for metric in reader.fieldnames:
            if metric != 'epoch':
                metrics[metric] = []
        for l in reader:
            for metric, val in l.items():
                ep = int(l['epoch'])
                if (metric in metrics) and ((not metric.startswith('val_')) or ep == 1 or ep % eval_interval == 0):
                    metrics[metric].append(float(val))
    return metrics


def read_metrics_json(json_file):

    with open(json_file) as f:
        metrics = json.load(f)['metrics']
    
    if 'val_loss' not in metrics:
        eval_interval = 0
    elif len(metrics['val_loss']) == len(metrics['loss']):
        eval_interval = 1
    else:
        eval_interval = len(metrics['loss']) // (len(metrics['val_loss']) - 1)
    
    return metrics, eval_interval


if __name__ == '__main__':

    # Parse Arguments
    parser = argparse.ArgumentParser(
        description='Plots metrics collected during a run of train.py.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('history', type=str,
                       help='Path to the history CSV or JSON file generated by train.py.')
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='Number of epochs between evaluation runs. '\
                             'Only required if history is saved as CSV file and the evaluation interval '
                             'during training was different than 1.')
    parser.add_argument('--smooth-train', type=int, default=15,
                        help='Window size for smoothing training metrics.')
    parser.add_argument('--smooth-val', type=int, default=15,
                        help='Window size for smoothing validation metrics.')
    args = parser.parse_args()

    # Create plot
    if args.history.lower().endswith('.json'):
        metrics, eval_interval = read_metrics_json(args.history)
    else:
        eval_interval = args.eval_interval
        metrics = read_metrics_csv(args.history, eval_interval)
    plot_training_history(metrics, eval_interval=eval_interval, smooth=args.smooth_train, val_smooth=args.smooth_val)
