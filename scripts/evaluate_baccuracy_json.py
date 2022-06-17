import argparse
import glob
import json
from numpy import mean, std
from collections import OrderedDict


def save_json(fn, results):

    with open(fn, 'w') as f:
        json.dump(OrderedDict((
            ('accs', results['accs']),
            ('mean', results['mean']),
            ('std', results['std']),
        )), f, indent=4)


def load_json(fname):
    
    with open(fname) as json_file:
        data = json.load(json_file)
    return data


if __name__ == '__main__':

    # Parse Arguments
    parser = argparse.ArgumentParser(
        description='Compute the average balanced accuracy and standard deviation from N repetitions of the same method on a given dataset (reading from JSON files).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('pathname', type=str,
                        help='A string in path-style containing a path specification for JSON files. '\
                             'The string is fed to glob.glob to find all matching files (e.g., /home/classifier*.json).')
    parser.add_argument('--save', type=str, default=None,
                        help='File name (JSON) under which the results are saved. '\
                            'Precisely the values from all repetitions, the mean and standard deviation.')
        
    args = parser.parse_args()
    
    
    # Read accuracy from files
    files = glob.glob(args.pathname)
    
    if len(files) == 0:
        print(f'No files found matching {args.pathname}')
        exit()
        
    accs = []
    d = {}
    for f in files:
        f_dict = load_json(f)
        accs.append(f_dict['metrics']['balanced_accuracy'] * 100)
    
    d['accs'] = accs
    d['mean'] = mean(accs)
    d['std'] = std(accs)
    
    print('Average balanced accuracy over {} reps: {:.2f} \u00B1 {:.1f}'.format((len(files)),(d['mean']), (d['std'])))
    
    if args.save is not None:
        save_json(args.save, d)
        print(f'Saved results to {args.save}')
    