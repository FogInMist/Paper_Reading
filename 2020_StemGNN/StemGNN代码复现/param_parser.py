"""Getting params from the command line."""

import argparse
import math
from texttable import Texttable

def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a high performance model without grid search.
    """
    parser = argparse.ArgumentParser(description="StemGNN")

    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--dataset', type=str, default='ECG_data')
    parser.add_argument('--window_size', type=int, default=12)
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--train_length', type=float, default=7)
    parser.add_argument('--valid_length', type=float, default=2)
    parser.add_argument('--test_length', type=float, default=1)
    parser.add_argument('--epoch', type=int, default=2) # 50
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--multi_layer', type=int, default=5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--validate_freq', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--norm_method', type=str, default='z_score')
    parser.add_argument('--optimizer', type=str, default='RMSProp')
    parser.add_argument('--early_stop', type=bool, default=False)
    parser.add_argument('--exponential_decay_step', type=int, default=5)
    parser.add_argument('--decay_rate', type=float, default=0.5)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--leakyrelu_rate', type=int, default=0.2)

    return parser.parse_args(args=[])

def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    rows = [["Parameter", "Value"]]
    for i in [[k.replace("_", " ").capitalize(), args[k]] for k in keys]:
        rows.append(i)
    # print(rows)
    t.add_rows(rows)
    print(t.draw())