"""Getting params from the command line."""

import argparse
import math
from texttable import Texttable

def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a high performance model without grid search.
    """
    parser = argparse.ArgumentParser(description = 'Show description')

    parser.add_argument('-data', '--dataset', type = str,
                        help = 'which dataset to run', default = 'uci')
    parser.add_argument('-b', '--batch_size', type= int,
                        help = 'batch_size', default = 200)
    parser.add_argument('-l', '--learning_rate', type = float,
                        help = 'learning_rate', default = 0.001)
    parser.add_argument('-nn', '--num_negative', type = int,
                        help = 'num_negative', default = 5)
    parser.add_argument('-tr', '--train_ratio', type = float,
                        help = 'train_ratio', default = 0.8)
    parser.add_argument('-vr', '--valid_ratio', type = float,
                        help = 'valid_ratio', default = 0.01)
    parser.add_argument('-act', '--act', type = str,
                        help = 'act function', default = 'tanh')
    parser.add_argument('-trans', '--transfer', type = int,
                        help = 'transfer to head, tail representations', default = 1)
    parser.add_argument('-dp' , '--drop_p', type = float,
                        help = 'dropout_rate', default = 0)
    parser.add_argument('-ip', '--if_propagation', type = int,
                        help = 'if_propagation', default=1)
    parser.add_argument('-ia', '--is_att', type = int,
                        help = 'use attention or not', default=1)
    parser.add_argument('-w', '--w', type = float,
                        help = 'w for decayer', default = 2)
    parser.add_argument('-s', '--seed', type = int,
                        help = 'random seed', default = 0)
    parser.add_argument('-rp', '--reset_rep', type = int,
                        help = 'whether reset rep', default = 1)
    parser.add_argument('-dc', '--decay_method', type = str,
                        help = 'decay_method', default = 'log')
    parser.add_argument('-nor', '--nor', type = int ,
                        help = 'normalize or not', default = 0)
    parser.add_argument('-iu', '--if_updated', type = int,
                        help = 'use updated representation in loss', default = 0)
    parser.add_argument('-wd', '--weight_decay', type = float,
                        help = 'weight decay', default = 0.001)
    parser.add_argument('-nt', '--if_no_time', type = int,
                        help = 'if no time interval information', default = 0)
    parser.add_argument('-th', '--threhold', type = float,
                        help = 'the threhold to filter the neighbors, if None, do not filter', default = None)
    parser.add_argument('-2hop', '--second_order', type = int,
                        help = 'whether to use 2-hop prop', default = 0)

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