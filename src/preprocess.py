""" Entry file for data preprocessing. """
import time
import argparse
from others.logging import init_logger
from others.utils import str2bool
from prepro import data_builder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-pretrained_model", default='bert', type=str)

    parser.add_argument("-mode", default='', type=str)
    parser.add_argument("-select_mode", default='greedy', type=str)
    parser.add_argument("-map_path", default='../../data/')
    parser.add_argument("-raw_path", default='../../line_data')
    parser.add_argument("-save_path", default='../../data/')

    parser.add_argument("-shard_size", default=2000, type=int)
    parser.add_argument('-min_src_nsents', default=0, type=int)
    parser.add_argument('-max_src_nsents', default=100, type=int)
    parser.add_argument('-min_src_ntokens_per_sent', default=5, type=int)
    parser.add_argument('-max_src_ntokens_per_sent', default=200, type=int)
    parser.add_argument('-max_src_ntokens', default=-1, type=int)
    parser.add_argument('-min_tgt_ntokens', default=0, type=int)
    parser.add_argument('-max_tgt_ntokens', default=500, type=int)

    parser.add_argument("-lower",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=True)
    parser.add_argument("-use_bert_basic_tokenizer",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False)

    parser.add_argument('-log_file', default='prepro.log')
    parser.add_argument('-dataset', default='')
    parser.add_argument("-limit_num_file_tokenized", default=-1, type=int)
    parser.add_argument('-n_cpus', default=8, type=int)

    parser.add_argument("-debug",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False)

    args = parser.parse_args()
    init_logger(args.log_file)
    eval('data_builder.' + args.mode + '(args)')
