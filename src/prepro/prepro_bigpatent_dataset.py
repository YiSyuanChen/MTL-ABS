""" Parses the raw BIGPATENT dataset.

There are two sources of raw datasets for BIGPATENT:
- From official page: https://drive.google.com/uc?export=download&id=1J3mucMFTWrgAYa3LuBZoLRR3CzzYD3fa (bigPatentData.tar.gz)
- From TFDS : https://drive.google.com/uc?export=download&confirm=fb2s&id=1mwH7eSh1kNci31xduR4Da_XcmTE8B8C3 (bigPatentDataNonTokenized.tar.gz)
We use the dataset from TFDS as default to compete the results with previous work.

Before runing this code:
1. Untar the downloaded main file (bigPatentData.tar.gz, bigPatentDataNonTokenized.tar.gz)
2. Untar the split files in the folder (test.tar.gz, val.tar.gz, train.tar.gz)

"""
import os
import argparse
import gzip
from multiprocess import Pool
import tensorflow as tf
import tensorflow_datasets as tfds
import spacy
import numpy as np
from tqdm import tqdm, trange
from os.path import join as pjoin
import json

import pdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_path", type=str, required=True)
    parser.add_argument("-output_path", type=str, required=True)
    parser.add_argument("-ssplit_target",
                        type=str,
                        required=True,
                        choices=['both', 'art', 'sum', 'none'])
    parser.add_argument("-data_src",
                        type=str,
                        default='tfds',
                        choices=['official', 'tfds'])
    parser.add_argument("-cpu_num", type=int, default=8)
    args = parser.parse_args()

    # For sentence seperation
    nlp = spacy.load("en_core_web_lg")

    if (args.data_src == 'official'):
        args.input_path = pjoin(args.input_path, 'bigPatentData')
    else:
        args.input_path = pjoin(args.input_path, 'bigPatentDataNonTokenized')

    # Create output directory
    split_names = ['test', 'valid', 'train']
    for split_name in split_names:
        if not os.path.isdir(os.path.join(args.output_path, split_name)):
            os.makedirs(os.path.join(args.output_path, split_name))
    split_names = ['test', 'val', 'train']

    # CPC code list
    cpc_code_list = ["a", "b", "c", "d", "e", "f", "g", "h", "y"]

    for split_name in split_names:
        idx = 0
        if (split_name == 'test'):
            d_path = pjoin(args.input_path, 'test')
        if (split_name == 'val'):
            d_path = pjoin(args.input_path, 'val')
        if (split_name == 'train'):
            d_path = pjoin(args.input_path, 'train')

        idx = 0
        for cpc_code in cpc_code_list:
            file_names = os.listdir(
                os.path.join(args.input_path, split_name, cpc_code))
            print("Processing files in {} ...".format(
                pjoin(split_name, cpc_code)))
            for file_name in tqdm(file_names):
                with gzip.open(
                        os.path.join(args.input_path, split_name, cpc_code,
                                     file_name), 'r') as f_i:
                    for row in f_i:
                        json_obj = json.loads(row)
                        art = json_obj['description']
                        summ = json_obj['abstract']

                        # Make output folder 'valid' consistent
                        if (split_name == 'val'):
                            out_split_name = 'valid'
                        else:
                            out_split_name = split_name

                        o_path = pjoin(args.output_path, out_split_name,
                                       str(idx) + '.txt')
                        with open(o_path, 'w') as f_o:
                            if args.ssplit_target != 'none':

                                # Remove Original Seperation
                                if (args.ssplit_target == 'both'
                                        or args.ssplit_target == 'art'):
                                    art = art.replace('\n', ' ').replace(
                                        '\t', ' ').replace('\p', ' ').strip()
                                if (args.ssplit_target == 'both'
                                        or args.ssplit_target == 'sum'):
                                    summ = summ.replace('\n', ' ').replace(
                                        '\t', ' ').replace('\p', ' ').strip()

                                # Split article and write
                                if (args.ssplit_target == 'both'
                                        or args.ssplit_target == 'art'):
                                    doc = nlp(art)
                                    doc_list = list(doc.sents)
                                    doc = "\n".join(
                                        str(sent) for sent in doc_list)
                                    f_o.write(doc)
                                else:
                                    f_o.write(art)

                                # Split summary and write (with '@highlight' in begin)
                                if (args.ssplit_target == 'both'
                                        or args.ssplit_target == 'sum'):
                                    doc = nlp(summ)
                                    doc_list = list(doc.sents)
                                    doc = "\n".join(
                                        str(sent) for sent in doc_list)
                                    f_o.write('\n@highlight\n')
                                    f_o.write(doc)
                                else:
                                    f_o.write('\n@highlight\n')
                                    f_o.write(summ)
                            else:
                                f_o.write(art)
                                f_o.write('\n@highlight\n')
                                f_o.write(summ)
                            idx += 1
