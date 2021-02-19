""" Create meta dataset. """
import os
from os.path import join as pjoin
import argparse

from utils import str2bool
import pdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_path",
                        default="../../bert_data",
                        type=str,
                        required=True)
    parser.add_argument("-output_path",
                        default="../../bert_meta_data",
                        type=str,
                        required=True)
    parser.add_argument("-abbrev", default='', type=str, required=True)
    parser.add_argument("-train_dataset_list",
                        default='',
                        type=str,
                        required=True)
    parser.add_argument("-valid_dataset_list",
                        default='',
                        type=str,
                        required=True)
    parser.add_argument("-max_train_pt_files",
                        default=1,
                        type=int,
                        required=True)
    parser.add_argument("-max_valid_pt_files",
                        default=1,
                        type=int,
                        required=True)
    args = parser.parse_args()

    train_dataset_list = args.train_dataset_list.split(',')
    valid_dataset_list = args.valid_dataset_list.split(',')
    meta_dataset_name = "meta_data_" + args.abbrev

    # Create folders
    train_sup_dir = pjoin(args.output_path, meta_dataset_name, 'train',
                          'support')
    train_qry_dir = pjoin(args.output_path, meta_dataset_name, 'train',
                          'query')
    valid_sup_dir = pjoin(args.output_path, meta_dataset_name, 'valid',
                          'support')
    valid_qry_dir = pjoin(args.output_path, meta_dataset_name, 'valid',
                          'query')

    if not os.path.isdir(pjoin(args.output_path, meta_dataset_name)):
        for dataset in train_dataset_list:
            os.makedirs(pjoin(train_sup_dir, dataset))
            os.makedirs(pjoin(train_qry_dir, dataset))
        for dataset in valid_dataset_list:
            os.makedirs(pjoin(valid_sup_dir, dataset))
            os.makedirs(pjoin(valid_qry_dir, dataset))

    # Create meta training set
    for dataset in train_dataset_list:
        dataset_path = pjoin(args.input_path, dataset)
        pt_files = os.listdir(dataset_path)
        pt_files = [
            pt for pt in pt_files if os.path.isfile(pjoin(dataset_path, pt))
        ]
        train_pt_files = [pt for pt in pt_files if 'train' in pt]

        train_sup_count = 0
        train_qry_count = 0

        flag = True
        for idx, train_pt_file in enumerate(train_pt_files):
            if (flag):
                from_path = pjoin(dataset_path, train_pt_file)
                to_path = pjoin(train_sup_dir, dataset, train_pt_file)
                os.system('cp ' + from_path + ' ' + to_path)
                train_sup_count += 1
            else:
                from_path = pjoin(dataset_path, train_pt_file)
                to_path = pjoin(train_qry_dir, dataset, train_pt_file)
                os.system('cp ' + from_path + ' ' + to_path)
                train_qry_count += 1
            flag = not flag

            # Control data number
            if (idx == args.max_train_pt_files - 1):
                break

        print("Dataset : {}".format(dataset))
        print("train_sup {} | train_qry {}".format(train_sup_count,
                                                   train_qry_count))

    # Create meta validation set
    for dataset in valid_dataset_list:
        dataset_path = pjoin(args.input_path, dataset)
        pt_files = os.listdir(dataset_path)
        pt_files = [
            pt for pt in pt_files if os.path.isfile(pjoin(dataset_path, pt))
        ]
        train_pt_files = [pt for pt in pt_files if 'train' in pt]

        valid_sup_count = 0
        valid_qry_count = 0

        flag = True
        for idx, valid_pt_file in enumerate(
                train_pt_files):  # NOTE: also use training data
            if (flag):
                from_path = pjoin(dataset_path, valid_pt_file)
                to_path = pjoin(valid_sup_dir, dataset, valid_pt_file)
                os.system('cp ' + from_path + ' ' + to_path)
                valid_sup_count += 1
            else:
                from_path = pjoin(dataset_path, valid_pt_file)
                to_path = pjoin(valid_qry_dir, dataset, valid_pt_file)
                os.system('cp ' + from_path + ' ' + to_path)
                valid_qry_count += 1
            flag = not flag

            # Control data number
            if (idx == args.max_valid_pt_files - 1):
                break

        print("Dataset : {}".format(dataset))
        print("valid_sup {} | valid_qry {}".format(valid_sup_count,
                                                   valid_qry_count))
