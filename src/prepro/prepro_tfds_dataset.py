"""Downloads and processes datasets from TFDS. """
import os
import argparse
from multiprocess import Pool
from functools import partial

import tensorflow_datasets as tfds
import spacy
from tqdm import tqdm
import pdb

from utils import str2bool


def _reseperate_sentence(target, input_path):
    """ Sperates the sentence with Spacy model. """

    dir_path = os.path.split(input_path)[0]
    dir_name = os.path.split(dir_path)[1]
    dir_path = os.path.split(dir_path)[0]
    file_name = os.path.split(input_path)[1]
    output_path = os.path.join(dir_path + '_ssplit', dir_name, file_name)

    with open(input_path, 'r') as f_i, open(output_path, 'w') as f_o:
        content = f_i.read()

        # Seperate the article and summary
        sep = content.find('\n@highlight\n')
        art = content[:sep]
        summ = content[sep:].replace('\n@highlight\n', '')

        # Split article and write
        if target == 'art' or target == 'both':
            # Remove Original Seperation
            art = art.replace('\n', ' ').replace('\t', ' ').replace('\p',
                                                                    ' ').strip()
            doc = nlp(art)
            doc_list = list(doc.sents)
            doc = "\n".join(str(sent) for sent in doc_list)
        else:
            doc = art
        f_o.write(doc)

        # Split summary and write (with '@highlight' in begin)
        if target == 'summ' or target == 'both':
            # Remove Original Seperation
            summ = summ.replace('\n', ' ').replace('\t', ' ').replace('\p',
                                                                      ' ').strip()
            doc = nlp(summ)
            doc_list = list(doc.sents)
            doc = "\n".join(str(sent) for sent in doc_list)
        else:
            doc = summ
        f_o.write('\n@highlight\n')
        f_o.write(doc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode",
                        type=str,
                        required=True,
                        choices=['download', 'ssplit'])
    parser.add_argument("-dataset", default="", type=str)
    parser.add_argument("-output_path", type=str, required=True)
    parser.add_argument("-ssplit_target",
                        default="none",
                        type=str,
                        choices=['both', 'art', 'sum', 'none'])
    parser.add_argument("-art_feature_name", default="", type=str)
    parser.add_argument("-summ_feature_name", default="", type=str)
    parser.add_argument("-beam_dataset",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False)
    parser.add_argument("-cpu_num", default=8, type=int)
    args = parser.parse_args()

    # Sanity Check
    if (args.mode == 'download'
            and (args.dataset == "" or args.art_feature_name == ""
                 or args.summ_feature_name == "")):
        raise Exception(
            "Please specify the dataset and feature name to download.")
    if (args.mode == 'ssplit' and args.ssplit_target == "none"):
        raise Exception(
            "Please specify the target (both, art or sum) for sentence split.")

    # For sentence seperation
    nlp = spacy.load("en_core_web_lg")

    # Build tfds dataset instance
    dataset = tfds.builder(args.dataset)

    print(dataset.info)
    split_names = list(dataset.info.splits.keys())
    data_nums = [n.num_examples for n in list(dataset.info.splits.values())]
    contents = [args.art_feature_name, args.summ_feature_name]

    if not args.beam_dataset:
        # Download the data, prepare it, and write it to disk
        dataset.download_and_prepare()

    # Load data from disk as tf.data.Datasets
    dataset = dataset.as_dataset()

    # Create output directory
    args.output_path = os.path.join(args.output_path, args.dataset)
    for split_name in split_names:
        if not os.path.isdir(os.path.join(
                args.output_path, split_name)) and args.mode == 'download':
            os.makedirs(os.path.join(args.output_path, split_name))
        if not os.path.isdir(
                os.path.join(args.output_path + '_ssplit',
                             split_name)) and args.mode == 'ssplit':
            os.makedirs(os.path.join(args.output_path + '_ssplit', split_name))

    # Download dataset
    if args.mode == 'download':
        for split_name, data_num in zip(split_names, data_nums):
            split_dataset = dataset[split_name]

            for file_id, data in enumerate(
                    tqdm(split_dataset,
                         total=data_num,
                         desc=split_name,
                         unit=' file')):

                # Exceptional constrol: only process abstractive data for newsroom
                if (args.dataset != 'newsroom'
                        or (args.dataset == 'newsroom'
                            and data['density_bin'].numpy().decode('utf-8') ==
                            'abstractive')):
                    with open(
                            os.path.join(args.output_path, split_name,
                                         str(file_id) + '.txt'),
                            'w') as out_file:
                        # Write article
                        art = data[contents[0]].numpy().decode('utf-8')
                        out_file.write(art)

                        # Write summary
                        summ = data[contents[1]].numpy().decode('utf-8')
                        out_file.write('\n@highlight\n')
                        out_file.write(summ)

    # Sentence split for dataset
    if args.mode == 'ssplit':
        for split_name, data_num in zip(split_names, data_nums):
            files = os.listdir(os.path.join(args.output_path, split_name))
            file_list = [
                os.path.join(args.output_path, split_name, f) for f in files
            ]
            parallel_func = partial(_reseperate_sentence, args.ssplit_target)
            pool = Pool(args.cpu_num)
            for d in tqdm(pool.imap(parallel_func, file_list),
                          total=data_num,
                          desc=split_name,
                          unit=' file'):
                pass
            pool.close()
            pool.join()
