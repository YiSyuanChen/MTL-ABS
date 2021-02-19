import os
from os.path import join as pjoin
import gc
import re
import glob
import json
import random
import subprocess
import itertools
from collections import Counter
from multiprocess import Pool

import torch
from transformers import XLNetTokenizer

from others.logging import logger
from prepro.utils import clean
from prepro.utils import _get_word_ngrams
from others.tokenization import BertTokenizer

import pdb

##################################
##          Tokenize            ##
##################################


def tokenize_tfds(args):
    """ Tokenizes for TFDS datasert.

    Tokenizes raw txt files with Stanford CoreNLP toolkit.
    """
    txt_dir = os.path.abspath(args.raw_path)
    txt_sub_dirs = os.listdir(args.raw_path)
    tokenized_dir = os.path.abspath(args.save_path)

    # Make directory
    for sub_dir in txt_sub_dirs:
        if not os.path.isdir(pjoin(tokenized_dir, sub_dir)):
            os.makedirs(os.path.join(tokenized_dir, sub_dir))

    for sub_dir in txt_sub_dirs:
        # Path
        txt_sub_dir = pjoin(txt_dir, sub_dir)
        tokenized_sub_dir = pjoin(tokenized_dir, sub_dir)
        print("Preparing to tokenize %s to %s..." %
              (txt_sub_dir, tokenized_sub_dir))

        # Make IO list file
        stories = os.listdir(txt_sub_dir)
        print("Making list of files to tokenize...")
        with open("mapping_for_corenlp.txt", "w") as f:
            for file_num, s in enumerate(stories):
                if (not s.endswith('txt')):
                    continue
                f.write("%s\n" % (os.path.join(txt_sub_dir, s)))

                if (file_num == args.limit_num_file_tokenized - 1):
                    break

        command = [
            'java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators',
            'tokenize,ssplit', '-ssplit.newlineIsSentenceBreak', 'always',
            '-filelist', 'mapping_for_corenlp.txt', '-outputFormat', 'json',
            '-outputDirectory', tokenized_sub_dir
        ]
        print("Tokenizing %i files in %s and saving in %s..." %
              (len(stories), txt_sub_dir, tokenized_sub_dir))
        subprocess.call(command)
        print("Stanford CoreNLP Tokenizer has finished.")
        os.remove("mapping_for_corenlp.txt")

        # Check that the tokenized stories directory contains the same number of files as the original directory
        num_orig = len(os.listdir(txt_sub_dir))
        num_tokenized = len(os.listdir(tokenized_sub_dir))
        if num_orig != num_tokenized and args.limit_num_file_tokenized == -1:
            raise Exception(
                """ The tokenized stories directory %s contains %i files,""" +
                """ but it should contain the same number as %s (which has %i files).""" +
                """ Was there an error during tokenization?"""
                % (tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
        print("Successfully finished tokenizing %s to %s.\n" %
              (txt_sub_dir, tokenized_sub_dir))


##################################
##      Dataset Analysis        ##
##################################


def dataset_analysis_tfds(args):
    in_dir = os.path.abspath(args.raw_path)
    in_sub_dirs = os.listdir(args.raw_path)
    out_dir = os.path.abspath(args.save_path)

    # Make directory
    for sub_dir in in_sub_dirs:
        if not os.path.isdir(pjoin(out_dir, sub_dir)):
            os.makedirs(os.path.join(out_dir, sub_dir))

    for sub_dir in in_sub_dirs:
        num_src_sent = 0
        num_tgt_sent = 0
        num_src_token = 0
        num_tgt_token = 0
        min_src_sent = 1000000
        max_src_sent = 0
        min_src_token = 1000000
        max_src_token = 0
        min_tgt_sent = 1000000
        max_tgt_sent = 0
        min_tgt_token = 1000000
        max_tgt_token = 0
        min_src_token_per_sent = 1000000
        max_src_token_per_sent = 0
        min_tgt_token_per_sent = 1000000
        max_tgt_token_per_sent = 0

        # Path
        in_sub_dir = pjoin(in_dir, sub_dir)
        out_sub_dir = pjoin(out_dir, sub_dir)
        print("Analyse data in %s and output log to %s..." %
              (in_sub_dir, out_sub_dir))

        files = os.listdir(in_sub_dir)
        for f in files:
            art_summ = 0
            file_src_sent = 0
            file_tgt_sent = 0
            file_src_token = 0
            file_tgt_token = 0
            abs_path = pjoin(in_dir, sub_dir, f)
            print("Analyse file {} ...".format(abs_path))
            with open(abs_path, 'r') as f_i:
                data = json.load(f_i)
                for sent in data['sentences']:
                    if (sent['tokens'][0]['word'] == '@highlight'):
                        art_summ = 1
                        continue
                    if (art_summ == 0):
                        sent_src_token = len(sent['tokens'])
                        if (sent_src_token > max_src_token_per_sent):
                            max_src_token_per_sent = sent_src_token
                        if (sent_src_token < min_src_token_per_sent):
                            min_src_token_per_sent = sent_src_token

                        file_src_sent += 1
                        file_src_token += len(sent['tokens'])
                        num_src_sent += 1
                        num_src_token += len(sent['tokens'])
                    else:
                        sent_tgt_token = len(sent['tokens'])
                        if (sent_tgt_token > max_tgt_token_per_sent):
                            max_tgt_token_per_sent = sent_tgt_token
                        if (sent_tgt_token < min_tgt_token_per_sent):
                            min_tgt_token_per_sent = sent_tgt_token

                        file_tgt_sent += 1
                        file_tgt_token += len(sent['tokens'])
                        num_tgt_sent += 1
                        num_tgt_token += len(sent['tokens'])

            if (file_src_sent > max_src_sent):
                max_src_sent = file_src_sent
            if (file_src_sent < min_src_sent):
                min_src_sent = file_src_sent
            if (file_src_token > max_src_token):
                max_src_token = file_src_token
            if (file_src_token < min_src_token):
                min_src_token = file_src_token
            if (file_tgt_sent > max_tgt_sent):
                max_tgt_sent = file_tgt_sent
            if (file_tgt_sent < min_tgt_sent):
                min_tgt_sent = file_tgt_sent
            if (file_tgt_token > max_tgt_token):
                max_tgt_token = file_tgt_token
            if (file_tgt_token < min_tgt_token):
                min_tgt_token = file_tgt_token

            if (art_summ == 0):
                raise Exception(
                    'The file does not contain art/summ seperate word @highlight !'
                )

        avg_num_src_sent = num_src_sent / len(files)
        avg_num_tgt_sent = num_tgt_sent / len(files)
        avg_num_src_token = num_src_token / len(files)
        avg_num_tgt_token = num_tgt_token / len(files)
        print('=== Src ===')
        print('Max sent : {:>5d}  Min sent  : {:>5d}   Avg sent  : {}'.format(
            max_src_sent, min_src_sent, avg_num_src_sent))
        print('Max token: {:>5d}  Min token : {:>5d}   Avg token : {}'.format(
            max_src_token, min_src_token, avg_num_src_token))
        print('Max sent token : {:>5d}  Min sent token : {:>5d}'.format(
            max_src_token_per_sent, min_src_token_per_sent))
        print('=== Tgt ===')
        print('Max sent : {:>5d}  Min sent  : {:>5d}   Avg sent  : {}'.format(
            max_tgt_sent, min_tgt_sent, avg_num_tgt_sent))
        print('Max token: {:>5d}  Min token : {:>5d}   Avg token : {}'.format(
            max_tgt_token, min_tgt_token, avg_num_tgt_token))
        print('Max sent token : {:>5d}  Min sent token : {:>5d}'.format(
            max_tgt_token_per_sent, min_tgt_token_per_sent))

        with open(pjoin(out_sub_dir, 'analysis.txt'), 'w') as f_o:
            f_o.write('=== Src ===\n')
            f_o.write(
                'Max sent : {:>5d}  Min sent  : {:>5d}   Avg sent  : {}\n'.
                format(max_src_sent, min_src_sent, avg_num_src_sent))
            f_o.write(
                'Max token: {:>5d}  Min token : {:>5d}   Avg token : {}\n'.
                format(max_src_token, min_src_token, avg_num_src_token))
            f_o.write(
                'Max sent token : {:>5d}  Min sent token : {:>5d}\n'.format(
                    max_src_token_per_sent, min_src_token_per_sent))
            f_o.write('=== Tgt ===\n')
            f_o.write(
                'Max sent : {:>5d}  Min sent  : {:>5d}   Avg sent  : {}\n'.
                format(max_tgt_sent, min_tgt_sent, avg_num_tgt_sent))
            f_o.write(
                'Max token: {:>5d}  Min token : {:>5d}   Avg token : {}\n'.
                format(max_tgt_token, min_tgt_token, avg_num_tgt_token))
            f_o.write(
                'Max sent token : {:>5d}  Min sent token : {:>5d}\n'.format(
                    max_tgt_token_per_sent, min_tgt_token_per_sent))


##################################
##      Format to Lines         ##
##################################


def format_to_lines_tfds(args):
    """ Formats source text and target text as pt file. """

    tokenized_sub_dirs = os.listdir(args.raw_path)
    dataset_name = os.path.dirname(args.save_path).split('/')[-1]

    # Make directory
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    # Create file list for each split directory
    corpora = {}
    for tokenized_sub_dir in tokenized_sub_dirs:
        path = pjoin(args.raw_path, tokenized_sub_dir)
        files = []
        for f in glob.glob(pjoin(path, '*.json')):
            files.append(f)
        corpora[tokenized_sub_dir] = files
        files = []

    for corpus_type in tokenized_sub_dirs:
        a_lst = [(f, args) for f in corpora[corpus_type]]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_to_lines, a_lst):
            dataset.append(d)
            # NOTE: save files according to shard_size
            if (len(dataset) >= args.shard_size):
                if (corpus_type == 'validation'):
                    type_name = 'valid'
                else:
                    type_name = corpus_type
                pt_file = "{:s}.{:s}.{:d}.json".format(dataset_name, type_name,
                                                       p_ct)
                with open(pjoin(args.save_path, pt_file), 'w') as save:
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []
        pool.close()
        pool.join()

        # For the last few data (< shard size)
        if (len(dataset) > 0):
            if (corpus_type == 'validation'):
                type_name = 'valid'
            else:
                type_name = corpus_type
            pt_file = "{:s}.{:s}.{:d}.json".format(dataset_name, type_name,
                                                   p_ct)
            with open(pjoin(args.save_path, pt_file), 'w') as save:
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []


def _format_to_lines(params):
    """ Extracts source text and target text from json file. """
    f, args = params
    print(f)
    source, tgt = load_json(f, args.lower)
    return {'src': source, 'tgt': tgt}


def load_json(p, lower):
    """ Construct sentences from tokenized json file. """
    source = []
    tgt = []
    flag = False
    for sent in json.load(open(p))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]

        if (lower): # NOTE: convert word to lower case
            tokens = [t.lower() for t in tokens]

        if (tokens[0] == '@highlight'):
            flag = True
            tgt.append([])
            continue

        if (flag):
            # targets are concatnated as single sentence
            tgt[-1].extend(tokens)
        else:
            source.append(tokens)

    source = [clean(' '.join(sent)).split() for sent in source]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]
    return source, tgt


##################################
##      Format to Bert          ##
##################################


def format_to_bert(args):
    """ Transforms words to ids with BERT tokenizer. """

    # Create folders
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']

    # Multiprocessing for _format_to_bert()
    for corpus_type in datasets:
        if not args.debug:
            a_lst = []
            for json_f in glob.glob(
                    pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
                real_name = json_f.split('/')[-1]
                a_lst.append((corpus_type, json_f, args,
                              pjoin(args.save_path,
                                    real_name.replace('json', 'bert.pt'))))
            print("Processing {} dataset...".format(corpus_type))
            pool = Pool(args.n_cpus)
            for d in pool.imap(_format_to_bert, a_lst):
                pass

            pool.close()
            pool.join()
        else:
            # NOTE: debug without multiprocessing
            print("Processing {} dataset...".format(corpus_type))
            for json_f in glob.glob(
                    pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
                real_name = json_f.split('/')[-1]
                _format_to_bert((corpus_type, json_f, args,
                                 pjoin(args.save_path,
                                       real_name.replace('json', 'bert.pt'))))


def _format_to_bert(params):
    corpus_type, json_file, args, save_file = params
    is_test = corpus_type == 'test'

    # NOTE: not to overwrite the already generated files
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    # Create data class
    bert = BertData(args)

    # Open json data
    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))

    datasets = []
    for d in jobs:
        source, tgt = d['src'], d['tgt']

        # Create pseudo extraction labels
        sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 3)

        # Make tokens into lower case
        if (args.lower):
            source = [' '.join(s).lower().split() for s in source]
            tgt = [' '.join(s).lower().split() for s in tgt]

        b_data = bert.preprocess(
            source,
            tgt,
            sent_labels,
            use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
            is_test=is_test)

        # b_data = bert.preprocess(
        #    source,
        #    tgt,
        #    sent_labels,
        #    use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)

        # Skip empty data
        if (b_data is None):
            continue

        # Format single datum as dict
        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {
            "src": src_subtoken_idxs,
            "tgt": tgt_subtoken_idxs,
            "src_sent_labels": sent_labels,
            "segs": segments_ids,
            'clss': cls_ids,
            'src_txt': src_txt,
            "tgt_txt": tgt_txt
        }

        # Collect data as list
        datasets.append(b_data_dict)

    # Save data as pt file
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                       do_lower_case=True)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self,
                   src,
                   tgt,
                   sent_labels,
                   use_bert_basic_tokenizer=False,
                   is_test=False):

        """
        Args:
            src (list[list[str]])
            tgt (list[list[str]])
            sent_labels (list)

        Returns:
            src_subtoken_idxs (list[int]): tokenized src ids.
            sent_labels (list[int]): one-hot extraction labels.
            tgt_subtoken_idxs (list[int]): tokenized tgt ids.
            segments_ids (list[int]): segment embedding sequence.
            cls_ids (list[int]): positions of [CLS] tokens.
            src_txt (list[str]): raw source text.
            tgt_txt (str) raw target text.
        """

        # Skip data with empty source
        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]

        ##### Extraction Label #####
        # Create binary extraction pesudo labels with truncation
        if (self.args.max_src_ntokens > 0):
            idxs = []
            token_num = 0
            for i, s in enumerate(src):
                token_num += len(s)
                if (len(s) > self.args.min_src_ntokens_per_sent):
                    if (token_num > self.args.max_src_ntokens):
                        break
                    else:
                        idxs.append(i)
        else:
            idxs = [
                i for i, s in enumerate(src)
                if (len(s) > self.args.min_src_ntokens_per_sent)
            ]

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            _sent_labels[l] = 1

        # Truncate pesudo labels with max_src_ntokens_sent 
        sent_labels = [_sent_labels[i] for i in idxs]
        sent_labels = sent_labels[:self.args.max_src_nsents]


        ##### Source #####
        # Truncate src tokens with max_src_ntokens_per_sent and max_src_nsents
        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        src = src[:self.args.max_src_nsents]

        # Skip data without enough sentences
        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        # Retokenize with BERT-tokenizer, and add special tokens
        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]

        # Covert src tokens into ids.
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)


        ##### [SEP] IDs #####
        _segs = [-1] + [
            i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid
        ]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]


        ##### [CLS] IDs #####
        cls_ids = [
            i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid
        ]
        sent_labels = sent_labels[:len(cls_ids)]


        ##### Target #####
        # Retokenize with BERT-tokenizer, and add special tokens
        tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join([
            ' '.join(
                self.tokenizer.tokenize(
                    ' '.join(tt),
                    use_bert_basic_tokenizer=use_bert_basic_tokenizer))
            for tt in tgt
        ]) + ' [unused1]'

        # Truncate targets with max_tgt_ntokens
        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]

        # Skip data without enough target tokens
        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        # Covert tgt tokens into ids.
        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)


        ##### Raw Text #####
        src_txt = [original_src_txt[i] for i in idxs]
        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])


        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    """ Create pseudo extraction labels. 

    Args:
        doc_sent_list (list[list[str]]):
            source text to be processed.

        abstract_sent_list (list[list[str]]):
            target text  to be processed.

        summary_size(int) :
            maximum number of extracted sentences.

    Returns:
        A list of extracted sentence indices in ascending order.
    """
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0

    # Clean and concat all target sentences
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()

    # Clean all source sentences
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]

    # Get 1 grams and 2 grams from source and target
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1

        # Iterates through all sentences
        for i in range(len(sents)):
            if (i in selected):
                continue

            # Consider selected and candidate sentences together
            c = selected + [i]

            # Calculate ROUGE-1-F + ROUGE-2-F with target
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i

        # If no sentence exceeds current max score then stop 
        if (cur_id == -1):
            return selected

        # Record currently chosen sentence and score
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def cal_rouge(evaluated_ngrams, reference_ngrams):
    """ Calculates ROUGE scores.

    Args:
        evaluated_ngrams (set(tuple)):
            source ngrams to be considered.

        reference_ngrams (set(tuple)):
            target_ngrams to be considered.

    Returns:
        A dict of ROUGE-F/P/R scores.
    """
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}

