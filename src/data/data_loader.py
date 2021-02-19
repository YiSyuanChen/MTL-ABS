""" Loads data

This file contains functions to load data. Class Dataloader use load_dataset()
generator to get data in a pt file, and use class DataIterator to produce
Class Batch. The Class Batch will be sent to training/validation/testing functions.

"""
import bisect
import gc
import glob
import random
import os
import torch
from others.logging import logger
import pdb


class Batch(object):
    """ Encapsulate data as a batch

    The __init__() function performs padding and produces masking for data.
    The processed data will be change to Torch tensor and stored as attributes.

    Attributes:
        batch_size (int)
        src (tensor):
            shape=(batch_size, src_len)
        mask_src (tensor):
            shape=(batch_size, src_len)
        clss (tensor):
            position of token [CLS], shape=(batch_size, src_sent_num)
        mask_cls (tensor):
            shape=(batch_size, src_sent_num)
        segs (tensor):
            segment word id (1 or 0), shape=(batch_size, src_len)
        src_sent_labels (tensor):
            ground-truth of extractive sum, shape=(batch_size, src_sent_num)
        tgt (tensor):
            shape=(batch_size, tgt_len)
        mask_tgt (tensor):
            shape=(batch_size, tgt_len)
        src_str (list, optional):
            raw string of article.
        tgt_str (list, optional):
            raw string of summary.

    """
    def __init__(self, data=None, device=None, is_test=False):
        """ Create a Batch from a list of examples.

        Args:
            data (list):
                a list of data, and each data is a tuple. The data tuple
                contains five (or seven in testing) lists include src, tgt,
                segs, clss and src_sent_labels (src_str and tgt_str in testing).
            device (string):
                indicates whether using GPU, which should be "cuda" or "cpu".
            is_test (bool):
                indicates whether including src_str and tgt_str.
        """
        if data is not None:
            self.batch_size = len(data)
            pre_src = [x[0] for x in data]
            pre_tgt = [x[1] for x in data]
            pre_segs = [x[2] for x in data]
            pre_clss = [x[3] for x in data]
            pre_src_sent_labels = [x[4] for x in data]

            # Padding
            src = torch.tensor(self._pad(pre_src, 0))
            clss = torch.tensor(self._pad(
                pre_clss, -1))  # use -1 since values of clss could be 0
            segs = torch.tensor(self._pad(pre_segs, 0))
            src_sent_labels = torch.tensor(self._pad(pre_src_sent_labels, 0))
            tgt = torch.tensor(self._pad(pre_tgt, 0))

            # Produce masking
            mask_src = ~(src == 0)
            mask_tgt = ~(tgt == 0)
            mask_cls = ~(clss == -1)
            clss[clss == -1] = 0

            # Change to tensor and set as attributes
            setattr(self, 'src', src.to(device))
            setattr(self, 'mask_src', mask_src.to(device))
            setattr(self, 'clss', clss.to(device))
            setattr(self, 'mask_cls', mask_cls.to(device))
            setattr(self, 'segs', segs.to(device))
            setattr(self, 'src_sent_labels', src_sent_labels.to(device))
            setattr(self, 'tgt', tgt.to(device))
            setattr(self, 'mask_tgt', mask_tgt.to(device))

            if is_test:
                src_str = [x[-2] for x in data]
                setattr(self, 'src_str', src_str)
                tgt_str = [x[-1] for x in data]
                setattr(self, 'tgt_str', tgt_str)

    def __len__(self):
        """ Return batch size """
        return self.batch_size

    def _pad(self, data, pad_id, width=-1):
        """ Append pad tokens to the data

        Args:
            data (list):
                the batch of data to be padded.
            pad_id (int):
                the word id of pad token.
            width (int):
                maximum length to pad. Defualt value is the maximum data length in the batch.
        Returns:
            A padded data (list)
        """
        if width == -1:
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data


def load_dataset(args, corpus_type, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type:
            'train', 'valid' or 'test'
    Returns:
        A list of dataset (list(dict)), the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))

        # Type = list[dict, dict, ...], len = # of files
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    # Original: pts = sorted(glob.glob(args.bert_data_path + '.' + corpus_type + '.[0-9]*.pt'))
    # Modified according to  issue 92
    # Type = list[str,str,...], len = # of files
    # Ex  : ['xsum.train.1.bert.pt', 'xsum.train.2.bert.pt', ...]
    pts = sorted(
        glob.glob(args.bert_data_path + '/' + r'[a-z]*.' + corpus_type +
                  r'.[0-9]*.bert.pt'))
    if pts:
        if shuffle:
            random.shuffle(pts)
        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        raise """Currently not support single file loading.
            Make sure the data folder has at least two .pt file"""


def load_meta_dataset(args, corpus_type, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type:
            'train', 'valid' or 'valid_new_task'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid", "valid_new_task"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d',
                    corpus_type, pt_file, len(dataset))

        # Type: list[dict, dict, ...], len = # of files
        return dataset

    type_path = os.path.join(args.bert_data_path, corpus_type)
    sup_path = os.path.join(type_path, 'support')
    qry_path = os.path.join(type_path, 'query')
    assert os.listdir(sup_path) == os.listdir(
        qry_path)  # Make sure the tasks are same
    dataset_names = os.listdir(sup_path)

    # Only need the 'valid_new_task' option to load path
    if corpus_type == 'valid_new_task':
        corpus_type = 'valid'

    # Sort the glob output by file name (by increasing indexes).
    # Original: pts = sorted(glob.glob(args.bert_data_path + '.' + corpus_type + '.[0-9]*.pt'))
    # Modified according to  issue 92
    # Type: list[str,str,...], len = # of files
    # Ex  : ['xsum.train.1.bert.pt', 'xsum.train.2.bert.pt', ...]
    sup_pts = []
    qry_pts = []
    for name in dataset_names:
        s_path = os.path.join(sup_path, name)
        q_path = os.path.join(qry_path, name)
        s_pts = sorted(
            glob.glob(s_path + '/' + r'[a-z]*.' + corpus_type +
                      r'.[0-9]*.bert.pt'))
        q_pts = sorted(
            glob.glob(q_path + '/' + r'[a-z]*.' + corpus_type +
                      r'.[0-9]*.bert.pt'))
        sup_pts.append(s_pts)
        qry_pts.append(q_pts)

    # Suffle the tasks (sup & qry with same order)
    #combine = list(zip(sup_pts, qry_pts))
    #random.shuffle(combine)
    #sup_pts, qry_pts = zip(*combine)

    # Crossover
    #random.shuffle(qry_pts)

    # Suffle files in each task
    if shuffle:
        for i in range(len(sup_pts)):  # iterate through tasks
            random.shuffle(sup_pts[i])
            random.shuffle(qry_pts[i])

    # Interleaving load data from different task
    # Some amount of file will be load for each task
    min_file_num = min([len(pts)
                        for pts in sup_pts]) - 1  # min # of files in all tasks
    for file_id in range(min_file_num):
        for task_id in range(len(dataset_names)):
            yield _lazy_dataset_loader(sup_pts[task_id][file_id], corpus_type),\
                  _lazy_dataset_loader(qry_pts[task_id][file_id], corpus_type)


def abs_batch_size_fn(new, count):
    """ Calculate current cost for batch size

    Args:
        new (tuple):
            the data, which contain fives list includeing src,tgt,...
        count (int):
            number of accumulated data.
    """
    src, tgt = new[0], new[1]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents = 0
        max_n_tokens = 0

    # max_size = max tgt len in accumulated data
    max_n_sents = max(max_n_sents, len(tgt))
    max_size = max(max_size, max_n_sents)

    # calculate cost
    src_elements = count * max_size

    # [Not sure] maybe prevent accumulate too many data
    if count > 6:
        return src_elements + 1e3
    return src_elements


class Dataloader(object):
    """ Loading data

    Attributes:
        args
        dataset (generator):
            lazy data loader, which returns a list of dict in each iteration.
            Each list represent a file, and each dict represents a data.
        batch_size (int)
        device (string)
        shuffle (bool)
        is_test (bool)
        cur_iter (class DataIterator):
            yield a batch of data

    """
    def __init__(self, args, datasets, batch_size, device, shuffle, is_test):
        self.args = args
        self.datasets = datasets  # generator of list[dict, dict, ...], len=# of files
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(
            datasets)  # Class DataIterator, return Class Batch
        assert self.cur_iter is not None

    def __iter__(self):
        """ Return a realization of class Batch """
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch  # Class Batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def _next_dataset_iterator(self, dataset_iter):
        """ Prepare next file """
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(
                dataset_iter)  # list[dict, dict, ...], len=# of files
        except StopIteration:
            return None

        return DataIterator(args=self.args,
                            dataset=self.cur_dataset,
                            batch_size=self.batch_size,
                            device=self.device,
                            shuffle=self.shuffle,
                            is_test=self.is_test)


class MetaDataloader(object):
    """ Loading data

    Attributes:
        args
        dataset (generator):
            lazy data loader, which returns a list of dict in each iteration.
            Each list represent a file, and each dict represents a data.
        batch_size (int)
        device (string)
        shuffle (bool)
        is_test (bool)
        cur_iter (class DataIterator):
            yield a batch of data

    """
    def __init__(self, args, datasets, batch_size, device, shuffle, is_test):
        self.args = args
        self.datasets = datasets  # tuple of generator of list[dict, dict, ...], len=# of files
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_sup_iter, self.cur_qry_iter = self._next_dataset_iterator(
            datasets)  # Class DataIterator, return Class Batch
        assert self.cur_sup_iter is not None
        assert self.cur_qry_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_sup_iter is not None and self.cur_qry_iter is not None:
            for idx, (sup_batch, qry_batch) in enumerate(
                    zip(self.cur_sup_iter,
                        self.cur_qry_iter)):  # In current file
                yield sup_batch, qry_batch  # Class Batch
                if (idx == self.args.num_batch_in_task -
                        1):  # break when enough batches
                    break
            self.cur_sup_iter, self.cur_qry_iter = self._next_dataset_iterator(
                dataset_iter)  # Prepare next file

    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_sup_dataset") or hasattr(
                    self, 'cur_qry_dataset'):
                self.cur_sup_dataset = None
                self.cur_qry_dataset = None
                gc.collect()
                del self.cur_sup_dataset
                del self.cur_qry_dataset
                gc.collect()

            self.cur_sup_dataset, self.cur_qry_dataset = next(
                dataset_iter)  # list[dict, dict, ...], len=# of files
        except StopIteration:
            return None, None

        return DataIterator(args = self.args,
            dataset=self.cur_sup_dataset,  batch_size=self.batch_size,
            device=self.device, shuffle=self.shuffle, is_test=self.is_test), \
               DataIterator(args = self.args,
            dataset=self.cur_qry_dataset,  batch_size=self.batch_size,
            device=self.device, shuffle=self.shuffle, is_test=self.is_test)


class DataIterator(object):
    """ Dataloader for a file

    This class preprocess and return a batch of data whitin a pt file.

    Attributes:
        args
        dataset (list): a list of dict, each dict is data
        batch_size (int)
        device (string)
        is_test (bool)
        shuffle (bool)
        iterations (int)
        sort_key (function): to sort data
        self.batch_size_fn (function): to calculate batch size
    """
    def __init__(self,
                 args,
                 dataset,
                 batch_size,
                 device=None,
                 is_test=False,
                 shuffle=True):
        self.args = args
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.is_test = is_test
        self.shuffle = shuffle

        self.iterations = 0
        self.sort_key = lambda x: len(x[1])  # use tgt to sort (for abs)
        self._iterations_this_epoch = 0
        if self.args.task == 'abs':
            self.batch_size_fn = abs_batch_size_fn
        else:
            raise "Currently only support abstractive summarization."
            #self.batch_size_fn = ext_batch_size_fn

    def __iter__(self):
        while True:
            self.batches = self.create_batches(
            )  # generator of list[tuple, tuple, ...], len = dynamic batch_size 2
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_test)

                yield batch
            return

    def create_batches(self):
        """ Create batches """
        # Type: list[dict, dict, ...], len = # of files
        data = self.data()

        buffer_coeff = 300
        # Type : list[tuple, tuple, ...], len <= (self.batch_size*300)/max_tgt_len (dynamic batch_size 1)
        for buffer in self.batch_buffer(data, self.batch_size * buffer_coeff):
            if self.args.task == 'abs':
                # Sort first with tgt_len, than src_len
                p_batch = sorted(buffer, key=lambda x: len(x[2]))
                p_batch = sorted(p_batch, key=lambda x: len(x[1]))
            else:
                p_batch = sorted(buffer, key=lambda x: len(x[2]))

            # Type: generator of list[tuple, tuple, ...], len <= self.batch/max_tgt_len (dynamic batch size 2)
            p_batch = self.batch(p_batch, self.batch_size)
            p_batch = list(
                p_batch
            )  # list[list[tuple, tuple, ...], list[tuple, tuple, ...], ...]

            if self.shuffle:
                random.shuffle(p_batch)
            for b in p_batch:  # b -> list[tuple, tuple, ...], len=p_batch
                if len(b) == 0:
                    continue

                # When there are at least two batches, skip the last batch if
                # the number of data does not reach batch size.
                if (self.args.deterministic_batch_size
                        and (len(b) != self.batch_size) and len(p_batch) > 1):
                    continue
                yield b

    def batch_buffer(self, data, batch_size):
        """ Split the data in a file into mini-batch

        Args:
            data (list): a list of dict, each dict is a data.
            batch_size(int): actual batch_size * buffer_coeff
        """
        minibatch, size_so_far = [], 0

        for ex in data:  # ex -> dict
            # Skip the data is lenght of src is 0
            if len(ex['src']) == 0:
                continue

            # Preprocess the dict, return a tuple of list
            ex = self.preprocess(ex, self.is_test)  # tuple of list
            if ex is None:
                continue

            # Accumulate data
            minibatch.append(ex)  # list[tuple, tuple, ...]
            size_so_far = self.batch_size_fn(ex, len(minibatch))

            # Yield mini-batch is number of data reach batch size
            # NOTE: the batch size here is real batch size * batch coefficient
            if size_so_far == batch_size:  # batch_size = args.batch_size * 300
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                # Continue to accumulate data with left one
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(
                    ex, 1)

        # Yield data at last that are not enough for a batch
        if minibatch:
            yield minibatch

    def data(self):
        """ Shuffle the dataset (list of dict) """
        if self.shuffle:
            random.shuffle(self.dataset)  # randomize in a file
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_test):
        """ Preprocess a data

        Args:
            ex (dict):
                a data
            is_test (bool):
                whether to include src_txt and tgt_txt.
        """
        src = ex['src']
        clss = ex['clss']
        segs = ex['segs']
        if not self.args.use_interval:
            segs = [0] * len(segs)
        src_sent_labels = ex['src_sent_labels']
        # NOTE: cut tgt token with max_tgt_len, and append [EOQ]
        tgt = ex['tgt'][:self.args.max_tgt_len][:-1] + [2]
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']

        end_id = [src[-1]]
        # NOTE: cut src token with max_pos, and append [SEP]
        src = src[:-1][:self.args.max_pos - 1] + end_id
        segs = segs[:self.args.max_pos]
        max_sent_id = bisect.bisect_left(clss, self.args.max_pos)
        clss = clss[:max_sent_id]
        src_sent_labels = src_sent_labels[:max_sent_id]
        # src_txt = src_txt[:max_sent_id]

        # Return a tuple of list
        if is_test:
            return src, tgt, segs, clss, src_sent_labels, src_txt, tgt_txt
        else:
            return src, tgt, segs, clss, src_sent_labels

    def batch(self, data, batch_size):
        """Yield elements from data in chunks of batch_size."""
        minibatch, size_so_far = [], 0

        if self.args.deterministic_batch_size:
            for data_idx, ex in enumerate(data):
                # Accumulate data
                minibatch.append(ex)
                # Yield mini-batch is number of data reach batch size
                if (data_idx + 1) % batch_size == 0:
                    yield minibatch
                    minibatch, size_so_far = [], 0
            # Yield data at last that are not enough for a batch
            if minibatch:
                yield minibatch
        else:
            for ex in data:
                # Accumulate data
                minibatch.append(ex)
                size_so_far = self.batch_size_fn(ex, len(minibatch))

                # Yield mini-batch is number of data reach batch size
                # NOTE: the batch size here is the real batch size
                if size_so_far == batch_size:
                    yield minibatch
                    minibatch, size_so_far = [], 0
                elif size_so_far > batch_size:
                    yield minibatch[:-1]
                    # Continue to accumulate data with left one
                    minibatch, size_so_far = minibatch[
                        -1:], self.batch_size_fn(ex, 1)
            # Yield data at last that are not enough for a batch
            if minibatch:
                yield minibatch


class TextDataloader(object):
    def __init__(self, args, datasets, batch_size, device, shuffle, is_test):
        self.args = args
        self.batch_size = batch_size
        self.device = device

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_test):
        src = ex['src']
        tgt = ex['tgt'][:self.args.max_tgt_len][:-1] + [2]
        src_sent_labels = ex['src_sent_labels']
        segs = ex['segs']
        if (not self.args.use_interval):
            segs = [0] * len(segs)
        clss = ex['clss']
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']

        end_id = [src[-1]]
        src = src[:-1][:self.args.max_pos - 1] + end_id
        segs = segs[:self.args.max_pos]
        max_sent_id = bisect.bisect_left(clss, self.args.max_pos)
        src_sent_labels = src_sent_labels[:max_sent_id]
        clss = clss[:max_sent_id]
        # src_txt = src_txt[:max_sent_id]

        if (is_test):
            return src, tgt, segs, clss, src_sent_labels, src_txt, tgt_txt
        else:
            return src, tgt, segs, clss, src_sent_labels

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if (len(ex['src']) == 0):
                continue
            ex = self.preprocess(ex, self.is_test)
            if (ex is None):
                continue
            minibatch.append(ex)
            size_so_far = simple_batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], simple_batch_size_fn(
                    ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 300):
            if (self.args.task == 'abs'):
                p_batch = sorted(buffer, key=lambda x: len(x[2]))
                p_batch = sorted(p_batch, key=lambda x: len(x[1]))
            else:
                p_batch = sorted(buffer, key=lambda x: len(x[2]))
                p_batch = batch(p_batch, self.batch_size)

            p_batch = batch(p_batch, self.batch_size)

            p_batch = list(p_batch)
            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                if (len(b) == 0):
                    continue
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_test)

                yield batch
            return
