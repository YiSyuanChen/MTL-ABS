"""Functions for training/validation/testing. """
from __future__ import division

import argparse
import glob
import os
import random
import signal
import time
import pdb

import torch
from transformers import BertTokenizer

from data import data_loader
from data.data_loader import load_dataset, load_meta_dataset
from models import model_builder
from models.loss import abs_loss
from models.model_builder import AbsSummarizer, MTLAbsSummarizer
from models.predictor import build_predictor
from pipelines.trainer_abs import build_trainer
from pipelines.trainer_abs_mtl import build_MTLtrainer
from others.logging import logger, init_logger
from others.utils import str2bool 
import distributed

model_flags = [
    'hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers',
    'enc_hidden_size', 'enc_ff_size', 'dec_layers', 'dec_hidden_size',
    'dec_ff_size', 'encoder', 'ff_actv', 'use_interval'
]

def train_abs(args, device_id):
    """Decides to run training in multi-GPU or single-GPU mode.
    Args:
        device_id (int) : the GPU id to be used
    """
    if args.world_size > 1:
        train_abs_multi(args)
    else:
        train_abs_single(args, device_id)


def train_abs_single(args, device_id):
    """Implements training process (meta / non-meta)
    Args:
        device_id (int) : the GPU id to be used
    """

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d', device_id)
    logger.info('Device %s', device)

    # Fix random seed to control experiement
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    if device_id >= 0:  # if use GPU
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    # Load checkpoint and args
    if args.train_from != '':
        logger.info('Loading checkpoint from %s', args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])  # which is self.args
        for k in opt.keys():
            if k in model_flags:
                setattr(args, k, opt[k])
    else:
        checkpoint = None

    # Load extractive model as initial parameter (proposed by Presumm)
    if args.load_from_extractive != '':
        logger.info('Loading bert from extractive model %s',
                    args.load_from_extractive)
        bert_from_extractive = torch.load(
            args.load_from_extractive,
            map_location=lambda storage, loc: storage)
        bert_from_extractive = bert_from_extractive['model']
    else:
        bert_from_extractive = None

    # Prepare dataloader
    if args.meta_mode:
        def meta_train_iter_fct():
            return data_loader.MetaDataloader(args,
                                              load_meta_dataset(args,
                                                                'train',
                                                                shuffle=True),
                                              args.batch_size,
                                              device,
                                              shuffle=True,
                                              is_test=False)
    else:
        def train_iter_fct():
            return data_loader.Dataloader(args,
                                          load_dataset(args,
                                                       'train',
                                                       shuffle=True),
                                          args.batch_size,
                                          device,
                                          shuffle=True,
                                          is_test=False)

    # Prepare model
    if args.meta_mode:
        model = MTLAbsSummarizer(args, device, checkpoint,
                                 bert_from_extractive)
    else:
        model = AbsSummarizer(args, device, checkpoint, bert_from_extractive)

    # Prepare optimizer for inner loop
    # The optimizer for each task is seperated
    if args.meta_mode:
        optims_inner = []
        for _ in range(args.num_task):
            if args.sep_optim:
                optim_bert_inner = model_builder.build_optim_bert_inner(
                    args, model, checkpoint, 'maml')
                optim_dec_inner = model_builder.build_optim_dec_inner(
                    args, model, checkpoint, 'maml')
                optims_inner.append([optim_bert_inner, optim_dec_inner])
            else:
                optims_inner.append([
                    model_builder.build_optim_inner(args, model, checkpoint,
                                                    'maml')
                ])

    # Prepare optimizer for outer loop
    if args.sep_optim:
        optim_bert = model_builder.build_optim_bert(args, model, checkpoint)
        optim_dec = model_builder.build_optim_dec(args, model, checkpoint)
        optims = [optim_bert, optim_dec]
    else:
        optims = [model_builder.build_optim(args, model, checkpoint)]

    # Prepare tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              do_lower_case=True,
                                              cache_dir=args.temp_dir)
    symbols = {
        'BOS': tokenizer.vocab['[unused0]'], # id = 1
        'EOS': tokenizer.vocab['[unused1]'], # id = 2
        'EOQ': tokenizer.vocab['[unused2]'], # id = 3
        'PAD': tokenizer.vocab['[PAD]'] # id = 0
    }

    # Self Check : special word ids
    special_words = [w for w in tokenizer.vocab.keys() if "[" in w]
    special_word_ids = [
        tokenizer.convert_tokens_to_ids(w) for w in special_words
    ]

    # Prepare loss computation
    train_loss = abs_loss(model.generator,
                          symbols,
                          model.vocab_size,
                          device,
                          train=True,
                          label_smoothing=args.label_smoothing)

    # Prepare trainer and perform training
    if args.meta_mode:
        trainer = build_MTLtrainer(args, device_id, model, optims,
                                   optims_inner, train_loss)
        trainer.train(meta_train_iter_fct)
    else:
        trainer = build_trainer(args, device_id, model, optims, train_loss)
        trainer.train(train_iter_fct, args.train_steps)


def train_abs_multi(args):
    """ Spawns 1 process per GPU """
    init_logger()

    nb_gpu = args.world_size
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        device_id = i
        procs.append(
            mp.Process(target=run,
                       args=(
                           args,
                           device_id,
                           error_queue,
                       ),
                       daemon=True))
        procs[i].start()
        logger.info(" Starting process pid: %d  " % procs[i].pid)
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()


def run(args, device_id, error_queue):
    """ run process """

    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])

    try:
        gpu_rank = distributed.multi_init(device_id, args.world_size,
                                          args.gpu_ranks)
        print('gpu_rank %d' % gpu_rank)
        if gpu_rank != args.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")

        train_abs_single(args, device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""
    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(target=self.error_listener,
                                             daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def validate_abs(args, device_id):
    """Decides to check all checkpoint or keep tracking last one
    Arguments:
        device_id (int) : the GPU id to be used
    Process:
        - sort checkpoint by time
        - perform validate()
        - perform test_abs()
    """
    timestep = 0
    if args.test_all:

        # Sort ckpt by time
        cp_files = sorted(
            glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
        cp_files.sort(key=os.path.getmtime)

        # Compute the cross-entropy loss for each ckpt
        xent_lst = []
        for i, cp in enumerate(cp_files):
            step = int(cp.split('.')[-2].split('_')[-1])

            # Test from after specific step
            if (args.test_start_from != -1 and step < args.test_start_from):
                xent_lst.append((1e6, cp))
                continue

            # Validation
            xent = validate(args, device_id, cp, step)
            xent_lst.append((xent, cp))

            # Record best step
            max_step = xent_lst.index(min(xent_lst))

            # Stop if following 50 ckpt are not better
            if i - max_step > 50:
                break

        # Sort according to CE loss, only get top 5
        #xent_lst = sorted(xent_lst, key=lambda x: x[0])[:5]
        xent_lst = sorted(xent_lst,
                          key=lambda x: x[0])[:args.valid_all_test_num]
        logger.info('Xent %s', str(xent_lst))

        # Skip testing in meta learning setting
        # Since we do not need to test on validation task
        if not args.meta_mode:
            for xent, cp in xent_lst:
                step = int(cp.split('.')[-2].split('_')[-1])
                test_abs(args, device_id, cp, step)
    else:
        while True:  # Keep moniter the ckpts

            # Sort ckpt by time
            cp_files = sorted(
                glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if cp_files:
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)

                # If the ckpt has not been establish completely, then skip
                if not os.path.getsize(cp) > 0:
                    time.sleep(60)
                    continue

                # If there is new ckpt, then perform validation and test
                if time_of_cp > timestep:
                    timestep = time_of_cp
                    step = int(cp.split('.')[-2].split('_')[-1])
                    validate(args, device_id, cp, step)
                    test_abs(args, device_id, cp, step)

            cp_files = sorted(
                glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if cp_files:
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if time_of_cp > timestep:
                    continue
            else:
                time.sleep(300)


def validate(args, device_id, pt, step):
    ''' Implements validation process (meta / non-memta)
    Arguments:
        device_id (int) : the GPU id to be used
        pt() : checkpoint model
        step (int) : checkpoint step
    Process:
        - load checkpoint
        - prepare dataloader class
        - prepare model class
        - prepare loss func, which return loss class
        - prepare trainer
        - trainer.validate()
    Meta vs Normal
        - MetaDataloader      vs Dataloader
        - load_dataset        vs load_meta_dataset
        - MTLAbsSummarizer    vs AbsSummarizer
        - build_MTLtrainer    vs MTLTrainer
    '''
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)

    # Fix random seed to control experiement
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    # Load checkpoint ard args
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from,
                            map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])  # which is self.args
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])

    # Prepare dataloader
    if (args.meta_mode):

        def valid_iter_fct():
            return data_loader.MetaDataloader(args,
                                              load_meta_dataset(
                                                  args,
                                                  'valid',
                                                  shuffle=True),
                                              args.batch_size,
                                              device,
                                              shuffle=True,
                                              is_test=False)

    else:
        valid_iter = data_loader.Dataloader(args,
                                            load_dataset(args,
                                                         'valid',
                                                         shuffle=False),
                                            args.batch_size,
                                            device,
                                            shuffle=False,
                                            is_test=False)

    # Prepare model
    if (args.meta_mode):
        model = MTLAbsSummarizer(args, device, checkpoint)
    else:
        model = AbsSummarizer(args, device, checkpoint)
    #model.eval()

    # Prepare optimizer for inner loop
    # The optimizer for each task is seperated
    if (args.meta_mode):
        optims_inner = []
        for i in range(args.num_task):
            if (args.sep_optim):
                optim_bert_inner = model_builder.build_optim_bert_inner(
                    args, model, checkpoint, 'maml')
                optim_dec_inner = model_builder.build_optim_dec_inner(
                    args, model, checkpoint, 'maml')
                optims_inner.append([optim_bert_inner, optim_dec_inner])
            else:
                self.optims_inner.append([
                    model_builder.build_optim_inner(args, model, checkpoint,
                                                    'maml')
                ])

    # Prepare optimizer (not actually used, but get the step information)
    if (args.sep_optim):
        optim_bert = model_builder.build_optim_bert(args, model, checkpoint)
        optim_dec = model_builder.build_optim_dec(args, model, checkpoint)
        optim = [optim_bert, optim_dec]
    else:
        optim = [model_builder.build_optim(args, model, checkpoint)]

    # Prepare loss
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              do_lower_case=True,
                                              cache_dir=args.temp_dir)
    symbols = {
        'BOS': tokenizer.vocab['[unused0]'],
        'EOS': tokenizer.vocab['[unused1]'],
        'PAD': tokenizer.vocab['[PAD]'],
        'EOQ': tokenizer.vocab['[unused2]']
    }

    # Prepare loss computation
    valid_loss = abs_loss(model.generator,
                          symbols,
                          model.vocab_size,
                          device,
                          train=False)

    # Prepare trainer and perform validation
    if (args.meta_mode):
        trainer = build_MTLtrainer(args, device_id, model, optim, optims_inner,
                                   valid_loss)
        stats = trainer.validate(valid_iter_fct, step)
    else:
        trainer = build_trainer(args, device_id, model, None, valid_loss)
        stats = trainer.validate(valid_iter, step)

    return stats.xent()


def test_abs(args, device_id, pt, step):
    """ Implements testing process (meta / non-memta)
    Arguments:
        device_id (int) : the GPU id to be used
        pt() : checkpoint model
        step (int) : checkpoint step
    Process:
        - load checkpoint
        - prepare dataloader class
        - prepare model class
        - prepare predictor
        - predictor.translate()
    """
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d', device_id)
    logger.info('Device %s', device)

    # Load chekcpoint
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from,
                            map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])

    # Prepare dataloader
    test_iter = data_loader.Dataloader(args,
                                       load_dataset(args,
                                                    'test',
                                                    shuffle=False),
                                       args.test_batch_size,
                                       device,
                                       shuffle=False,
                                       is_test=True)
    # Prepare model
    if (args.meta_mode):
        model = MTLAbsSummarizer(args, device, checkpoint)
    else:
        model = AbsSummarizer(args, device, checkpoint)
    model.eval()

    # Prepare predictor
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              do_lower_case=True,
                                              cache_dir=args.temp_dir)
    symbols = {
        'BOS': tokenizer.vocab['[unused0]'],
        'EOS': tokenizer.vocab['[unused1]'],
        'PAD': tokenizer.vocab['[PAD]'],
        'EOQ': tokenizer.vocab['[unused2]']
    }

    predictor = build_predictor(args, tokenizer, symbols, model, logger)
    predictor.translate(test_iter, step)  # long time


def test_text_abs(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)

    checkpoint = torch.load(test_from,
                            map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    model = AbsSummarizer(args, device, checkpoint)
    model.eval()

    test_iter = data_loader.Dataloader(args,
                                       load_dataset(args,
                                                    'test',
                                                    shuffle=False),
                                       args.test_batch_size,
                                       device,
                                       shuffle=False,
                                       is_test=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              do_lower_case=True,
                                              cache_dir=args.temp_dir)
    symbols = {
        'BOS': tokenizer.vocab['[unused0]'],
        'EOS': tokenizer.vocab['[unused1]'],
        'PAD': tokenizer.vocab['[PAD]'],
        'EOQ': tokenizer.vocab['[unused2]']
    }
    predictor = build_predictor(args, tokenizer, symbols, model, logger)
    predictor.translate(test_iter, step)

