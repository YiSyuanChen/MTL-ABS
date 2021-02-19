"""Main entry file. """

from __future__ import division
import argparse
import os

from others.logging import init_logger
from others.utils import str2bool
from train_abstractive import train_abs, validate_abs, test_abs, test_text_abs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", default='abs', type=str,
                        choices=['abs'])  # FIXME: include extractive training
    parser.add_argument("-mode",
                        default='train',
                        type=str,
                        choices=['train', 'validate', 'test'])

    # File path
    parser.add_argument("-bert_data_path", default='../datasets')
    parser.add_argument("-model_path", default='../models')
    parser.add_argument('-log_path', default='../logs')
    parser.add_argument("-result_path", default='../results')
    parser.add_argument("-temp_dir", default='../temp')

    # Initialization
    parser.add_argument("-param_init",
                        default=0,
                        type=float,
                        help="Intialize with uniform distribution")
    parser.add_argument(
        "-param_init_glorot",
        default=True,
        type=str2bool,
        nargs='?',
        const=True,
        help="Initialize with Glorot initialization (overwritten param_init)")

    # Training
    parser.add_argument("-batch_size", default=140, type=int)
    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-load_from_extractive", default='', type=str)
    parser.add_argument("-train_from", default='')

    # Validation & Testing
    parser.add_argument("-test_batch_size", default=200, type=int)
    parser.add_argument("-alpha", default=0.7, type=float)
    parser.add_argument("-beam_size", default=5, type=int)
    parser.add_argument("-min_length", default=5, type=int)
    parser.add_argument("-max_length", default=256, type=int)
    parser.add_argument("-max_tgt_len", default=256, type=int)
    parser.add_argument("-test_all",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False)
    parser.add_argument("-test_from", default='')
    parser.add_argument("-test_start_from", default=-1, type=int)
    parser.add_argument("-report_rouge",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=True)
    parser.add_argument("-block_trigram",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=True,
                        help="Avoid tri-gram repeatition in prediction")
    parser.add_argument("-recall_eval",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False,
                        help="""Control # of sentence in prediction,
                            such that # of word are close to target.""")

    # Report & Save
    parser.add_argument("-train_steps", default=1000, type=int)
    parser.add_argument("-save_checkpoint_steps", default=5, type=int)
    parser.add_argument("-report_every", default=1, type=int)

    # BERT
    parser.add_argument("-encoder",
                        default='bert',
                        type=str,
                        choices=['bert', 'baseline'])
    parser.add_argument("-max_pos", default=512, type=int)
    parser.add_argument("-use_interval",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=True)
    parser.add_argument("-large",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False, help="bert-base or bert-large")
    parser.add_argument("-use_bert_emb",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False,help="Use pre-trained BERT emb")
    parser.add_argument("-share_emb",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False,help="share emb between encoder and decoder")
    parser.add_argument("-finetune_bert",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=True)

    # Abstractive Model
    parser.add_argument("-enc_dropout", default=0.2, type=float)
    parser.add_argument("-enc_layers", default=6, type=int)
    parser.add_argument("-enc_hidden_size", default=512, type=int)
    parser.add_argument("-enc_ff_size", default=512, type=int)
    parser.add_argument("-dec_dropout", default=0.2, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=8, type=int)
    parser.add_argument("-dec_ff_size", default=2048, type=int)

    # Optimization
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-inner_optim", default='inner_adam', type=str)
    parser.add_argument("-sep_optim",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False)
    parser.add_argument("-lr", default=2e-3, type=float)
    parser.add_argument("-lr_bert", default=2e-3, type=float)
    parser.add_argument("-lr_dec", default=2e-3, type=float)
    parser.add_argument("-beta1", default=0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-warmup_steps", default=1500, type=int)
    parser.add_argument("-warmup_steps_bert", default=1500, type=int)
    parser.add_argument("-warmup_steps_dec", default=1500, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-generator_shard_size", default=32, type=int)

    # Adapter related
    parser.add_argument("-enc_adapter",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False)
    parser.add_argument("-dec_adapter",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False)
    parser.add_argument("-ckpt_from_no_adapter",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False)
    parser.add_argument("-adapter_size", default=64, type=int)
    parser.add_argument("-adapter_act", default='relu', type=str)
    parser.add_argument("-adapter_initializer_range", default=1e-2, type=float)
    parser.add_argument("-layer_norm_eps", default=1e-6, type=float)

    # MAML
    parser.add_argument("-meta_mode",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False)
    parser.add_argument("-maml_type",
                        default='maml',
                        type=str,
                        choices=['maml'])  # FIXME: support different MAML type
    parser.add_argument("-num_task", default=3, type=int)
    parser.add_argument("-num_batch_in_task", default=5, type=int)

    # Inner loop
    parser.add_argument("-init_optim",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False)
    parser.add_argument("-lr_inner", default=2e-3, type=float)
    parser.add_argument("-lr_bert_inner", default=2e-3, type=float)
    parser.add_argument("-lr_dec_inner", default=2e-3, type=float)
    parser.add_argument("-inner_train_steps", default=4, type=int)
    parser.add_argument("-report_inner_every", default=4, type=int)
    parser.add_argument("-inner_no_warm_up",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False)

    # Outer loop
    parser.add_argument("-outer_no_warm_up",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False)

    # Aux
    parser.add_argument("-deterministic_batch_size",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False)
    parser.add_argument("-valid_all_test_num", default=5, type=int)

    # System
    parser.add_argument('-visible_gpus',
                        default='-1',
                        type=str,
                        help="GPU IDs, ex: 0,1,2 ")
    parser.add_argument('-gpu_ranks',
                        default='0',
                        type=str,
                        help="will be overwritten by visible_gpus")
    parser.add_argument('-seed', default=168, type=int)

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    DEVICE = "cpu" if args.visible_gpus == '-1' else "cuda"
    DEVICE_ID = 0 if DEVICE == "cuda" else -1

    # Create directories
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.result_path, exist_ok=True)

    # Train/valid/test
    if args.task == 'abs':
        if args.mode == 'train':
            init_logger(os.path.join(args.log_path, 'train.log'))
            train_abs(args, DEVICE_ID)
        elif args.mode == 'validate':
            init_logger(os.path.join(args.log_path, 'valid.log'))
            validate_abs(args, DEVICE_ID)
        elif args.mode == 'test':
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                print("Not correct model name (EX: model_step_1200.pt)")
            init_logger(
                os.path.join(args.log_path, 'test.' + str(step) + '.log'))
            test_abs(args, DEVICE_ID, cp, step)
        elif args.mode == 'test_text':
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                print("Not correct model name (EX: model_step_1200.pt)")
            init_logger(
                os.path.join(args.log_path, 'test_text.' + str(step) + '.log'))
            test_text_abs(args, DEVICE_ID, cp, step)
