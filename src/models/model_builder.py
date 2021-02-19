"""Models and optimizers. """
import copy
import pdb

from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

from transformers import BertModel, BertConfig
from transformers.modeling_bert import BertEmbeddings

from models.decoder import TransformerDecoder
from models.encoder import Classifier, ExtTransformerEncoder, PositionalEncoding, TransformerEncoderLayer
from models.optimizers import Optimizer
from models.adapter import add_enc_adapters, add_dec_adapters, add_layer_norm
from models.adapter import AdapterConfig, Adapter_func, LayerNorm_func

#######################
##  Outer Optimizer  ##
#######################


def build_optim(args, model, checkpoint):
    """Builds shared optimizer for encoder and decoder.

    Args:
        model (models.model_builder.ABsSummarizer/MTLAbsSummarizer)
        checkpoint (dict)
    Returns:
        A optimizer in type models.optimizers.Optimizer.
    """

    # Load optimizer
    if checkpoint is not None and args.init_optim == False:
        optim = checkpoint['optim'][0]
        optim.optimizer.load_state_dict(optim.optimizer.state_dict())
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")
    else:
        # Disable warm up
        if args.outer_no_warm_up:
            optim = Optimizer(args.optim,
                              args.lr,
                              args.max_grad_norm,
                              beta1=args.beta1,
                              beta2=args.beta2)
        else:
            optim = Optimizer(args.optim,
                              args.lr,
                              args.max_grad_norm,
                              beta1=args.beta1,
                              beta2=args.beta2,
                              decay_method='noam',
                              warmup_steps=args.warmup_steps)

    # Feed parameters to be optimized
    params = list(model.named_parameters())
    optim.set_parameters(params)

    return optim


def build_optim_bert(args, model, checkpoint):
    """Builds optimizer for encoder (BERT).

    Args:
        model (models.model_builder.ABsSummarizer/MTLAbsSummarizer)
        checkpoint (dict)
    Returns:
        A optimizer in type models.optimizers.Optimizer.
    """

    # Load optimizer
    if checkpoint is not None and args.init_optim == False:
        optim = checkpoint['optims'][0]  # [0] -> encoder, [1] -> decoder
        optim.optimizer.load_state_dict(optim.optimizer.state_dict())
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")
    else:
        # Disable warm up
        if (args.outer_no_warm_up):
            optim = Optimizer(args.optim,
                              args.lr_bert,
                              args.max_grad_norm,
                              beta1=args.beta1,
                              beta2=args.beta2)
        else:
            optim = Optimizer(args.optim,
                              args.lr_bert,
                              args.max_grad_norm,
                              beta1=args.beta1,
                              beta2=args.beta2,
                              decay_method='noam',
                              warmup_steps=args.warmup_steps_bert)

    # Feed parameters to be optimized
    params = [(n, p) for n, p in list(model.named_parameters())
              if n.startswith('bert.model')]
    optim.set_parameters(params)

    return optim


def build_optim_dec(args, model, checkpoint):
    """Builds optimizer for decoder.

    Args:
        model (models.model_builder.ABsSummarizer/MTLAbsSummarizer)
        checkpoint (dict)
    Returns:
        A optimizer in type models.optimizers.Optimizer.
    """

    # Load optimizer
    if checkpoint is not None and args.init_optim == False:
        optim = checkpoint['optims'][1]  # [0] -> encoder, [1] -> decoder
        optim.optimizer.load_state_dict(optim.optimizer.state_dict())
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        # Disable warm up
        if (args.outer_no_warm_up):
            optim = Optimizer(args.optim,
                              args.lr_dec,
                              args.max_grad_norm,
                              beta1=args.beta1,
                              beta2=args.beta2)
        else:
            optim = Optimizer(args.optim,
                              args.lr_dec,
                              args.max_grad_norm,
                              beta1=args.beta1,
                              beta2=args.beta2,
                              decay_method='noam',
                              warmup_steps=args.warmup_steps_dec)

    # Feed parameters to be optimized
    params = [(n, p) for n, p in list(model.named_parameters())
              if not n.startswith('bert.model')]
    optim.set_parameters(params)

    return optim


#######################
##  Inner Optimizer  ##
#######################


def build_optim_inner(args, model, checkpoint, maml_type=None):
    """Builds shared inner optimizer for encoder and decoder.

    We don't need to load trained optimizer in inner loop.

    Args:
        model (models.model_builder.ABsSummarizer/MTLAbsSummarizer)
        checkpoint (dict)
    Returns:
        A optimizer in type models.optimizers.Optimizer.
    """

    assert maml_type == 'maml'  # only support MAML currently

    # NOTE: no warm-up
    optim = Optimizer(args.inner_optim,
                      args.lr_inner,
                      args.max_grad_norm,
                      beta1=args.beta1,
                      beta2=args.beta2)

    # NOTE: these params is pseudo, which will be replaced in forwarding
    optim.set_parameters(list(model.named_parameters()))

    return optim


def build_optim_bert_inner(args, model, checkpoint, maml_type=None):
    """Builds inner optimizer for encoder (BERT).

    We don't need to load trained optimizer in inner loop.

    Args:
        model (models.model_builder.ABsSummarizer/MTLAbsSummarizer)
        checkpoint (dict)
    Returns:
        A optimizer in type models.optimizers.Optimizer.
    """

    assert maml_type == 'maml'  # only support MAML currently

    # NOTE: no warm-up
    optim = Optimizer(args.inner_optim,
                      args.lr_bert_inner,
                      args.max_grad_norm,
                      beta1=args.beta1,
                      beta2=args.beta2)

    # NOTE: these params is pseudo, which will be replaced in forwarding
    params = [(n, p) for n, p in list(model.named_parameters())
              if n.startswith('bert.model')]
    optim.set_parameters(params)

    return optim


def build_optim_dec_inner(args, model, checkpoint, maml_type=None):
    """Builds inner optimizer for decoder.

    We don't need to load trained optimizer in inner loop.

    Args:
        model (models.model_builder.ABsSummarizer/MTLAbsSummarizer)
        checkpoint (dict)
    Returns:
        A optimizer in type models.optimizers.Optimizer.
    """

    assert maml_type == 'maml'  # only support MAML currently

    # NOTE: no warm up
    optim = Optimizer(args.inner_optim,
                      args.lr_dec_inner,
                      args.max_grad_norm,
                      beta1=args.beta1,
                      beta2=args.beta2)

    # NOTE: these params is pseudo, which will be replaced in forwarding
    params = [(n, p) for n, p in list(model.named_parameters())
              if not n.startswith('bert.model')]
    optim.set_parameters(params)

    return optim


#################
##  Generator  ##
#################


def get_generator(vocab_size, dec_hidden_size, device):
    """Builds output layer for decoder.

    Args:
        vocab_size (int)
        dec_hidden_size (int)
        device (string)

    Returns:
        A network in type torch.nn.modules.container.Sequential.
    """
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(nn.Linear(dec_hidden_size, vocab_size), gen_func)
    generator.to(device)

    return generator


######################
##  Encoder - BERT  ##
######################


class Bert(nn.Module):
    """Wraps BERT model from HuggingFace. """
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if (large):
            self.model = BertModel.from_pretrained('bert-large-uncased',
                                                   cache_dir=temp_dir)
        else:
            self.model = BertModel.from_pretrained('bert-base-uncased',
                                                   cache_dir=temp_dir)

        self.finetune = finetune

    def forward(self, x, segs, mask):
        """Forward process.

        Args:
            x (tensor(batch, max_src_len_batch)):
                A batch of token ids.
            segs (tensor(batch, max_src_len_batch)):
                corresponding segement id (0 or 1) to speparate sentences.
            mask (tensor(batch, max_src_len_batch))
                corresponding mask (0 or 1) for padding tokens.

        Returns:
            A tensor in shape (batch, max_src_len_batch, hidden_dim).
        """
        if (self.finetune):
            top_vec, _ = self.model(x,
                                    token_type_ids=segs,
                                    attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x,
                                        token_type_ids=segs,
                                        attention_mask=mask)
        return top_vec


##################
##  Summarizer  ##
##################


class AbsSummarizer(nn.Module):
    def __init__(self,
                 args,
                 device,
                 checkpoint=None,
                 bert_from_extractive=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device

        # NOTE: for document similarity
        self.doc_vecs = []

        # Initial Bert
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        # Load ckpt from extractive model
        if bert_from_extractive is not None:
            self.bert.model.load_state_dict(dict([
                (n[11:], p) for n, p in bert_from_extractive.items()
                if n.startswith('bert.model')
            ]),
                                            strict=True)

        # Default Bert
        if (args.encoder == 'baseline'):
            bert_config = BertConfig(
                self.bert.model.config.vocab_size,
                hidden_size=args.enc_hidden_size,
                num_hidden_layers=args.enc_layers,
                num_attention_heads=8,
                intermediate_size=args.enc_ff_size,
                hidden_dropout_prob=args.enc_dropout,
                attention_probs_dropout_prob=args.enc_dropout)
            self.bert.model = BertModel(bert_config)

        # The positional embedding is 512 in original Bert, repeat it for cases > 512
        if args.max_pos > 512:
            my_pos_embeddings = nn.Embedding(
                args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = \
                    self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = \
                    self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size,
                                      self.bert.model.config.hidden_size,
                                      padding_idx=0)
        if self.args.share_emb:
            tgt_embeddings.weight = copy.deepcopy(
                self.bert.model.embeddings.word_embeddings.weight)

        # Initial Transformer decoder
        self.decoder = TransformerDecoder(self.args.dec_layers,
                                          self.args.dec_hidden_size,
                                          heads=self.args.dec_heads,
                                          d_ff=self.args.dec_ff_size,
                                          dropout=self.args.dec_dropout,
                                          embeddings=tgt_embeddings)

        # Initial generator
        self.generator = get_generator(self.vocab_size,
                                       self.args.dec_hidden_size, device)
        self.generator[0].weight = self.decoder.embeddings.weight

        # Insert Adaptor modules
        if args.enc_adapter:
            enc_hidden_size = self.bert.model.embeddings.word_embeddings.weight.shape[
                1]
            config = AdapterConfig(
                hidden_size=enc_hidden_size,
                adapter_size=args.adapter_size,
                adapter_act=args.adapter_act,
                adapter_initializer_range=args.adapter_initializer_range)
            self.bert.model = add_enc_adapters(self.bert.model, config)
            self.bert.model = add_layer_norm(self.bert.model,
                                             d_model=enc_hidden_size,
                                             eps=args.layer_norm_eps)
        if args.dec_adapter:
            config = AdapterConfig(
                hidden_size=args.dec_hidden_size,
                adapter_size=args.adapter_size,
                adapter_act=args.adapter_act,
                adapter_initializer_range=args.adapter_initializer_range)
            self.decoder = add_dec_adapters(self.decoder, config)
            self.decoder = add_layer_norm(self.decoder,
                                          d_model=args.dec_hidden_size,
                                          eps=args.layer_norm_eps)
            self.generator[0].weight.requires_grad = False
            self.generator[0].bias.requires_grad = False

        # Load ckpt
        def modify_ckpt_for_enc_adapter(ckpt):
            """Modifies no-adpter ckpt for adapter-equipped encoder. """
            keys_need_modified_enc = []
            for k in list(ckpt['model'].keys()):
                if ('output' in k):
                    keys_need_modified_enc.append(k)
            for mk in keys_need_modified_enc:
                ckpt['model'] = OrderedDict([
                    (mk.replace('output', 'output.self_output'),
                     v) if k == mk else (k, v)
                    for k, v in ckpt['model'].items()
                ])

        def modify_ckpt_for_dec_adapter(ckpt):
            """Modifies no-adpter ckpt for adapter-equipped decoder. """
            keys_need_modified_dec = []
            for k in list(ckpt['model'].keys()):
                if ('layers' in k):
                    keys_need_modified_dec.append(k)
            for mk in keys_need_modified_dec:
                p = mk.find('layers.')
                new_k = mk[:p + 8] + '.dec_layer' + mk[p + 8:]
                ckpt['model'] = OrderedDict([(new_k, v) if k == mk else (k, v)
                                             for k, v in ckpt['model'].items()
                                             ])

        def identify_unmatched_keys(ckpt1, ckpt2):
            """Report the unmatched keys in ckpt1 for loading ckpt2 to ckpt1. (debug use) """
            fp = open("unmatched_keys.txt", 'w')
            num = 0
            ckpt1_keys = list(ckpt1.keys())
            ckpt2_keys = list(ckpt2.keys())
            for k in ckpt1_keys:
                if not (k in ckpt2_keys) and not ("var" in k) and not (
                        "feed_forward" in k):
                    # NOTE: since var and feed_forward use shared weights from other modules
                    fp.write(k + '\n')
                    print(k)
                    num += 1
            print("# of Unmatched Keys: {}".format(num))
            fp.close()

        if checkpoint is not None:
            if (self.args.enc_adapter and self.args.ckpt_from_no_adapter):
                modify_ckpt_for_enc_adapter(checkpoint)
            if (self.args.dec_adapter and self.args.ckpt_from_no_adapter):
                modify_ckpt_for_dec_adapter(checkpoint)

            # NOTE: not strict for load model
            #identify_unmatched_keys(self.state_dict(), checkpoint['model'])
            self.load_state_dict(checkpoint['model'], strict=False)
        else:
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            if (args.use_bert_emb):
                tgt_embeddings = nn.Embedding(
                    self.vocab_size,
                    self.bert.model.config.hidden_size,
                    padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(
                    self.bert.model.embeddings.word_embeddings.weight)
                self.decoder.embeddings = tgt_embeddings
                self.generator[0].weight = self.decoder.embeddings.weight

        self.to(device)

    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
        """Forward process.

        Args:
            src (tensor(batch, max_src_len_batch)):
                Source token ids.
            tgt (tensor(batch, max_tgt_len_batch)):
                Target token ids.
            segs (tensor(batch, max_src_len_batch)):
                Segement id (0 or 1) to speparate source sentences.
            clss (tensor(batch, max_cls_num_batch)):
                the position of [CLS] token.
            mask_src (tensor(batch, max_src_len_batch))
                Mask (0 or 1) for source padding tokens.
            mask_tgt (tensor(batch, max_tgt_len_batch))
                Mask (0 or 1) for target padding tokens.
            mask_cls (tensor(batch, max_cls_num_batch)):
                Mask (0 or 1) for [CLS] position.

        Returns:
            decoder_outputs (tensor(batch, max_tgt_len_batch, dec_hidden_dim)):
                The hidden states from decoder.
        """
        top_vec = self.bert(src, segs, mask_src)
        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)

        return decoder_outputs, None


########################
##  Summarizer - MTL  ##
########################


class MTLAbsSummarizer(nn.Module):
    def __init__(self,
                 args,
                 device,
                 checkpoint=None,
                 bert_from_extractive=None):
        super(MTLAbsSummarizer, self).__init__()
        self.args = args
        self.device = device

        # Initial Bert
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        # Load ckpt from extractive model
        if bert_from_extractive is not None:
            self.bert.model.load_state_dict(dict([
                (n[11:], p) for n, p in bert_from_extractive.items()
                if n.startswith('bert.model')
            ]),
                                            strict=True)

        # Default Bert
        if args.encoder == 'baseline':
            bert_config = BertConfig(
                self.bert.model.config.vocab_size,
                hidden_size=args.enc_hidden_size,
                num_hidden_layers=args.enc_layers,
                num_attention_heads=8,
                intermediate_size=args.enc_ff_size,
                hidden_dropout_prob=args.enc_dropout,
                attention_probs_dropout_prob=args.enc_dropout)
            self.bert.model = BertModel(bert_config)

        # The positional embedding is 512 in original Bert, repeat it for cases > 512
        if (args.max_pos > 512):
            my_pos_embeddings = nn.Embedding(
                args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = \
                    self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = \
                    self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size,
                                      self.bert.model.config.hidden_size,
                                      padding_idx=0)
        if self.args.share_emb:
            tgt_embeddings.weight = copy.deepcopy(
                self.bert.model.embeddings.word_embeddings.weight)

        # Initial Transformer decoder
        self.decoder = TransformerDecoder(self.args.dec_layers,
                                          self.args.dec_hidden_size,
                                          heads=self.args.dec_heads,
                                          d_ff=self.args.dec_ff_size,
                                          dropout=self.args.dec_dropout,
                                          embeddings=tgt_embeddings)

        # Initial generator
        self.generator = get_generator(self.vocab_size,
                                       self.args.dec_hidden_size, device)
        self.generator[0].weight = self.decoder.embeddings.weight

        # Insert Adaptor modules
        if (args.enc_adapter):
            enc_hidden_size = self.bert.model.embeddings.word_embeddings.weight.shape[
                1]
            config = AdapterConfig(
                hidden_size=enc_hidden_size,
                adapter_size=args.adapter_size,
                adapter_act=args.adapter_act,
                adapter_initializer_range=args.adapter_initializer_range)
            self.bert.model = add_enc_adapters(self.bert.model, config)
            self.bert.model = add_layer_norm(self.bert.model,
                                             d_model=enc_hidden_size,
                                             eps=args.layer_norm_eps)
        if (args.dec_adapter):
            config = AdapterConfig(
                hidden_size=args.dec_hidden_size,
                adapter_size=args.adapter_size,
                adapter_act=args.adapter_act,
                adapter_initializer_range=args.adapter_initializer_range)
            self.decoder = add_dec_adapters(self.decoder, config)
            self.decoder = add_layer_norm(self.decoder,
                                          d_model=args.dec_hidden_size,
                                          eps=args.layer_norm_eps)

            self.generator[0].weight.requires_grad = False
            self.generator[0].bias.requires_grad = False

        # Load ckpt
        def modify_ckpt_for_enc_adapter(ckpt):
            """Modifies no-adpter ckpt for adapter-equipped encoder. """
            keys_need_modified_enc = []
            for k in list(ckpt['model'].keys()):
                if ('output' in k):
                    keys_need_modified_enc.append(k)
            for mk in keys_need_modified_enc:
                ckpt['model'] = OrderedDict([
                    (mk.replace('output', 'output.self_output'),
                     v) if k == mk else (k, v)
                    for k, v in ckpt['model'].items()
                ])

        def modify_ckpt_for_dec_adapter(ckpt):
            """Modifies no-adpter ckpt for adapter-equipped decoder. """
            keys_need_modified_dec = []
            for k in list(ckpt['model'].keys()):
                if ('layers' in k):
                    keys_need_modified_dec.append(k)
            for mk in keys_need_modified_dec:
                p = mk.find('layers.')
                new_k = mk[:p + 8] + '.dec_layer' + mk[p + 8:]
                ckpt['model'] = OrderedDict([(new_k, v) if k == mk else (k, v)
                                             for k, v in ckpt['model'].items()
                                             ])

        def identify_unmatched_keys(ckpt1, ckpt2):
            """Report the unmatched keys in ckpt1 for loading ckpt2 to ckpt1. (debug use) """
            fp = open("unmatched_keys.txt", 'w')
            num = 0
            ckpt1_keys = list(ckpt1.keys())
            ckpt2_keys = list(ckpt2.keys())
            for k in ckpt1_keys:
                if not (k in ckpt2_keys) and not ("var" in k) and not (
                        "feed_forward" in k):
                    # NOTE: since var and feed_forward use shared weights from other modules
                    fp.write(k + '\n')
                    print(k)
                    num += 1
            print("# of Unmatched Keys: {}".format(num))
            fp.close()

        if checkpoint is not None:
            if (self.args.enc_adapter and self.args.ckpt_from_no_adapter):
                modify_ckpt_for_enc_adapter(checkpoint)
            if (self.args.dec_adapter and self.args.ckpt_from_no_adapter):
                modify_ckpt_for_dec_adapter(checkpoint)

            # NOTE: not strict for load model
            #identify_unmatched_keys(self.state_dict(), checkpoint['model']) # DEBUG
            self.load_state_dict(checkpoint['model'], strict=False)
        else:
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            if (args.use_bert_emb):
                tgt_embeddings = nn.Embedding(
                    self.vocab_size,
                    self.bert.model.config.hidden_size,
                    padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(
                    self.bert.model.embeddings.word_embeddings.weight)
                self.decoder.embeddings = tgt_embeddings
                self.generator[0].weight = self.decoder.embeddings.weight

        self.to(device)

    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
        """Forward process.

        Args:
            src (tensor(batch, max_src_len_batch)):
                Source token ids.
            tgt (tensor(batch, max_tgt_len_batch)):
                Target token ids.
            segs (tensor(batch, max_src_len_batch)):
                Segement id (0 or 1) to speparate source sentences.
            clss (tensor(batch, max_cls_num_batch)):
                the position of [CLS] token.
            mask_src (tensor(batch, max_src_len_batch))
                Mask (0 or 1) for source padding tokens.
            mask_tgt (tensor(batch, max_tgt_len_batch))
                Mask (0 or 1) for target padding tokens.
            mask_cls (tensor(batch, max_cls_num_batch)):
                Mask (0 or 1) for [CLS] position.

        Returns:
            A tuple of variable:
                decoder_outputs (tensor(batch, max_tgt_len_batch, dec_hidden_dim)):
                    The hidden states from decoder.
                top_vec (tensor(batch, max_src_len_batch, enc_hidden_dim)):
                    The hidden states from encoder.
        """
        # top_vec -> tensor(batch, max_src_len_batch, enc_hidden_dim)
        top_vec = self.bert(src, segs, mask_src)

        # dec_state -> models.decoder.TransformerDecoderState
        dec_state = self.decoder.init_decoder_state(src, top_vec)

        # decoder_outputs -> tensor(batch, max_tgt_len_batch, dec_hidden_dim)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)

        return decoder_outputs, top_vec

    # [For Inner Loop]
    def _cascade_fast_weights_grad(self, fast_weights):
        """Sets fast-weight mode for adapter and layer norm modules. """
        offset = 0
        for name, sub_module in self.named_modules():
            if isinstance(
                    sub_module,
                (Adapter_func, LayerNorm_func)) and sub_module.trainable:
                param_num = len(sub_module._parameters)
                setattr(sub_module, 'fast_weights_flag', True)
                delattr(sub_module, 'fast_weights')
                setattr(sub_module, 'fast_weights',
                        fast_weights[offset:offset + param_num])
                offset += param_num
        return offset

    # [For Outer Loop]
    def _clean_fast_weights_mode(self):
        """Cleans fast-weight mode for adapter and layer norm modules. """
        module_num = 0
        for name, sub_module in self.named_modules():
            if isinstance(
                    sub_module,
                (Adapter_func, LayerNorm_func)) and sub_module.trainable:
                setattr(sub_module, 'fast_weights_flag', False)
                delattr(sub_module, 'fast_weights')
                setattr(sub_module, 'fast_weights', None)
                module_num += 1
        return module_num

    def _adapter_fast_weights(self):
        """Returns fast (task) weights from full model. """
        for name, sub_module in self.named_modules():
            if isinstance(
                    sub_module,
                (Adapter_func, LayerNorm_func)) and sub_module.trainable:
                for param in sub_module.fast_weights:
                    yield param

    def _adapter_fast_weights_bert(self):
        """Returns fast (task) weights from encoder. """
        for name, sub_module in self.bert.named_modules():
            if isinstance(
                    sub_module,
                (Adapter_func, LayerNorm_func)) and sub_module.trainable:
                for param in sub_module.fast_weights:
                    yield param

    def _adapter_fast_weights_dec(self):
        """Returns fast (task) weights from decoder. """
        for name, sub_module in self.decoder.named_modules():
            if isinstance(
                    sub_module,
                (Adapter_func, LayerNorm_func)) and sub_module.trainable:
                for param in sub_module.fast_weights:
                    yield param

    def _adapter_vars(self):
        """Returns true (meta) parameters from full model. """
        for name, sub_module in self.named_modules():
            if isinstance(
                    sub_module,
                (Adapter_func, LayerNorm_func)) and sub_module.trainable:
                for param in sub_module.vars:
                    yield param

    def _adapter_vars_bert(self):
        """Returns true (meta) parameters from encoder. """
        for name, sub_module in self.bert.named_modules():
            if isinstance(
                    sub_module,
                (Adapter_func, LayerNorm_func)) and sub_module.trainable:
                for param in sub_module.vars:
                    yield param

    def _adapter_vars_dec(self):
        """Returns true (meta) parameters from decoder. """
        for name, sub_module in self.decoder.named_modules():
            if isinstance(
                    sub_module,
                (Adapter_func, LayerNorm_func)) and sub_module.trainable:
                for param in sub_module.vars:
                    yield param
