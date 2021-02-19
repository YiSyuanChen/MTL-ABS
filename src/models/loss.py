"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.

From https://opennmt.net/OpenNMT-py/_modules/onmt/utils/loss.html#LossComputeBase
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from models.reporter import Statistics
import math
import pdb

def abs_loss(generator, symbols, vocab_size, device, train=True, label_smoothing=0.0):
    """Returns the NMLLossCompute instance. """
    compute = NMTLossCompute(
        generator, symbols, vocab_size,
        label_smoothing=label_smoothing if train else 0.0)
    compute.to(device)
    return compute

# From https://github.com/lonePatient/label_smoothing_pytorch/blob/master/lsr.py
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)


class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations

    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, generator, pad_id):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.padding_idx = pad_id

    def _make_shard_state(self, batch, output,  attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    # [For Validation]
    def monolithic_compute_loss(self, batch, output):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions
              `[tgt_len x batch x src_len]`
        Returns:
            :obj:`onmt.utils.Statistics`: loss statistics
        """
        shard_state = self._make_shard_state(batch, output)
        _, batch_stats = self._compute_loss(batch, **shard_state)

        return batch_stats

    # [For Inner Loop]
    def monolithic_compute_loss_return(self, batch, output):

        shard_state = self._make_shard_state(batch, output)
        loss, batch_stats = self._compute_loss(batch, **shard_state)

        return loss, batch_stats

    # [For Outer Loop]
    def monolithic_compute_loss_backprop(self, batch, output,
                                         normalization):

        shard_state = self._make_shard_state(batch, output)
        loss, batch_stats = self._compute_loss(batch, **shard_state)
        if isinstance(normalization, torch.Tensor):
            loss.div(normalization).backward(retain_graph=True) # for task-norm
        else:
            loss.div(float(normalization)).backward(retain_graph=True)
        return batch_stats

    def sharded_compute_loss(self, batch, output,
                              shard_size,
                             normalization):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard
          normalization (int) : Loss is divided by this number

        Returns:
            :obj:`onmt.utils.Statistics`: validation loss statistics

        """
        batch_stats = Statistics()
        # Type:
        # normaliztion          = number of non-padding words in batch
        # shard_state           -> dict, keys=['output', 'target']
        # shard_state['output'] -> tensor(batch_size, max_tgt_len_batch, hidden_size)
        # shard_state['target'] -> tensor(batch_size, max_tgt_len_batch)
        shard_state = self._make_shard_state(batch, output) # remove the first token of tgt
        for shard in shards(shard_state, shard_size): # generator
            # NOTE: the default shard size = 32, which is greater than practical batch size (around 4~6)
            #       thus basically there is only one iteration here.
            # Type: 
            # shard           -> dict, keys=['output', 'target']
            # shard['output'] -> tensor(shard_size, max_tgt_len_batch, hidden_size)
            # shard['target'] -> tensor(shard_size, max_tgt_len_batch)
            loss, stats = self._compute_loss(batch, **shard)
            loss.div(float(normalization)).backward()
            batch_stats.update(stats)
        return batch_stats

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        # Type:
        # loss              -> tensor(1,1)
        # pred, non_padding -> tensor(batch_size * max_tgt_len_batch)
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx) # filter out non padding word (word_id != 0)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum() \
                          .item()
        num_non_padding = non_padding.sum().item()
        return Statistics(loss.item(), num_non_padding, num_correct)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, generator, symbols, vocab_size,
                 label_smoothing=0.0):
        super(NMTLossCompute, self).__init__(generator, symbols['PAD'])
        self.sparse = not isinstance(generator[1], nn.LogSoftmax)
        # NOTE: adaptive embedding
        #self.sparse = not isinstance(generator.gen_func, nn.LogSoftmax)
        if label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(
                label_smoothing, vocab_size, ignore_index=self.padding_idx
            )
        else:
            self.criterion = nn.NLLLoss(
                ignore_index=self.padding_idx, reduction='sum'
            )

    def _make_shard_state(self, batch, output):
        # Type
        # output          -> tensor(batch_size, max_tgt_len_batch, hidden_size)
        # btach.tgt       -> tensor(batch_size, max_tgt_len_batch + 1)
        # batch.tgt[:,1:] -> tensor(batch_size, max_tgt_len_batch)
        return {
            "output": output,
            "target": batch.tgt[:,1:], # remove first token (word_id = 1)
        }

    def _compute_loss(self, batch, output, target):
        # Type:
        # output        -> tensor(batch_size, max_tgt_len_batch, hidden_size)
        # bottle_output -> tensor(batch_size * max_tgt_len_batch, hidden_size)
        # gtruth        -> tensor(batch_size * max_tgt_len_batch)
        bottled_output = self._bottle(output)
        scores = self.generator(bottled_output)
        gtruth = target.contiguous().view(-1)

        loss = self.criterion(scores, gtruth)
        stats = self._stats(loss.clone(), scores, gtruth)

        return loss, stats

    def _compute_loss_ensemble(self, batch, scores, target):
        gtruth = target.contiguous().view(-1)

        loss = self.criterion(scores, gtruth)
        stats = self._stats(loss.clone(), scores, gtruth)

        return loss, stats

def filter_shard_state(state, shard_size=None):
    """ ? """
    # Type: 
    # state           -> dict, keys=['output', 'target']
    # state['output'] -> tensor(batch_size, max_tgt_len_batch, hidden_size)
    # state['target'] -> tensor(batch_size, max_tgt_len_batch)
    # k               -> 'output', 'target'
    # v               -> state['output'], state['target']
    for k, v in state.items():
        if shard_size is None:
            yield k, v
        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                # Type
                # torch.split(v,shard_size) -> tuple(tensor, tensor, ...)
                # v_chunck                  -> tensor(shard_size, ~)
                # Note: the shard are basically no effect for abs summ, 
                #       since the shard_size (default=32) > real batch size (1~6)
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone() # break the computation graph
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            # Type
            # v_split -> list[tensor(shard_size,~), ...]
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """

    # Type: 
    # state           -> dict, keys=['output', 'target']
    # state['output'] -> tensor(batch_size, max_tgt_len_batch, hidden_size)
    # state['target'] -> tensor(batch_size, max_tgt_len_batch)
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        # Type:
        # non_none           -> dict, keys=['output', 'target']
        # non_none['output'] -> tuple(tensor, list[shard_tensor, ...])
        # non_none['target'] -> tuple(tensor, list[shard_tensor, ...])
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        # Type:
        # keys         -> tuple('output','target')
        # values       -> tuple(shard_output, shard_target)
        # shard_output -> list[shard_tensor, ...]
        # shard_target -> list[shard_tensor, ...]
        keys, values = zip(*( (k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items() ) )

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        # Type:
        # shard_tensors -> tuple(shard_output_tensor, shard_target_tensor)
        for shard_tensors in zip(*values):
            # Type:
            # dict, keys   = ['output', 'target']
            #     , values = [shard_output_tensor, shard_target_tensor]
            # Note: the output format is identical to input "state" (batch size -> shard_size)
            yield dict(zip(keys, shard_tensors))


        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)

