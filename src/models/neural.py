import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


def aeq(*args):
    """ Assert all arguments have the same value (debug purpose). """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def gelu(x):
    """ Gaussian Error Linear Unit (GELU). """
    return 0.5 * x * (1 + torch.tanh(
        math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        d_model (int):
            the size of input for the first-layer of the FFN.
        d_ff (int):
            the hidden layer size of the second-layer of the FFN.
        dropout (float):
            dropout probability in [0, 1).
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x

class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.
    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.
    .. mermaid::
       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O
    Also includes several additional tricks.
    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1, use_final_linear=True):

        super(MultiHeadedAttention, self).__init__()
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.use_final_linear = use_final_linear
        if (self.use_final_linear):
            self.final_linear = nn.Linear(model_dim, model_dim)

        self.debug = False

    def forward(self, key, value, query, mask=None,
                layer_cache=None, type=None, predefined_graph_1=None):
        """
        Compute the context vector and the attention vectors.
        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :
           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """

        batch_size = key.size(0)
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """  Reshape feature into head-seperated form.

            Args:
                x : shape=(batch_size, seq_len, hidden_dim)
            Returns:
                tensor with shape = (batch_size, head_count,
                                        seq_len, hidden_dim/head_count)
            """
            return x.view(batch_size, -1, self.head_count, self.dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """  Reshape feature into head-combined form.

            Args:
                x : shape=(batch_size, head_count,
                            seq_len, hidden_dim/head_count)
            Returns:
                tensor with shape = (batch_size, seq_len, hidden_dim)
            """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, self.head_count * self.dim_per_head)

        # [For decoding] use attention cache to accelerate decoding
        if layer_cache is not None:
            if type == "self":
                # Query/key/value are identical before tranformation
                # The sequence length is always 1
                query, key, value = self.linear_query(query), \
                                    self.linear_keys(query), \
                                    self.linear_values(query)
                query = shape(query)
                key = shape(key)
                value = shape(value)

                # Append already calculated keys/values with new key/value
                device = key.device
                if layer_cache["self_keys"] is not None:
                    key = torch.cat(
                        (layer_cache["self_keys"].to(device), key),
                        dim=2)
                if layer_cache["self_values"] is not None:
                    value = torch.cat(
                        (layer_cache["self_values"].to(device), value),
                        dim=2)
                layer_cache["self_keys"] = key
                layer_cache["self_values"] = value

            elif type == "context":
                # The keys and values should come from encoder
                query = self.linear_query(query)
                query = shape(query)

                # Only need to calculate encoder keys and values once
                if layer_cache["memory_keys"] is None:
                    key, value = self.linear_keys(key), \
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
                else:
                    key, value = layer_cache["memory_keys"], \
                                 layer_cache["memory_values"]
                layer_cache["memory_keys"] = key
                layer_cache["memory_values"] = value
        else:
            query = self.linear_query(query)
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = shape(query)
            key = shape(key)
            value = shape(value)

        # Scaled dot product attention 
        query = query / math.sqrt(self.dim_per_head)
        # scores, shape = (bathc_size,head_count,seq_len,seq_len)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            # Expand mask for all heads
            mask = mask.unsqueeze(1).expand_as(scores)
            # Assign small value for masked token
            scores = scores.masked_fill(mask, -1e18)

        # Softmax, Dropout and weighted sum for value
        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        if (self.use_final_linear):
            context = unshape(torch.matmul(drop_attn, value))
            output = self.final_linear(context)
            return output
        else:
            context = torch.matmul(drop_attn, value)
            return context


class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.

    Modules need to implement this to utilize beam search decoding.
    """
    def detach(self):
        """ Need to document this """
        self.hidden = tuple([_.detach() for _ in self.hidden])
        self.input_feed = self.input_feed.detach()

    def beam_update(self, idx, positions, beam_size):
        """ Need to document this """
        for e in self._all:
            sizes = e.size()
            br = sizes[1]
            if len(sizes) == 3:
                sent_states = e.view(sizes[0], beam_size, br // beam_size,
                                     sizes[2])[:, :, idx]
            else:
                sent_states = e.view(sizes[0], beam_size, br // beam_size,
                                     sizes[2], sizes[3])[:, :, idx]

            sent_states.data.copy_(sent_states.data.index_select(1, positions))

    def map_batch_fn(self, fn):
        raise NotImplementedError()
