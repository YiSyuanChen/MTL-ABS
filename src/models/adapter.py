import logging
import numpy as np
import copy
from typing import NamedTuple, Callable, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from transformers.activations import ACT2FN
from transformers.modeling_bert import BertLayerNorm, BertSelfOutput
from transformers import BertModel
from models.decoder import TransformerDecoderLayer, TransformerDecoder
from models.neural import PositionwiseFeedForward, MultiHeadedAttention
import pdb

logging.basicConfig(level=logging.INFO)

##############
## Adapter  ##
##############

class AdapterConfig(NamedTuple):
    hidden_size: int
    adapter_size: int
    adapter_act: Union[str, Callable]
    adapter_initializer_range: float

class Adapter(nn.Module):
    def __init__(self, config: AdapterConfig):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(config.hidden_size, config.adapter_size)
        nn.init.normal_(self.down_project.weight, std=config.adapter_initializer_range)
        nn.init.zeros_(self.down_project.bias)

        if isinstance(config.adapter_act, str):
            self.activation = ACT2FN[config.adapter_act]
        else:
            self.activation = config.adapter_act

        self.up_project = nn.Linear(config.adapter_size, config.hidden_size)
        nn.init.normal_(self.up_project.weight, std=config.adapter_initializer_range)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, hidden_states):
        down_projected = self.down_project(hidden_states)
        activated = self.activation(down_projected)
        up_projected = self.up_project(activated)
        return hidden_states + up_projected

# Pure adapter
class Adapter_func(nn.Module):
    def __init__(self, config: AdapterConfig):
        super(Adapter_func, self).__init__()

        self.config = config
        self.vars = nn.ParameterList()

        # Private side
        self.down_project_w = nn.Parameter(torch.ones([config.adapter_size, config.hidden_size]))
        self.down_project_b = nn.Parameter(torch.zeros(config.adapter_size))
        self.up_project_w   = nn.Parameter(torch.ones([config.hidden_size, config.adapter_size]))
        self.up_project_b   = nn.Parameter(torch.zeros(config.hidden_size))
        nn.init.normal_(self.down_project_w, std=config.adapter_initializer_range)
        nn.init.normal_(self.up_project_w, std=config.adapter_initializer_range)
        nn.init.zeros_(self.down_project_b)
        nn.init.zeros_(self.up_project_b)
        self.vars.append(self.down_project_w)
        self.vars.append(self.down_project_b)
        self.vars.append(self.up_project_w)
        self.vars.append(self.up_project_b)

        # Fast weight setting
        self.fast_weights      = None
        self.fast_weights_flag = False
        self.trainable         = True

        # For debug
        self.check = False

    def forward(self, hidden_states):
        if(self.fast_weights_flag == False):
            temp_vars = self.vars
        else:
            temp_vars = self.fast_weights

        down_projected = F.linear(hidden_states, temp_vars[0], temp_vars[1])
        activated      = F.relu(down_projected)
        up_projected   = F.linear(activated, temp_vars[2], temp_vars[3])

        outputs = hidden_states + up_projected
 
        return outputs

# ===== Layer Norm =====

class LayerNorm_func(nn.Module):
    def __init__(self, d_model, eps):
        super(LayerNorm_func, self).__init__()
        self.vars = nn.ParameterList()
        self.weight = nn.Parameter(torch.ones([d_model]))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.vars.append(self.weight)
        self.vars.append(self.bias)
        self.d_model = d_model
        self.eps = eps

        self.fast_weights = None
        self.fast_weights_flag = False

        self.trainable = True

    def forward(self, hidden_states):
        if(self.fast_weights_flag == False):
            temp_vars = self.vars
        else:
            temp_vars = self.fast_weights # assign from reparam function

        hidden_states = F.layer_norm(hidden_states, normalized_shape=(self.d_model,),
                                     weight=temp_vars[0], bias=temp_vars[1], eps=self.eps)
        return hidden_states

# ===== Adapted BERT Layer =====

class BertAdaptedSelfOutput(nn.Module):
    def __init__(self,
                 self_output: BertSelfOutput,
                 config: AdapterConfig):
        super(BertAdaptedSelfOutput, self).__init__()
        self.self_output = self_output
        self.adapter = Adapter_func(config)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.self_output.dense(hidden_states)
        hidden_states = self.self_output.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.self_output.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# ===== Adapted Decoder Layer

class PositionwiseAdaptedFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
    """

    def __init__(self,
                 feed_forward: PositionwiseFeedForward,
                 config: AdapterConfig):
        super(PositionwiseAdaptedFeedForward, self).__init__()
        self.feed_forward = feed_forward
        self.adapter = Adapter_func(config)

    def forward(self, x):
        inter = self.feed_forward.dropout_1(self.feed_forward.actv(self.feed_forward.w_1(self.feed_forward.layer_norm(x))))
        output = self.feed_forward.dropout_2(self.feed_forward.w_2(inter))
        output = self.adapter(output) + x
        return output

class TransformerAdaptedDecoderLayer(nn.Module):

    def __init__(self,
                 dec_layer: TransformerDecoderLayer,
                 config: AdapterConfig):
        super(TransformerAdaptedDecoderLayer, self).__init__()
        self.dec_layer = dec_layer

        # Adapter modules
        self.adapted_ff = PositionwiseAdaptedFeedForward(self.dec_layer.feed_forward,config)
        self.adapter_1 = Adapter_func(config)
        self.adapter_2 = Adapter_func(config)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                previous_input=None, layer_cache=None, step=None):
        """
        Args:
            inputs (`FloatTensor`): `[batch_size x 1 x model_dim]`
            memory_bank (`FloatTensor`): `[batch_size x src_len x model_dim]`
            src_pad_mask (`LongTensor`): `[batch_size x 1 x src_len]`
            tgt_pad_mask (`LongTensor`): `[batch_size x 1 x 1]`

        Returns:
            (`FloatTensor`, `FloatTensor`, `FloatTensor`):

            * output `[batch_size x 1 x model_dim]`
            * attn `[batch_size x 1 x src_len]`
            * all_input `[batch_size x current_step x model_dim]`

        """
        dec_mask = torch.gt(tgt_pad_mask +
                            self.dec_layer.mask[:, :tgt_pad_mask.size(1),
                                      :tgt_pad_mask.size(1)], 0)
        input_norm = self.dec_layer.layer_norm_1(inputs)
        all_input = input_norm
        if previous_input is not None:
            all_input = torch.cat((previous_input, input_norm), dim=1)
            dec_mask = None

        query = self.dec_layer.self_attn(all_input, all_input, input_norm,
                                     mask=dec_mask,
                                     layer_cache=layer_cache,
                                     type="self")

        query = self.adapter_1(self.dec_layer.drop(query)) + inputs

        query_norm = self.dec_layer.layer_norm_2(query)
        mid = self.dec_layer.context_attn(memory_bank, memory_bank, query_norm,
                                      mask=src_pad_mask,
                                      layer_cache=layer_cache,
                                      type="context")
        #output = self.feed_forward(self.drop(mid) + query)
        output = self.adapter_2(self.dec_layer.drop(mid)) + query
        output = self.adapted_ff(output)
        return output, all_input

# ===== Functions to add adapter =====

def adapt_bert_self_output(config: AdapterConfig):
    return lambda self_output: BertAdaptedSelfOutput(self_output, config=config)

def adapt_bert_output(config: AdapterConfig):
    return lambda self_output: BertAdaptedSelfOutput(self_output, config=config)

def adapt_transformer_output(config: AdapterConfig):
    return lambda dec_layer: TransformerAdaptedDecoderLayer(dec_layer, config=config)

def add_enc_adapters(bert_model: BertModel,
                 config: AdapterConfig) -> BertModel:

    # Replace specific layer with adapter-added layer
    bert_encoder = bert_model.encoder
    for i in range(len(bert_model.encoder.layer)):
        bert_encoder.layer[i].attention.output = adapt_bert_self_output(config)(
            bert_encoder.layer[i].attention.output)
        bert_encoder.layer[i].output = adapt_bert_output(config)(
            bert_encoder.layer[i].output)

    # Freeze all parameters
    for param in bert_model.parameters():
        param.requires_grad = False
    # Unfreeze trainable parts — layer norms and adapters
    for name, sub_module in bert_model.named_modules():
        if isinstance(sub_module, (Adapter_func, BertLayerNorm)):
            for param_name, param in sub_module.named_parameters():
                param.requires_grad = True
    return bert_model

def add_dec_adapters(dec_model:TransformerDecoder,
                     config: AdapterConfig) -> TransformerDecoder:

    # Replace specific layer with adapter-added layer
    for i in range(len(dec_model.transformer_layers)):
        dec_model.transformer_layers[i] = adapt_transformer_output(config)(
            dec_model.transformer_layers[i])

    # Freeze all parameters
    for param in dec_model.parameters():
        param.requires_grad = False

    # Unfreeze trainable parts — layer norms and adapters
    for name, sub_module in dec_model.named_modules():
        if isinstance(sub_module, (Adapter_func, nn.LayerNorm)):
            for param_name, param in sub_module.named_parameters():
                param.requires_grad = True
    return dec_model

def add_layer_norm(module, d_model, eps):

    # Replace all layer norm with special layer norm module
    def replace_ln(m):
        matches = ["LayerNorm", "layer_norm", "layer_norm_1", "layer_norm_2"]
        for attr_str in dir(m):
            if any(x in attr_str for x in matches):
                target_attr = getattr(m, attr_str)
                if type(target_attr) == nn.LayerNorm:
                    setattr(m, attr_str, LayerNorm_func(d_model, eps))
        for n, ch in m.named_children():
            replace_ln(ch)

    replace_ln(module)

    # Freeze all parameters
    for param in module.parameters():
        param.requires_grad = False

    # Unfreeze trainable parts — layer norms and adapters
    for name, sub_module in module.named_modules():
        if isinstance(sub_module, (Adapter_func, LayerNorm_func)):
            for param_name, param in sub_module.named_parameters():
                param.requires_grad = True
    return module



