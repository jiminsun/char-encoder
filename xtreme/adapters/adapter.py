import tydiqa.data as data
from transformers import CanineModel, CaninePreTrainedModel
import copy
import torch
import torch.nn as nn
from xtreme.third_party.tydiqa.model import TyDiQAModelOutput
from transformers.models.canine.modeling_canine import (
    CanineEmbeddings,
    CanineAttention,
    CanineEncoder,
    CanineLayer,
    CanineIntermediate,
    CanineOutput,
    CharactersToMolecules,
    ConvProjection,
    CaninePooler,
    _PRIMES
)
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import apply_chunking_to_forward


class Adapter(nn.Module):
    def __init__(
            self,
            hidden_size,
            bottleneck_size,
    ):
        super().__init__()
        self.down_projection = nn.Linear(hidden_size, bottleneck_size)
        self.activation = nn.ReLU()
        self.up_projection = nn.Linear(bottleneck_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input):
        h = self.down_projection(input)     # [B x L x H] -> [B x L x R]
        h = self.activation(h)
        h = self.up_projection(h)           # [B x L x R] -> [B x L x H]
        return h


class CanineLayerWithAdapter(nn.Module):
    """
    Additional parameters:
        config.adapter = 'attention' or 'feedforward' or None
        config.bottleneck_size = 32, 64, 256, 512
            additional params: 768 (hidden dim) * bottleneck_size * 2
    """
    def __init__(
        self,
        config,
        local,
        always_attend_to_first_position,
        first_position_attends_to_all,
        attend_from_chunk_width,
        attend_from_chunk_stride,
        attend_to_chunk_width,
        attend_to_chunk_stride,
    ):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = CanineAttention(
            config,
            local,
            always_attend_to_first_position,
            first_position_attends_to_all,
            attend_from_chunk_width,
            attend_from_chunk_stride,
            attend_to_chunk_width,
            attend_to_chunk_stride,
        )

        self.adapter_loc = config.adapter
        self.adapter = Adapter(
            hidden_size=config.hidden_size,
            bottleneck_size=config.bottleneck_size,
        )

        self.intermediate = CanineIntermediate(config)
        self.output = CanineOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        if self.adapter_loc == 'attention':
            # sequential adapter
            adapter_output = self.adapter(attention_output)
            attention_output += adapter_output
        elif self.adapter_loc == 'parallel_attention':
            adapter_output = self.adapter(hidden_states)
            attention_output += adapter_output

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        if self.adapter_loc == 'feedforward':
            # sequential adapter
            adapter_output = self.adapter(layer_output)
            layer_output += adapter_output
        elif self.adapter_loc == 'parallel_feedforward':
            adapter_output = self.adapter(attention_output)
            layer_output += adapter_output

        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class CanineEncoderWithAdapter(CanineEncoder):
    def __init__(
        self,
        config,
        local=False,
        always_attend_to_first_position=False,
        first_position_attends_to_all=False,
        attend_from_chunk_width=128,
        attend_from_chunk_stride=128,
        attend_to_chunk_width=128,
        attend_to_chunk_stride=128,
    ):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [
                CanineLayer(
                    config,
                    local,
                    always_attend_to_first_position,
                    first_position_attends_to_all,
                    attend_from_chunk_width,
                    attend_from_chunk_stride,
                    attend_to_chunk_width,
                    attend_to_chunk_stride,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class CanineAdapterModel(CaninePreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        shallow_config = copy.deepcopy(config)
        shallow_config.num_hidden_layers = 1
        # TODO: add embedding adapters
        self.char_embeddings = CanineEmbeddings(config)

        # shallow/low-dim transformer encoder to get a initial character encoding
        self.initial_char_encoder = CanineEncoderWithAdapter(
            shallow_config,
            local=True,
            always_attend_to_first_position=False,
            first_position_attends_to_all=False,
            attend_from_chunk_width=config.local_transformer_stride,
            attend_from_chunk_stride=config.local_transformer_stride,
            attend_to_chunk_width=config.local_transformer_stride,
            attend_to_chunk_stride=config.local_transformer_stride,
        )

        self.chars_to_molecules = CharactersToMolecules(config)
        # deep transformer encoder
        self.encoder = CanineEncoder(config)
        self.projection = ConvProjection(config)
        # shallow/low-dim transformer encoder to get a final character encoding
        self.final_char_encoder = CanineEncoder(shallow_config)

        self.pooler = CaninePooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()



