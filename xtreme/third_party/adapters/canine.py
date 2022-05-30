# CanineForQuestionAnswering: https://huggingface.co/transformers/model_doc/canine.html#canineforquestionanswering
# Canine for TyDiQA: https://github.com/google-research/language/blob/master/language/canine/tydiqa/tydi_modeling.py

import copy
from transformers import (
    CanineModel,
    CaninePreTrainedModel,
    CanineForQuestionAnswering,
    CanineForSequenceClassification,
    CanineForTokenClassification
)
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

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
)
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import apply_chunking_to_forward

from .tydiqa_utils import TyDiQAModelOutput


class Adapter(nn.Module):
    def __init__(
            self,
            hidden_size,
            bottleneck_size,
    ):
        super().__init__()
        self.down_projection = nn.Linear(hidden_size, bottleneck_size)
        self.activation = nn.ReLU(inplace=False)
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
        adapter,
        bottleneck_size,
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

        self.adapter_loc = adapter
        self.adapter = Adapter(
            hidden_size=config.hidden_size,
            bottleneck_size=bottleneck_size,
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
            attention_output = attention_output + adapter_output
        elif self.adapter_loc == 'parallel_attention':
            adapter_output = self.adapter(hidden_states)
            attention_output = attention_output + adapter_output

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        if self.adapter_loc == 'feedforward':
            # sequential adapter
            adapter_output = self.adapter(layer_output)
            layer_output = layer_output + adapter_output
        elif self.adapter_loc == 'parallel_feedforward':
            adapter_output = self.adapter(attention_output)
            layer_output = layer_output + adapter_output

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
        adapter='attention',
        bottleneck_size=512,
    ):
        super().__init__(config)
        self.config = config
        self.layer = nn.ModuleList(
            [
                CanineLayerWithAdapter(
                    config,
                    local,
                    always_attend_to_first_position,
                    first_position_attends_to_all,
                    attend_from_chunk_width,
                    attend_from_chunk_stride,
                    attend_to_chunk_width,
                    attend_to_chunk_stride,
                    adapter,
                    bottleneck_size,
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

    
class CanineAdapterModel(CanineModel):
    def __init__(self, config, adapter, bottleneck_size, adapter_location='initial', add_pooling_layer=True):
        super().__init__(config)
        shallow_config = copy.deepcopy(config)
        shallow_config.num_hidden_layers = 1
        # TODO: add embedding adapters
        self.char_embeddings = CanineEmbeddings(config)

        assert adapter_location in ['initial', 'final']

        # shallow/low-dim transformer encoder to get a initial character encoding
        if adapter_location == 'initial':
            self.initial_char_encoder = CanineEncoderWithAdapter(
                shallow_config,
                local=True,
                always_attend_to_first_position=False,
                first_position_attends_to_all=False,
                attend_from_chunk_width=config.local_transformer_stride,
                attend_from_chunk_stride=config.local_transformer_stride,
                attend_to_chunk_width=config.local_transformer_stride,
                attend_to_chunk_stride=config.local_transformer_stride,
                adapter=adapter,
                bottleneck_size=bottleneck_size,
            )
        else:
            self.initial_char_encoder = CanineEncoder(
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
        if adapter_location == 'final':
            self.final_char_encoder = CanineEncoderWithAdapter(
                shallow_config,
                adapter=adapter,
                bottleneck_size=bottleneck_size
            )
        else:
            self.final_char_encoder = CanineEncoder(shallow_config)

        self.pooler = CaninePooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()


class CanineAdapterForQuestionAnswering(CanineForQuestionAnswering):
    def __init__(self, config, adapter, bottleneck_size):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.canine = CanineAdapterModel(config, adapter, bottleneck_size)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()


class CanineAdapterForSequenceClassification(CanineForSequenceClassification):
    def __init__(self, config, adapter, bottleneck_size):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.canine = CanineAdapterModel(config, adapter, bottleneck_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()


class CanineAdapterForTokenClassification(CanineForTokenClassification):
    def __init__(self, config, adapter, bottleneck_size):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.canine = CanineAdapterModel(config, adapter, bottleneck_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()


class CanineAdapterTyDiQAModel(CaninePreTrainedModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.canine = CanineAdapterModel(
            config,
            kwargs['adapter'],
            kwargs['bottleneck_size']
        )
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.answer_type_output_weights = nn.Linear(config.hidden_size, 5, bias=True)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            answer_types=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
                start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                    Labels for position (index) of the start of the labelled span for computing the token classification loss.
                    Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
                    sequence are not taken into account for computing the loss.
                end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                    Labels for position (index) of the end of the labelled span for computing the token classification loss.
                    Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
                    sequence are not taken into account for computing the loss.
                """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.canine(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # [last_hidden_state, pooler_output, hidden_states, attentions]

        sequence_output = outputs[0]    # last hidden state     # [B x L x H]

        logits = self.qa_outputs(sequence_output)               # [B x L x 2]
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)                 # [B x L]
        end_logits = end_logits.squeeze(-1)                     # [B x L]

        pooled_output = outputs[1]      # hidden state of first token ([CLS])   # [B x H]
        answer_type_logits = self.answer_type_output_weights(pooled_output)     # [B x Q]

        total_loss = None
        # compute loss for positions
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            # compute loss for labels
            if answer_types is not None:
                loss_fct = CrossEntropyLoss()
                answer_type_loss = loss_fct(answer_type_logits, answer_types)
                total_loss = (start_loss + end_loss + answer_type_loss) / 3.0

        if not return_dict:
            output = (start_logits, end_logits, answer_type_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return TyDiQAModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            answer_type_logits=answer_type_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
