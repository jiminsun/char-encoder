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
    CanineModelOutputWithPooling
)
from transformers.modeling_outputs import BaseModelOutput, QuestionAnsweringModelOutput
from transformers.modeling_utils import apply_chunking_to_forward

from .tydiqa_utils import TyDiQAModelOutput
from .adapter import Adapter
from .canine import CanineAdapterTyDiQAModel
from .task_languages import TYDIQA_GOLDP_ID2LANG


class CanineMultilingualAdapterTyDiQAModel(CanineAdapterTyDiQAModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.canine = CanineMultilingualAdapterModel(
            config,
            kwargs['adapter'],
            kwargs['bottleneck_size'],
            kwargs['lang2id'],
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
            language_id=None,
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
            language_id=language_id,
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


class CanineMultilingualAdapterForQuestionAnswering(CanineForQuestionAnswering):
    def __init__(self, config, adapter, bottleneck_size, adapter_location, lang2id):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.canine = CanineMultilingualAdapterModel(
            config=config,
            adapter=adapter,
            bottleneck_size=bottleneck_size,
            adapter_location=adapter_location,
            lang2id=lang2id
        )
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

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
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            language_id=None,
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
            language_id=language_id,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
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

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class CanineMultilingualAdapterModel(CanineModel):
    def __init__(self, config, adapter, bottleneck_size, adapter_location='initial',
                 lang2id=None, add_pooling_layer=True):
        super().__init__(config)
        shallow_config = copy.deepcopy(config)
        shallow_config.num_hidden_layers = 1
        # TODO: add embedding adapters
        self.char_embeddings = CanineEmbeddings(config)

        # shallow/low-dim transformer encoder to get a initial character encoding
        if adapter_location == 'initial':
            self.initial_char_encoder = CanineEncoderWithMultilingualAdapter(
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
                lang2id=lang2id,
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
            self.final_char_encoder = CanineEncoderWithMultilingualAdapter(
                shallow_config,
                adapter=adapter,
                bottleneck_size=bottleneck_size,
                lang2id=lang2id
            )
        else:
            self.final_char_encoder = CanineEncoder(shallow_config)

        self.pooler = CaninePooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        language_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        molecule_attention_mask = self._downsample_attention_mask(
            attention_mask, downsampling_rate=self.config.downsampling_rate
        )
        extended_molecule_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            molecule_attention_mask, (batch_size, molecule_attention_mask.shape[-1]), device
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # `input_char_embeddings`: shape (batch_size, char_seq, char_dim)
        input_char_embeddings = self.char_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        # Contextualize character embeddings using shallow Transformer.
        # We use a 3D attention mask for the local attention.
        # `input_char_encoding`: shape (batch_size, char_seq_len, char_dim)
        char_attention_mask = self._create_3d_attention_mask_from_input_mask(input_ids, attention_mask)
        init_chars_encoder_outputs = self.initial_char_encoder(
            input_char_embeddings,
            language_id=language_id,
            attention_mask=char_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,

        )
        input_char_encoding = init_chars_encoder_outputs.last_hidden_state

        # Downsample chars to molecules.
        # The following lines have dimensions: [batch, molecule_seq, molecule_dim].
        # In this transformation, we change the dimensionality from `char_dim` to
        # `molecule_dim`, but do *NOT* add a resnet connection. Instead, we rely on
        # the resnet connections (a) from the final char transformer stack back into
        # the original char transformer stack and (b) the resnet connections from
        # the final char transformer stack back into the deep BERT stack of
        # molecules.
        #
        # Empirically, it is critical to use a powerful enough transformation here:
        # mean pooling causes training to diverge with huge gradient norms in this
        # region of the model; using a convolution here resolves this issue. From
        # this, it seems that molecules and characters require a very different
        # feature space; intuitively, this makes sense.
        init_molecule_encoding = self.chars_to_molecules(input_char_encoding)

        # Deep BERT encoder
        # `molecule_sequence_output`: shape (batch_size, mol_seq_len, mol_dim)
        encoder_outputs = self.encoder(
            init_molecule_encoding,
            attention_mask=extended_molecule_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        molecule_sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(molecule_sequence_output) if self.pooler is not None else None

        # Upsample molecules back to characters.
        # `repeated_molecules`: shape (batch_size, char_seq_len, mol_hidden_size)
        repeated_molecules = self._repeat_molecules(molecule_sequence_output, char_seq_length=input_shape[-1])

        # Concatenate representations (contextualized char embeddings and repeated molecules):
        # `concat`: shape [batch_size, char_seq_len, molecule_hidden_size+char_hidden_final]
        concat = torch.cat([input_char_encoding, repeated_molecules], dim=-1)

        # Project representation dimension back to hidden_size
        # `sequence_output`: shape (batch_size, char_seq_len, hidden_size])
        sequence_output = self.projection(concat)

        # Apply final shallow Transformer
        # `sequence_output`: shape (batch_size, char_seq_len, hidden_size])
        final_chars_encoder_outputs = self.final_char_encoder(
            sequence_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = final_chars_encoder_outputs.last_hidden_state

        if output_hidden_states:
            deep_encoder_hidden_states = encoder_outputs.hidden_states if return_dict else encoder_outputs[1]
            all_hidden_states = (
                    all_hidden_states
                    + init_chars_encoder_outputs.hidden_states
                    + deep_encoder_hidden_states
                    + final_chars_encoder_outputs.hidden_states
            )

        if output_attentions:
            deep_encoder_self_attentions = encoder_outputs.attentions if return_dict else encoder_outputs[-1]
            all_self_attentions = (
                    all_self_attentions
                    + init_chars_encoder_outputs.attentions
                    + deep_encoder_self_attentions
                    + final_chars_encoder_outputs.attentions
            )

        if not return_dict:
            output = (sequence_output, pooled_output)
            output += tuple(v for v in [all_hidden_states, all_self_attentions] if v is not None)
            return output

        return CanineModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class CanineEncoderWithMultilingualAdapter(CanineEncoder):
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
        lang2id=None,
    ):
        super().__init__(config)
        self.config = config
        self.layer = nn.ModuleList(
            [
                CanineLayerWithMultilingualAdapter(
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
                    lang2id,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states,
            language_id=None,
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

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    language_id,
                    attention_mask,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states=hidden_states,
                    language_id=language_id,
                    attention_mask=attention_mask,
                    head_mask=layer_head_mask,
                    output_attentions=output_attentions
                )

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


class CanineLayerWithMultilingualAdapter(nn.Module):
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
        lang2id,
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
        self.adapter_modules = nn.ModuleDict({
            lang: Adapter(
            hidden_size=config.hidden_size,
            bottleneck_size=bottleneck_size,
            ) for lang, idx in lang2id.items()
        })

        self.intermediate = CanineIntermediate(config)
        self.output = CanineOutput(config)

    def forward(
        self,
        hidden_states,
        language_id,
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
        language_names = [TYDIQA_GOLDP_ID2LANG[lid.item()] for lid in language_id]
        if self.adapter_loc == 'attention':
            # sequential adapter
            adapter_output = torch.stack([self.adapter_modules[lang](attention_output[idx, :])
                                          for idx, lang in enumerate(language_names)], dim=0)
            attention_output = attention_output + adapter_output
        elif self.adapter_loc == 'parallel_attention':
            adapter_output = torch.stack([self.adapter_modules[lang](hidden_states[idx, :])
                                          for idx, lang in enumerate(language_names)], dim=0)
            attention_output = attention_output + adapter_output

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        if self.adapter_loc == 'feedforward':
            # sequential adapter
            adapter_output = torch.stack([self.adapter_modules[lang](layer_output[idx, :])
                                          for idx, lang in enumerate(language_names)], dim=0)
            layer_output = layer_output + adapter_output
        elif self.adapter_loc == 'parallel_feedforward':
            adapter_output = torch.stack([self.adapter_modules[lang](attention_output[idx, :])
                                          for idx, lang in enumerate(language_names)], dim=0)
            layer_output = layer_output + adapter_output

        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output