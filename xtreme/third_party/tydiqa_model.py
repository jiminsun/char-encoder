# CanineForQuestionAnswering: https://huggingface.co/transformers/model_doc/canine.html#canineforquestionanswering
# Canine for TyDiQA: https://github.com/google-research/language/blob/master/language/canine/tydiqa/tydi_modeling.py

import xtreme.third_party.tydiqa.data as data
from transformers import CanineModel
from transformers import CaninePreTrainedModel
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TyDiQAModelOutput(ModelOutput):
    """
    Base class for outputs of question answering models.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    answer_type_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class CanineTyDiQAModel(CaninePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.canine = CanineModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.answer_type_output_weights = nn.Linear(config.hidden_size, len(data.AnswerType), bias=True)
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
