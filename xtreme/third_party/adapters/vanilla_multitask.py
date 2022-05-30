import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import (
    CanineModel,
    CaninePreTrainedModel,
    CanineForQuestionAnswering,
    CanineForSequenceClassification,
    CanineForTokenClassification
)

from transformers.modeling_outputs import QuestionAnsweringModelOutput

from .multilingual import CanineMultilingualAdapterForQuestionAnswering, CanineMultilingualAdapterModel
from .canine import CanineAdapterForQuestionAnswering, CanineAdapterModel
from .tydiqa_utils import TyDiQAModelOutput, QAWithSubwordPredictionOutput


class CanineForQAWithSubwordPrediction(CanineForQuestionAnswering):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.canine = CanineModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # layers for subword prediction as a token classification task
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.subword_classifier = nn.Linear(config.hidden_size, kwargs['vocab_size'])
        self.num_subword_labels = kwargs['vocab_size']

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
            subword_labels=None,
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

        # Subword prediction
        subword_loss = None
        sequence_output = self.dropout(sequence_output)
        subword_logits = self.subword_classifier(sequence_output)
        if subword_labels is not None:
            subword_loss_fct = CrossEntropyLoss(ignore_index=0)
            subword_loss = subword_loss_fct(
                subword_logits.view(-1, self.num_subword_labels),
                subword_labels.view(-1)
            )
        if not return_dict:
            output = (start_logits, end_logits, subword_logits) + outputs[2:]
            return ((total_loss, subword_loss) + output) if total_loss is not None else output

        return QAWithSubwordPredictionOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            subword_loss=subword_loss,
            subword_logits=subword_logits,
        )


class CanineAdapterForQAWithSubwordPrediction(CanineForQAWithSubwordPrediction):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, **kwargs)
        self.num_labels = config.num_labels

        self.canine = CanineAdapterModel(config, kwargs['adapter'], kwargs['bottleneck_size'])
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # layers for subword prediction as a token classification task
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.subword_classifier = nn.Linear(config.hidden_size, kwargs['vocab_size'])

        self.init_weights()


class CanineMultilingualAdapterForQAWithSubwordPrediction(CanineAdapterForQAWithSubwordPrediction):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, **kwargs)
        self.num_labels = config.num_labels
        self.canine = CanineMultilingualAdapterModel(
            config,
            kwargs['adapter'],
            kwargs['bottleneck_size'],
            kwargs['lang2id']
        )
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # layers for subword prediction as a token classification task
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.subword_classifier = nn.Linear(config.hidden_size, kwargs['vocab_size'])

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
            subword_labels=None,
    ):
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

        # Subword prediction
        subword_loss = None
        sequence_output = self.dropout(sequence_output)
        subword_logits = self.subword_classifier(sequence_output)
        if subword_labels is not None:
            subword_loss_fct = CrossEntropyLoss(ignore_index=0)
            subword_loss = subword_loss_fct(
                subword_logits.view(-1, self.num_subword_labels),
                subword_labels.view(-1)
            )

        if not return_dict:
            output = (start_logits, end_logits, subword_logits) + outputs[2:]
            return ((total_loss, subword_loss) + output) if total_loss is not None else output

        return QAWithSubwordPredictionOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            subword_loss=subword_loss,
            subword_logits=subword_logits,
        )