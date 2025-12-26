#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel


class FinptBertForSequenceClassification(BertPreTrainedModel):
    _CHECKPOINT_FOR_DOC = "bert-base-uncased"
    _CONFIG_FOR_DOC = "BertConfig"

    _CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "textattack/bert-base-uncased-yelp-polarity"
    _SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_1'"
    _SEQ_CLASS_EXPECTED_LOSS = 0.01

    _keys_to_ignore_on_load_missing = [r"classifier.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.tokenizer = None
        self.neg_to_pos = float(1.0)
        self.use_pos_weight = False
        self.nan_batch_count = 0

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
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

        cls_hidden = outputs[1]
        cls_hidden = self.dropout(cls_hidden)
        logits = self.classifier(cls_hidden)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and labels.dtype in (torch.long, torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = (
                    loss_fct(logits.squeeze(), labels.squeeze())
                    if self.num_labels == 1
                    else loss_fct(logits, labels)
                )

            elif self.config.problem_type == "single_label_classification":
                if self.use_pos_weight:
                    cur_dev = logits.device
                    pos_weight = torch.tensor(
                        [1.0, self.neg_to_pos],
                        dtype=torch.float32,
                        device=cur_dev,
                    )
                    loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight)
                    labels_bce = F.one_hot(
                        labels, num_classes=self.num_labels
                    ).to(dtype=torch.float32, device=cur_dev)
                    logits_bce = logits.to(dtype=torch.float32, device=cur_dev)
                    loss = loss_fct(logits_bce, labels_bce)
                    loss = loss.to(dtype=logits.dtype, device=cur_dev)
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(
                        logits.view(-1, self.num_labels),
                        labels.view(-1),
                    )

            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
