#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional, Tuple, Union
import numpy as np

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers import GPT2Model, GPT2PreTrainedModel


class FinptGPT2ForSequenceClassification(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        r"h\.\d+\.attn\.bias",
        r"h\.\d+\.attn\.masked_bias",
    ]
    _keys_to_ignore_on_load_missing = [
        r"h\.\d+\.attn\.masked_bias",
        r"classifier.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.n_embd, self.num_labels, bias=False)

        # model parallel
        self.model_parallel = False
        self.device_map = None

        self.tokenizer = None
        self.neg_to_pos = float(1.0)
        self.use_pos_weight = False
        self.nan_batch_count = 0

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        if np.isnan(hidden_states.detach().cpu().numpy()).any():
            self.nan_batch_count += 1

        # use the hidden states of the last non-padding token
        batch_size = attention_mask.size(0)
        sequence_lengths = attention_mask.sum(-1)

        cls_hidden = []
        for b_idx in range(batch_size):
            cur_seq_len = sequence_lengths[b_idx]
            cls_hidden.append(
                hidden_states[b_idx : b_idx + 1, cur_seq_len - 1, :]
            )
        cls_hidden = torch.cat(cls_hidden, dim=0)

        cls_hidden = self.dropout(cls_hidden)
        pooled_logits = self.classifier(cls_hidden)

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
                    loss_fct(pooled_logits.squeeze(), labels.squeeze())
                    if self.num_labels == 1
                    else loss_fct(pooled_logits, labels)
                )

            elif self.config.problem_type == "single_label_classification":
                if self.use_pos_weight:
                    cur_dev = pooled_logits.device
                    pos_weight = torch.tensor(
                        [1.0, self.neg_to_pos],
                        dtype=torch.float32,
                        device=cur_dev,
                    )
                    loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight)
                    labels_bce = F.one_hot(
                        labels, num_classes=self.num_labels
                    ).float()
                    loss = loss_fct(
                        pooled_logits.float(), labels_bce
                    )
                    loss = loss.to(dtype=pooled_logits.dtype, device=cur_dev)
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(
                        pooled_logits.view(-1, self.num_labels),
                        labels.view(-1),
                    )

            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
