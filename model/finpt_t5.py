#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import Optional, Tuple, Union
import warnings

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from transformers.models.t5.modeling_t5 import T5Stack
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutputWithPast
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers import T5Config, T5PreTrainedModel


class FinptT5ForSequenceClassification(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = [
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
        "lm_head.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # classification head
        self.tokenizer = None
        self.neg_to_pos = float(1.0)
        self.use_pos_weight = False
        self.nan_batch_count = 0

        self.num_labels = 2
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.d_model, self.num_labels, bias=False)

        self.post_init()

        # model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        warnings.warn(
            "parallelize is deprecated and will be removed in future versions.",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        warnings.warn(
            "deparallelize is deprecated and will be removed in future versions.",
            FutureWarning,
        )
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], SequenceClassifierOutputWithPast]:

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        labels_lm = input_ids

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1]
                if len(encoder_outputs) > 1
                else None,
                attentions=encoder_outputs[2]
                if len(encoder_outputs) > 2
                else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = self._shift_right(labels_lm)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        batch_size = attention_mask.size(0)
        sequence_lengths = attention_mask.sum(-1) + 1

        cls_hidden = []
        for b_idx in range(batch_size):
            cur_seq_len = sequence_lengths[b_idx]
            cls_hidden.append(
                sequence_output[b_idx : b_idx + 1, cur_seq_len - 1, :]
            )
        cls_hidden = torch.cat(cls_hidden, dim=0)

        cls_hidden = self.dropout(cls_hidden)
        cls_logits = self.classifier(cls_hidden)

        loss = None
        if labels is not None:
            labels = labels.to(cls_logits.device)
            if self.use_pos_weight:
                pos_weight = torch.tensor(
                    [1.0, self.neg_to_pos],
                    dtype=torch.float32,
                    device=cls_logits.device,
                )
                loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight)
                labels_bce = F.one_hot(
                    labels, num_classes=self.num_labels
                ).float()
                loss = loss_fct(cls_logits.float(), labels_bce)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    cls_logits.view(-1, self.num_labels),
                    labels.view(-1),
                )

        if not return_dict:
            output = (cls_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=cls_logits,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.last_hidden_state,
            attentions=decoder_outputs.attentions,
        )
