import sys
import os
import operator
from operator import itemgetter
import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import argparse

from transformers import BertForPreTraining, BertTokenizer, BertConfig, BertModel
from torch.nn import CrossEntropyLoss

class BERTContrastivePretraining(nn.Module):
    def __int__(self, model_name, sim='cosine', temperature=0.02, use_contrastive_loss=False):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForPreTraining.from_pretrained(model_name)
        self.bert = self.model.bert
        self.vocab_size = self.tokenizer.vocab_size
        self.config = BertConfig.from_pretrained(model_name)
        self.embed_dim = self.config.hidden_size
        self.loss_fact = CrossEntropyLoss()
        self.sim = sim
        # change sim to dot product for improvements
        self.temperature = temperature
        self.use_contrastive_loss = use_contrastive_loss
        if self.use_contrastive_loss:
            self.teacher = BertModel.from_pretrained(model_name)
            for param in self.teacher.parameters():
                param.requires_grad = False


    def save_model(self, spath):
        if os.path.exists(spath):
            pass
        else:
            os.mkdir(spath, exist_ok=True)
        self.bert.save_pretrained(spath)
        self.tokenizer.save_pretrained(spath)

    def compute_teacher_rep(self, inputs, tokenizer_id, attention_mask):
        batch_size, seq_len = inputs.size()
        outputs = self.teacher(input_ids=inputs, token_type_ids=tokenizer_id, attention_mask=attention_mask)
        reprens, pool_output = outputs[0], outputs[1]
        reprens = reprens.view(batch_size, seq_len, self.embed_dim)
        logits, sen_rel_scores = self.cls(reprens, pool_output)
        return reprens, logits, sen_rel_scores

    def compute_rep(self, inputs, tokenizer_id, attention_mask):
        batch_size, seq_len = inputs.size()
        outputs = self.bert(input_ids=inputs, token_type_ids=tokenizer_id, attention_mask=attention_mask)
        reprens, pool_output = outputs[0], outputs[1]
        reprens = reprens.view(batch_size, seq_len, self.embed_dim)
        logits, sen_rel_scores = self.cls(reprens, pool_output)
        return reprens, logits, sen_rel_scores


