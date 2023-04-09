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
        self.temperature = temperature
        self.use_contrastive_loss = use_contrastive_loss
        if self.use_contrastive_loss:
            pass

    def save_model(self, spath):
        if os.path.exists(spath):
            pass
        else:
            os.mkdir(spath, exist_ok=True)
        self.bert.save_pretrained(spath)
        self.tokenizer.save_pretrained(spath)

    def teacher_rep(self, inputs, tokenizer_id, attention_mask):
        pass