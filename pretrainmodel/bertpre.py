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

# understand this one
    def compute_loss(self, truth, mask, logits):
        truth = truth.transpose(0,1)
        mask = mask.transpose(0,1)
        y_mlm = logits.transpose(0, 1).masked_select(mask.unsqueeze(-1).to(torch.bool))
        y_mlm = y_mlm.view(-1, self.vocab_size)
        gold = truth.masked_select(mask.to(torch.bool))
        log_probs_mlm = torch.log_softmax(y_mlm, -1)
        mlm_loss = F.nll_loss(log_probs_mlm, gold, reduction='mean')
        _, pred_mlm = log_probs_mlm.max(-1)
        mlm_correct_num = torch.eq(pred_mlm, gold).float().sum().item()
        return mlm_loss, mlm_correct_num

    def forward(self, truth, input, seg, mask, attn_mask, labels, contrastive_labels, nxt_snt_flag):
        if torch.cuda.is_available():
            cuda = True
        else:
            cuda = False
        bsz, seqlen = truth.size()
        masked_rep, logits, sen_relation_scores = self.compute_rep(inputs=input, tokenizer_id=seg, attention_mask=attn_mask)
        mlm_loss, mlm_correct_num = self.compute_loss(truth, mask, logits)
        if self.use_contrastive_loss:
            truth_rep, truth_logits, _ = self.compute_teacher_rep(inputs=truth, tokenizer_id=seg, attention_mask=attn_mask)
            ''' 
                            mask_rep, truth_rep : hidden_size
                            rep, left_rep, right_rep: bsz x seqlen x embed_dim
            '''
            masked_rep = masked_rep / masked_rep.norm(dim=2, keepdim=True)
            truth_rep = truth_rep / truth_rep.norm(dim=2, keepdim=True)
            contrastive_scores = torch.matmul(masked_rep, truth_rep.T) / self.temperature
            assert contrastive_scores.size() == torch.Size([bsz, seqlen, seqlen]) # lets see



