import random
import torch
import numpy as np
import re
from google_bert import create_instance_from_document

UNK, CLS, SEP, PAD, MASK = "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"

class PretrainCorpus(object):
    def __init__(self, tokenizer, filename, max_len, whole_word_mask=False):
        '''
                    tokenizer: BERT tokenizer
                    filename: pretraining corpus
                    max_len: maximum length for each sentence
        '''
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.special_id_token_list = self.tokenizer.convert_tokens_to_ids([UNK, CLS, SEP, PAD, MASK])
        self.unk_id, self.sep_id, self.pad_id, self.cls_id, self.mask_id = self.special_id_token_list
        self.max_len = max_len
        self.filename = filename
        self.stream = open(self.filename, encoding='utf-8')
        self.whole_word_masking = whole_word_mask
        self.epoch_id = 0
        self.load_line(self.filename)

    def load_line(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        docs = [[]]
        for line in lines:
            token = line.strip('\n').strip().split()
            if token:
                docs[-1].append(token)
            else:
                docs.append([])
        docs = [x for x in docs if x]
        random.shuffle(docs)
        data = []
        for idx, doc in enumerate(docs):
            data.extend(create_instance_from_document(doc,idx,self.max_len))
        self.data = data
        print('sentence pairs - {}'.format(len(self.data)))
        self.train_id_list = [i for i in range(len(self.data))]

    def get_batch_data(self, batch_size):
        batch_idx_list = random.sample(self.train_id_list, batch_size)
        batch_data = []
        for idx in batch_idx_list:
            batch_data.append(self.data[idx])

        return self.batchify(batch_data)

    def random_token(self):
        rand_idx = 1 + np.random.randint(self.vocab_size-1)
        while rand_idx in self.special_token_id_list:
            rand_idx = 1 + np.random.randint(self.vocab_size-1)
        random_token = self.tokenizer.convert_ids_to_tokens([rand_idx])[0]
        return random_token

    # Review
    def random_mask(self, tokens, mask_lm_probs, max_preds_per_seq):
        num_to_predict = min(max_preds_per_seq, max(1, int(round(len(tokens) * mask_lm_probs))))
        masked_tokens, mask = [], []
        cand = []
        for i, token in enumerate(tokens):
            if token in [UNK, SEP, PAD, CLS, MASK]:  # do not learn to predict special tokens
                continue
            cand.append(i)
        random.shuffle(cand)
        cand = set(cand[:num_to_predict])

        masked_tokens, mask, tgt = [], [], []
        for i, token in enumerate(tokens):
            if i in cand:
                if random.random() < 0.8:
                    masked_tokens.append(MASK)
                else:
                    if random.random() < 0.5:
                        masked_tokens.append(token)
                    else:
                        masked_tokens.append(self.random_token())
                mask.append(1)
                tgt.append(token)
            else:
                masked_tokens.append(token)
                mask.append(0)
                tgt.append(PAD)
        return masked_tokens, mask, tgt

    def whole_word_random_mask(self, tokens, masked_lm_prob):
       # remove partial words
        joined_word_list = []
        for token in tokens:
            if token.startswith('##'):
                assert len(joined_word_list) != 0
                joined_word_list[-1].append(token)
            else:
                joined_word_list.append([token])

        cand_idx_list = []
        for idx in range(len(joined_word_list)):
            item = joined_word_list[idx]
            if len(item) == 1 and item[0] in [UNK, SEP, PAD, CLS, MASK]:  # we do not mask special tokens
                pass
            else:
                cand_idx_list.append(idx)

        num_words = len(cand_idx_list)
        num_to_predict = min(int(num_words * 0.25), max(1, int(round(num_words * masked_lm_prob))))
        random.shuffle(cand_idx_list)
        cand = set(cand_idx_list[:num_to_predict])

        masked_tokens, mask, tgt = [], [], []
        for i, one_word_list in enumerate(joined_word_list):
            if i in cand:
                for token in one_word_list:
                    if random.random() < 0.8:
                        masked_tokens.append(MASK)
                    else:
                        if random.random() < 0.5:
                            masked_tokens.append(token)
                        else:
                            masked_tokens.append(self.random_token())
                    mask.append(1)
                    tgt.append(token)
            else:
                for token in one_word_list:
                    masked_tokens.append(token)
                    mask.append(0)
                    tgt.append(PAD)
        return masked_tokens, mask, tgt

    def ListsToTensor(self, xs, tokenize=False):
        max_len = max(len(x) for x in xs)
        ys = []
        for x in xs:
            if tokenize:
                y = self.tokenizer.convert_tokens_to_ids(x) + [self.pad_id] * (self.max_len - len(x))
            else:
                y = x + [0] * (self.max_len - len(x))
            ys.append(y)
        data = torch.LongTensor(ys).contiguous()  # batchsize x seqlen
        return data

    def process_tgt(self, tgt_matrix):
        max_len = max(len(x) for x in tgt_matrix)
        ys = []
        for x in tgt_matrix:
            y = self.tokenizer.convert_tokens_to_ids(x) + [self.pad_id] * (self.max_len - len(x))
            ys.append(y)
        ys = torch.LongTensor(ys).contiguous()  # batchsize x seqlen
        labels = ys.clone()
        labels[labels[:, :] == self.pad_id] = -100
        contrastive_labels = ys.clone()
        contrastive_labels[contrastive_labels[:, :] == self.pad_id] = 0
        contrastive_labels[contrastive_labels[:, :] != self.pad_id] = 1
        return labels, contrastive_labels.type(torch.FloatTensor)

    def batchify(self, data):
        truth, inp, seg, msk, tgt_matrix = [], [], [], [], []
        nxt_snt_flag = []
        for a, b, r in data:
            x = [CLS] + a + [SEP] + b + [SEP]
            truth.append(x)
            seg.append([0] * (len(a) + 2) + [1] * (len(b) + 1))
            if self.whole_word_masking:
                masked_x, mask, tgt = self.whole_word_random_mask(x, 0.15)
            else:
                masked_x, mask, tgt = self.random_mask(x, 0.15, self.max_len * 0.25)

            inp.append(masked_x)
            msk.append(mask)
            tgt_matrix.append(tgt)
            if r:  # r stands for is_random_text
                nxt_snt_flag.append(1)
            else:
                nxt_snt_flag.append(0)

        truth = self.ListsToTensor(truth, tokenize=True)
        inp = self.ListsToTensor(inp, tokenize=True)
        seg = self.ListsToTensor(seg, tokenize=False)
        msk = self.ListsToTensor(msk, tokenize=False).to(torch.uint8)  # batchsize x seqlen
        attn_msk = ~inp.eq(self.pad_id)
        nxt_snt_flag = torch.ByteTensor(nxt_snt_flag)
        labels, contrastive_labels = self.process_tgt(tgt_matrix)
        return truth, inp, seg, msk, attn_msk, labels, contrastive_labels, nxt_snt_flag

