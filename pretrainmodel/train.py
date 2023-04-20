import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import argparse, os
import random
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer
from dataclass import PretrainCorpus
from bertpre import BERTContrastivePretraining

import time
import logging

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="bert-base-uncased")
    parser.add_argument("--sim", type=str, default="cosine")
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--use_contrastive_loss", type=bool, default=True)

    parser.add_argument("--whole_word_masking", type=bool, default=False)
    parser.add_argument("--train_data", type=str, help="path to the pre-training data")
    parser.add_argument("--max_len", type=int)

    parser.add_argument("--gpus", type=int)
    parser.add_argument("--bsz_per_gpu", type=int)
    parser.add_argument("--grad_acc_steps", type=int)
    parser.add_argument("--eff_bsz", type=int)
    parser.add_argument("--total_steps", type=int)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument('--prints', type=int)
    parser.add_argument('--saves', type=int)
    parser.add_argument("--save_path", type=str)
    return parser.parse_args()

def lr_update(optimizer, lr):
    for param in optimizer.param_groups:
        param['lr'] = lr

def train_model(args, model, save_prefix_path, gradient_accumulation_steps, tokenizer,
        dataset_batch_size, max_len, whole_word_masking, total_steps):
    is_cuda = torch.cuda.is_available()
    multi_gpu = False
    if is_cuda:
        if torch.cuda.device_count() > 2:
            multi_gpu = True
    device = torch.device('cuda')

    warmup_steps = int(0.1 * total_steps)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,num_training_steps=total_steps)
    optimizer.zero_grad()

    contrastive_acc_acm, ntokens_acm, acc_nxt_acm, npairs_acm = 0., 0., 0., 0.
    mlm_acm = 0.0
    train_loss = 0.

    effective_batch_acm = 0
    all_batch_step = 1
    print_valid, save_valid = False, False
    data = PretrainCorpus(tokenizer, filename=args.train_data, max_len=max_len, whole_word_mask=whole_word_masking)

    while effective_batch_acm < total_steps:
        truth, inp, seg, msk, attn_msk, labels, contrastive_labels, nxt_snt_flag = data.get_batch_data(dataset_batch_size) #batchify
        all_batch_step += 1
        if effective_batch_acm <= warmup_steps:
            lr_update(optimizer, args.learning_rate*effective_batch_acm/warmup_steps)
        if is_cuda:
            truth = truth.cuda(device)
            inp = inp.cuda(device)
            seg = seg.cuda(device)
            msk = msk.cuda(device)
            attn_msk = attn_msk.cuda(device)
            labels = labels.cuda(device)
            contrastive_labels = contrastive_labels.cuda(device)
            nxt_snt_flag = nxt_snt_flag.cuda(device)

        bsz = truth.size()[0]
        loss, mlm_correct_num, tot_tokens, nxt_snt_correct_num, correct_contrastive_num, total_contrastive_num = model(truth, inp, seg, msk, attn_msk, labels, contrastive_labels, nxt_snt_flag)

        mlm_correct_num = torch.sum(mlm_correct_num).item()
        tot_tokens = torch.sum(tot_tokens).item()
        nxt_snt_correct_num = torch.sum(nxt_snt_correct_num).item()
        correct_contrastive_num = torch.sum(correct_contrastive_num).item()

        ntokens_acm += tot_tokens
        mlm_acm += mlm_correct_num
        contrastive_acc_acm += correct_contrastive_num
        acc_nxt_acm += nxt_snt_correct_num
        npairs_acm += bsz

        loss = loss.mean()
        loss.backward()
        train_loss += loss.item()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        if all_batch_step%gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            effective_batch_acm += 1
            prints, saves = True, True

        if effective_batch_acm % args.print_every == 0 and prints:
            one_train_loss = train_loss / (effective_batch_acm * gradient_accumulation_steps)
            middle_acc, contrastive_acc = mlm_acm / ntokens_acm, contrastive_acc_acm / ntokens_acm
            nxt_snt_acc = acc_nxt_acm / npairs_acm

            middle_acc = round(middle_acc * 100, 3)
            contrastive_acc = round(contrastive_acc * 100, 3)
            nxt_snt_acc = round(nxt_snt_acc * 100, 3)

            print('At training steps {}, training loss is {}, middle mlm acc is {}, contrastive acc is {}, \
                        next sentence acc is {}'.format(effective_batch_acm, one_train_loss, middle_acc,
                                                        contrastive_acc, nxt_snt_acc))
            print_valid = False

        if effective_batch_acm % args.save_every == 0 and saves:
            one_train_loss = train_loss / (args.save_every * gradient_accumulation_steps)
            middle_acc, contrastive_acc = mlm_acm / ntokens_acm, contrastive_acc_acm / ntokens_acm
            nxt_snt_acc = acc_nxt_acm / npairs_acm

            middle_acc = round(middle_acc * 100, 3)
            contrastive_acc = round(contrastive_acc * 100, 3)
            nxt_snt_acc = round(nxt_snt_acc * 100, 3)

            print('At training steps {}, training loss is {}, middle mlm acc is {}, contrastive acc is {}, \
                        next sentence acc is {}'.format(effective_batch_acm, one_train_loss, middle_acc,
                                                        contrastive_acc, nxt_snt_acc))
            print('saving step')
            save_name = 'training_step_{}_middle_mlm_acc_{}_contrastive_acc_{}_nxt_sen_acc_{}'.format(
                effective_batch_acm,
                middle_acc, contrastive_acc, nxt_snt_acc)
            saves = False
            model_path = save_prefix_path + '/' + save_name
            if os.path.exists(model_path):
                pass
            else:
                os.makedirs(model_path, exist_ok=True)

            if is_cuda and torch.cuda.device_count() > 1:
                model.module.save_model(model_path)
            else:
                model.save_model(model_path)

            print('Model Saved')
            train_loss = 0.
            contrastive_acc_acm, ntokens_acm, acc_nxt_acm, npairs_acm = 0., 0., 0., 0.
            mlm_acm = 0.0

        print('Training done')
        return model

if __name__ == '__main__':
    is_cuda = torch.cuda.is_available()
    multi_gpu = False
    if is_cuda:
        if torch.cuda.device_count() > 2:
            multi_gpu = True

    args = parse_config()
    device = torch.device('cuda')
    model_name = args.model_name
    model = BERTContrastivePretraining(model_name, args.sim, args.temperature, args.use_contrastive_loss)

    if is_cuda:
        if multi_gpu:
            model = nn.DataParallel(model)
        else:
            model = model.to(device)

    whole_word_masking = False

    bsz_per_gpu, grad_acc_steps, gpus, eff_bsz = args.bsz_per_gpu, args.grad_acc_steps, args.gpus, args.eff_bsz
    assert eff_bsz == bsz_per_gpu * gpus * grad_acc_steps
    max_len = args.max_len

    tokenizer = BertTokenizer.from_pretrained(model_name)
    save_prefix_path = args.save_path + '/'
    model = train_model(args=args, model=model, save_prefix_path=save_prefix_path, gradient_accumulation_steps=grad_acc_steps, \
                        tokenizer=tokenizer, dataset_batch_size=bsz_per_gpu*gpus, max_len=max_len, whole_word_masking=whole_word_masking,\
                        total_steps=args.total_steps)
    print('Training done!')






