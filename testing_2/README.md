---
tags:
- generated_from_trainer
datasets:
- squad_v2
model-index:
- name: testing_2
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# testing_2

This model is a fine-tuned version of [/common/home/sb2311/PycharmProjects/TaCL/TaCL/pretrainmodel/model/tacl_english_0/training_step_140000_middle_mlm_acc_63.886_contrastive_acc_669.055_nxt_sen_acc_99.751](https://huggingface.co//common/home/sb2311/PycharmProjects/TaCL/TaCL/pretrainmodel/model/tacl_english_0/training_step_140000_middle_mlm_acc_63.886_contrastive_acc_669.055_nxt_sen_acc_99.751) on the squad_v2 dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 3e-05
- train_batch_size: 12
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 2.0

### Training results



### Framework versions

- Transformers 4.29.0.dev0
- Pytorch 2.0.0+cu117
- Datasets 2.11.0
- Tokenizers 0.13.3
