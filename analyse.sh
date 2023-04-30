#!/bin/bash -l

#SBATCH --output=/common/home/sb2311/PycharmProjects/TaCL/TaCL/logfile

#SBATCH -o /common/home/sb2311/PycharmProjects/TaCL/TaCL/out1.txt


export PATH="$PATH:/koko/system/anaconda/bin"

cd /common/home/sb2311/PycharmProjects/TaCL/TaCL

source activate python39
CUDA_VISIBLE_DEVICES=0 python  layer-wise-sim.py\
    --model_name /common/home/sb2311/PycharmProjects/TaCL/TaCL/pretrainmodel/model/tacl_english_0/training_step_140000_middle_mlm_acc_63.886_contrastive_acc_669.055_nxt_sen_acc_99.751\
    --file_path wiki_50k.txt\
    --output_path tacl_result.json