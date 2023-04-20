CUDA_VISIBLE_DEVICES=0,1 python train.py\
    --language english\
    --model_name bert-base-uncased\
    --train_data ../pretrainingdata/english_wiki.txt\
    --gpus 2\
    --max_len 256\
    --bsz_per_gpu 64\
    --grad_acc_steps 2\
    --eff_bsz 256\
    --learning_rate 1e-4\
    --total_steps 150010\
    --prints 500\
    --saves 10000\
    --save_path ./model/tacl_english/