# TaCL
Model results on SQuAD 2.0

## Run this script
```bibtex
CUDA_VISIBLE_DEVICES=0 python run_qa.py \
  --model_name_or_path <model_dir> \
  --dataset_name squad_v2 \
  --do_train \
  --do_eval \
    --version_2_with_negative \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir <output_dir>
```

