# TaCL
Model results on SQuAD 1.1

## Run this script
```bibtex

CUDA_VISIBLE_DEVICES=0 python run_qa.py \
  --model_name_or_path <your model path> \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir <your_output>
```

