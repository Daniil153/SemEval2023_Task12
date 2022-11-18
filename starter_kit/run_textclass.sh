LANG=$1
SUBTASK=$2
CUDA_VISIBLE_DEVICES=0 python starter_kit/run_textclass.py \
  --model_name_or_path Davlan/afro-xlmr-mini \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 1.0 \
  --max_seq_length 128 \
  --data_dir SUBTASK/train/formatted-train-data/$LANG \
  --output_dir models/$LANG \
  --save_steps -1
