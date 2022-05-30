REPO=$PWD
MODEL=$1
LANG=${2:-en}
GPU=${3:-0}
SAMPLES=${4:-5010}

DATA_DIR=$REPO/download/
TASK=xnli
BATCH_SIZE=2

MAXL=4096
MAX_ANSWER_LEN=8

CUDA_VISIBLE_DEVICES=$GPU python third_party/run_t5_classification.py \
  --model_name_or_path ${MODEL} \
  --task ${TASK} \
  --output_dir ${MODEL} \
  --do_predict \
  --eval_lang ${LANG} \
  --dataset_name ${TASK} \
  --max_seq_length ${MAXL} \
  --output_dir ${MODEL} \
  --overwrite_cache \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --eval_accumulation_steps 4 \
  --predict_with_generate --generation_max_length $MAX_ANSWER_LEN \
  --generation_num_beams 5 --max_eval_samples 100 --max_predict_samples $SAMPLES