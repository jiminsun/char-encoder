REPO=$PWD
MODEL=${1:-google/mt5-base}
GPU=${2:-0}
TRAIN_LANG=${3:all}

DATA_DIR=${3:-"$REPO/download/"}
OUT_DIR=${4:-"$REPO/outputs/"}

TASK=xnli
BATCH_SIZE=4
GRAD_ACC=8

LR=3e-4
NUM_EPOCHS=10
MAX_STEPS=100000

if [ "${TRAIN_LANG}" == "all" ]; then
  CONFIG=multilingual/
else
  CONFIG=in_language/${TRAIN_LANG}_
fi

if [ $MODEL == "google/mt5-small" ] || [ $MODEL == "google/mt5-base" ] || [ $MODEL == "google/mt5-large" ]; then
  MAXL=1024
  MAX_ANSWER_LEN=32
elif [ $MODEL == "google/byt5-small" ] || [ $MODEL == "google/byt5-base" ] || [ $MODEL == "google/byt5-large" ]; then
  MAXL=4096
  MAX_ANSWER_LEN=32
fi


MODEL_PATH=$OUT_DIR/${TASK}/${CONFIG}${MODEL}_LR${LR}_EPOCH${NUM_EPOCHS}_batchsize${BATCH_SIZE}_gradacc${GRAD_ACC}
mkdir -p $MODEL_PATH
mkdir -p $MODEL_PATH/cache

# train
CUDA_VISIBLE_DEVICES=$GPU python third_party/run_t5_classification.py \
  --model_name_or_path ${MODEL} \
  --do_train \
  --do_eval \
  --train_lang ${TRAIN_LANG} \
  --dataset_name ${TASK} \
  --max_seq_length ${MAXL} \
  --output_dir ${MODEL_PATH} \
  --overwrite_output_dir \
  --overwrite_cache \
  --learning_rate ${LR} \
  --lr_scheduler_type "linear" \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRAD_ACC} \
  --report_to wandb \
  --max_answer_length ${MAX_ANSWER_LEN} \
  --save_strategy steps --save_steps 10000 \
  --eval_steps 10 \
  --max_steps $MAX_STEPS --logging_steps 50