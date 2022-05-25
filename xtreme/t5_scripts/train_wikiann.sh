REPO=$PWD
MODEL=${1:-google/mt5-base}
GPU=${2:-0}
TRAIN_LANG=${3:all}

DATA_DIR=${3:-"$REPO/download/"}
OUT_DIR=${4:-"$REPO/outputs/"}

TASK=wikiann
BATCH_SIZE=2
GRAD_ACC=8

LR=1e-4
NUM_EPOCHS=10
MAXL=1024

if [ "${TRAIN_LANG}" == "all" ]; then
  CONFIG=multilingual/
else
  CONFIG=in_language/${TRAIN_LANG}_
fi

if [ $MODEL == "google/mt5-small" ] || [ $MODEL == "google/mt5-base" ] || [ $MODEL == "google/mt5-large" ]; then
  MAXL=1024
  MAX_ANSWER_LEN=512
elif [ $MODEL == "google/byt5-small" ] || [ $MODEL == "google/byt5-base" ] || [ $MODEL == "google/byt5-large" ]; then
  MAXL=4096
  MAX_ANSWER_LEN=512
fi


MODEL_PATH=$OUT_DIR/${TASK}/${CONFIG}${MODEL}_LR${LR}_EPOCH${NUM_EPOCHS}_batchsize${BATCH_SIZE}_gradacc${GRAD_ACC}
mkdir -p $MODEL_PATH
mkdir -p $MODEL_PATH/cache

# train
CUDA_VISIBLE_DEVICES=$GPU python third_party/run_t5_ner.py \
  --model_name_or_path ${MODEL} \
  --do_train \
  --train_lang ${TRAIN_LANG} \
  --dataset_name ${TASK} \
  --max_seq_length ${MAXL} \
  --output_dir ${MODEL_PATH} \
  --overwrite_output_dir \
  --overwrite_cache \
  --learning_rate ${LR} \
  --lr_scheduler_type "constant" \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRAD_ACC} \
  --num_beams 30 --report_to wandb \
  --max_answer_length ${MAX_ANSWER_LEN} \
  --save_strategy steps --save_steps 5000 \
  --max_steps 30000
#  --num_train_epochs ${NUM_EPOCHS} \
# --evaluation_strategy steps --eval_steps 5000 \
# eval while training causes memory error for base models