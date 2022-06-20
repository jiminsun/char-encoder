REPO=$PWD
MODEL=${1:-google/mt5-base}
GPU=${2:-0}
TRAIN_LANG=${3:-all}

DATA_DIR=${4:-"$REPO/download/"}
OUT_DIR=${5:-"$REPO/outputs/"}

TASK=xnli

if [ "${TRAIN_LANG}" == "all" ]; then
  CONFIG=multilingual/
  MAX_STEPS=100000
  SAVE=500
  EVAL_LANG=all
else
  CONFIG=in_language/${TRAIN_LANG}_
  MAX_STEPS=5000
  SAVE=200
  EVAL_LANG=${TRAIN_LANG}
fi

if [ $MODEL == "google/mt5-small" ]; then
  MAXL=512
  MAX_ANSWER_LEN=4
  BATCH_SIZE=16
  GRAD_ACC=4
  LR=1e-4
elif [ $MODEL == "google/mt5-base" ]; then
  MODEL_TYPE="mt5"
  MAXL=512
  MAX_ANSWER_LEN=4
  BATCH_SIZE=8
  GRAD_ACC=8
  LR=1e-4
elif [ $MODEL == "google/byt5-small" ] || [ $MODEL == "google/byt5-base" ] || [ $MODEL == "google/byt5-large" ]; then
  MAXL=512
  MAX_ANSWER_LEN=8
  BATCH_SIZE=16
  GRAD_ACC=8
  LR=1e-4
fi


CACHE_DIR=$OUT_DIR/${TASK}/${CONFIG}${MODEL}_LR${LR}_${MAX_STEPS}steps_batchsize4_gradacc8/cache
mkdir -p $CACHE_DIR
MODEL_DIR=$OUT_DIR/${TASK}/${CONFIG}${MODEL}_LR${LR}_${MAX_STEPS}steps_batchsize${BATCH_SIZE}_gradacc${GRAD_ACC}_maxlen${MAXL}

#if [ $LANG == "en" ]; then
#    CACHE_DIR=/data/private/char-encoder/xtreme/outputs/xnli/in_language/en_google/mt5-small_LR3e-4_50000steps_batchsize4_gradacc8/cache
#elif [ $LANG == "hi" ]; then
#    CACHE_DIR=/data/private/cache/818164464f9c9fd15776ca8a00423b074344c3e929d00a2c1a84aa5a50c928bd
#    mkdir -p $CACHE_DIR
#fi


# train
CUDA_VISIBLE_DEVICES=$GPU python third_party/run_t5_classification.py \
  --model_name_or_path ${MODEL} \
  --do_train \
  --task $TASK \
  --train_lang ${TRAIN_LANG} \
  --dataset_name ${TASK} \
  --output_dir ${MODEL_DIR} \
  --cache_dir ${CACHE_DIR} \
  --report_to wandb \
  --overwrite_cache \
  --overwrite_output_dir \
  --max_seq_length ${MAXL} \
  --max_answer_length ${MAX_ANSWER_LEN} \
  --generation_max_length ${MAX_ANSWER_LEN} \
  --learning_rate ${LR} \
  --optim adafactor \
  --lr_scheduler_type "constant" \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRAD_ACC} \
  --save_strategy steps --save_steps $SAVE \
  --evaluation_strategy steps \
  --eval_steps $MAX_STEPS \
  --max_steps $MAX_STEPS --logging_steps 20 \
  --predict_with_generate 

bash $REPO/t5_scripts/test_xnli.sh ${MODEL_DIR}/checkpoint-5000 ${EVAL_LANG} ${GPU}