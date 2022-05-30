REPO=$PWD
MODEL=${1:-google/mt5-base}
GPU=${2:-0}
TRAIN_LANG=${3:all}

DATA_DIR=${4:-"$REPO/download/"}
OUT_DIR=${5:-"$REPO/outputs/"}

TASK=xnli
LR=3e-4

if [ "${TRAIN_LANG}" == "all" ]; then
  CONFIG=multilingual/
  MAX_STEPS=500000
else
  CONFIG=in_language/${TRAIN_LANG}_
  MAX_STEPS=50000
fi

if [ $MODEL == "google/mt5-small" ] || [ $MODEL == "google/mt5-base" ] || [ $MODEL == "google/mt5-large" ]; then
  MAXL=1024
  MAX_ANSWER_LEN=8
  BATCH_SIZE=4
  GRAD_ACC=8
elif [ $MODEL == "google/byt5-small" ] || [ $MODEL == "google/byt5-base" ] || [ $MODEL == "google/byt5-large" ]; then
  MAXL=2048
  MAX_ANSWER_LEN=8
  BATCH_SIZE=2
  GRAD_ACC=16
fi


MODEL_PATH=$OUT_DIR/${TASK}/${CONFIG}${MODEL}_LR${LR}_${MAX_STEPS}steps_batchsize${BATCH_SIZE}_gradacc${GRAD_ACC}
mkdir -p $MODEL_PATH
mkdir -p $MODEL_PATH/cache

# train
CUDA_VISIBLE_DEVICES=$GPU python third_party/run_t5_classification.py \
  --model_name_or_path ${MODEL} \
  --do_train \
  --task $TASK \
  --train_lang ${TRAIN_LANG} \
  --dataset_name ${TASK} \
  --output_dir ${MODEL_PATH} \
  --overwrite_output_dir \
  --cache_dir $MODEL_PATH/cache \
  --max_seq_length ${MAXL} \
  --max_answer_length ${MAX_ANSWER_LEN} \
  --generation_max_length ${MAX_ANSWER_LEN} \
  --learning_rate ${LR} \
  --optim adafactor \
  --lr_scheduler_type "constant" \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRAD_ACC} \
  --report_to wandb \
  --save_strategy steps --save_steps 10000 \
  --evaluation_strategy steps \
  --max_steps $MAX_STEPS --logging_steps 500 \
  --load_best_model_at_end --metric_for_best_model eval_accuracy \
  --logging_first_step --sortish_sampler --group_by_length --predict_with_generate 
