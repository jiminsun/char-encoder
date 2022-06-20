REPO=$PWD
MODEL=${1:-google/mt5-base}
GPU=${2:-0}
TRAIN_LANG=${3:-all}
EVAL_LANG=${4:-en}
DATA_DIR=${5:-"$REPO/download/"}
OUT_DIR=${6:-"$REPO/outputs/"}

TASK=wikiann



NUM_EPOCHS=10
MAXL=512

if [ "${TRAIN_LANG}" == "all" ]; then
  CONFIG=multilingual/
  MAX_STEPS=200000
  LOG=50
else
  CONFIG=in_language/${TRAIN_LANG}_
  MAX_STEPS=5000
  LOG=20
fi

SAVE_STEPS=200

if [ $MODEL == "google/mt5-small" ] || [ $MODEL == "google/mt5-large" ]; then
  MAXL=512
  MAX_ANSWER_LEN=128
  BATCH_SIZE=8
  LR=1e-4
  if [ "${TRAIN_LANG}" == "all" ]; then
      GRAD_ACC=4
  else
      GRAD_ACC=2
  fi
elif [ $MODEL == "google/mt5-base" ]; then
  MAXL=512
  MAX_ANSWER_LEN=128
  BATCH_SIZE=4
  LR=1e-4
  if [ "${TRAIN_LANG}" == "all" ]; then
      GRAD_ACC=8
  else
      GRAD_ACC=4
  fi
elif [ $MODEL == "google/byt5-small" ] || [ $MODEL == "google/byt5-base" ] || [ $MODEL == "google/byt5-large" ]; then
  MAXL=512
  MAX_ANSWER_LEN=512
  BATCH_SIZE=8
  LR=1e-3
  if [ "${TRAIN_LANG}" == "all" ]; then
      GRAD_ACC=4
  else
      GRAD_ACC=2
  fi
fi




MODEL_PATH=$OUT_DIR/${TASK}/${CONFIG}${MODEL}_LR${LR}_${MAX_STEPS}steps_bsz${BATCH_SIZE}_gradacc${GRAD_ACC}_maxlen${MAXL}

# train
CUDA_VISIBLE_DEVICES=$GPU python third_party/run_t5_ner.py \
  --model_name_or_path ${MODEL} \
  --task $TASK \
  --do_train \
  --train_lang ${TRAIN_LANG} \
  --eval_lang ${EVAL_LANG} \
  --dataset_name ${TASK} \
  --max_seq_length ${MAXL} \
  --output_dir ${MODEL_PATH} \
  --overwrite_output_dir \
  --optim adafactor \
  --learning_rate ${LR} \
  --lr_scheduler_type "constant" \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRAD_ACC} \
  --num_beams 30 \
  --max_answer_length ${MAX_ANSWER_LEN} \
  --save_strategy steps --save_steps $SAVE_STEPS \
  --max_steps $MAX_STEPS --predict_with_generate \
  --per_device_eval_batch_size 8 \
  --logging_steps $LOG --ignore_pad_token_for_loss \
  --max_eval_samples 100 --eval_accumulation_steps 4 \
  --evaluation_strategy steps --eval_steps 10000 \
  --generation_max_length $MAX_ANSWER_LEN 
  # --max_eval_samples 100 --eval_accumulation_steps 4 --prediction_loss_only \
  # --evaluation_strategy steps --eval_steps 200
#  --num_train_epochs ${NUM_EPOCHS} \
# --max_eval_samples 100 --eval_accumulation_steps 4 \
# eval while training causes memory error for base models
#  --evaluation_strategy steps --metric_for_best_model eval_loss \
  # --max_eval_samples 100 --eval_steps 200 --prediction_loss_only \