REPO=$PWD
MODEL=${1:-google/mt5-base}
GPU=${2:-0}
TRAIN_LANG=${3:-en}
EVAL_LANG=${4:-en}
DATA_DIR=${5:-"$REPO/download/"}
OUT_DIR=${6:-"$REPO/outputs/"}

TASK=tydiqa

LR=3e-4
#NUM_EPOCHS=10
MAX_STEPS=10000

if [ $MODEL == "google/mt5-small" ] || [ $MODEL == "google/mt5-base" ] || [ $MODEL == "google/mt5-large" ]; then
  MODEL_TYPE="mt5"
  MAXL=1024
  MAX_ANSWER_LEN=512
  DOC_STRIDE=256
  BATCH_SIZE=4
  GRAD_ACC=8
elif [ $MODEL == "google/byt5-small" ] || [ $MODEL == "google/byt5-base" ] || [ $MODEL == "google/byt5-large" ]; then
  MODEL_TYPE="byt5"
  MAXL=2048
  MAX_ANSWER_LEN=512
  DOC_STRIDE=256
  BATCH_SIZE=2
  GRAD_ACC=16
fi


# Model path where trained model should be stored
MODEL_PATH=$OUT_DIR/${TASK}/in_language/${TRAIN_LANG}_${MODEL}_${MAX_STEPS}steps_LR${LR}_maxlen${MAXL}_batchsize${BATCH_SIZE}_gradacc${GRAD_ACC}
mkdir -p $MODEL_PATH
mkdir -p $MODEL_PATH/cache

# train
CUDA_VISIBLE_DEVICES=$GPU python third_party/run_t5_tydiqa.py \
  --model_name_or_path ${MODEL} \
  --task $TASK \
  --do_train \
  --train_file $DATA_DIR/tydiqa/tydiqa-goldp-v1.1-train-processed-t5/tydiqa.$TRAIN_LANG.train.json \
  --validation_file $DATA_DIR/tydiqa/tydiqa-goldp-v1.1-dev-processed-t5/tydiqa.$EVAL_LANG.dev.json \
  --train_lang $TRAIN_LANG \
  --eval_lang $EVAL_LANG \
  --context_column context \
  --question_column question \
  --answer_column answers \
  --max_seq_length ${MAXL} \
  --max_answer_length ${MAX_ANSWER_LEN} \
  --pad_to_max_length \
  --doc_stride ${DOC_STRIDE} \
  --output_dir ${MODEL_PATH} \
  --overwrite_output_dir \
  --overwrite_cache \
  --optim adafactor \
  --learning_rate ${LR} \
  --lr_scheduler_type "linear" \
  --warmup_steps 500 \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps ${GRAD_ACC} \
  --eval_accumulation_steps 4 \
  --max_steps ${MAX_STEPS} \
  --save_strategy steps --save_steps 500 \
  --logging_steps 20 --ignore_pad_token_for_loss \
  --predict_with_generate --generation_max_length $MAX_ANSWER_LEN \
  --generation_num_beams 30 \
  --evaluation_strategy steps --eval_steps 500 \
  --max_eval_samples 32




