REPO=$PWD
MODEL=${1:-google/mt5-base}
GPU=${2:-0}

TASK=tydiqa
BATCH_SIZE=8
GRAD_ACC=4

LR=1e-5
NUM_EPOCHS=10


if [ $MODEL == "google/mt5-small" ] || [ $MODEL == "google/mt5-base" ] || [ $MODEL == "google/mt5-large" ]; then
  MODEL_TYPE="mt5"
  MAXL=1024
  MAX_ANSWER_LEN=30
  DOC_STRIDE=256
elif [ $MODEL == "google/byt5-small" ] || [ $MODEL == "google/byt5-base" ] || [ $MODEL == "google/byt5-large" ]; then
  MODEL_TYPE="byt5"
  MAXL=1024
  MAX_ANSWER_LEN=100
  DOC_STRIDE=256
fi


# Model path where trained model should be stored
MODEL_PATH=$OUT_DIR/$SRC/${MODEL}_LR${LR}_EPOCH${NUM_EPOCHS}_maxlen${MAXL}_batchsize${BATCH_SIZE}_gradacc${GRAD_ACC}
mkdir -p $MODEL_PATH


# train
CUDA_VISIBLE_DEVICES=$GPU python third_party/run_t5.py \
  --model_name_or_path ${MODEL} \
  --do_train \
  --do_eval \
  --dataset_name tydiqa \
  --dataset_config_name secondary_task \
  --context_column context \
  --question_column question \
  --answer_column answers \
  --max_seq_length ${MAXL} \
  --max_answer_length ${MAX_ANSWER_LEN} \
  --doc_stride ${DOC_STRIDE} \
  --dataset_name ${TASK} \
  --output_dir ${MODEL_PATH} \
  --learning_rate ${LR} \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRAD_ACC} \
  --num_train_epochs ${NUM_EPOCHS} \
  --save_steps 200 \
  --num_beams 30 \
  --val_max_answer_length ${MAX_ANSWER_LEN} \




