# This code supports only CANINE models for now!

REPO=$PWD
MODEL=${1:canine-s}
TASK=tydiqa_primary
DATA_DIR=${2:-"$REPO/download/$TASK"}
OUT_DIR=${3:-"$REPO/outputs/$TASK"}

MAXL=2048
MAX_QUERY_LEN=256
MAX_ANSWER_LEN=100
DOC_STRIDE=512

BATCH_SIZE=8
GRAD_ACC=16
NUM_EPOCHS=10

TRAIN_FILE=${DATA_DIR}/tydiqa-v1.0-train.jsonl
PREDICT_FILE=${DATA_DIR}/tydiqa-v1.0-dev.jsonl
CONFIG_FILE=${DATA_DIR}/model/$MODEL/canine_config.json

OUTPUT_DIR=$OUT_DIR/${MODEL}_EPOCH${NUM_EPOCHS}_maxlen${MAXL}_batchsize${BATCH_SIZE}_gradacc${GRAD_ACC}
mkdir -p $OUTPUT_DIR

python third_party/run_tydiqa.py \
  --model_type canine --model_name_or_path google/$MODEL \
  --do_train --do_eval \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --train_file $TRAIN_FILE \
  --predict_file $PREDICT_FILE \
  --cache_dir $OUTPUT_DIR/cache \
  --overwrite_output_dir --num_train_epochs $NUM_EPOCHS \
  --gradient_accumulation_steps $GRAD_ACC --per_gpu_eval_batch_size $BATCH_SIZE \
  --max_seq_length $MAXL --max_query_len $MAX_QUERY_LEN \
  --max_answer_length $MAX_ANSWER_LEN --doc_stride $DOC_STRIDE \
  --threads 8 --warmup_steps 0.1 --save_steps 1000