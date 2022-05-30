#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

REPO=$PWD
MODEL=${1:-bert-base-multilingual-cased}
LANG=${2:-en}
GPU=${3:-0}
DATA_DIR=${4:-"$REPO/download/"}
OUT_DIR=${5:-"$REPO/outputs/"}

TASK='panx'

NUM_EPOCHS=10
#MAX_STEPS=5000
LR=2e-5

LC=""
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
  MAXL=512
elif [ $MODEL == "google/canine-s" ] || [ $MODEL == "google/canine-c" ]; then
  MODEL_TYPE="canine"
  MAXL=2048
fi

BATCH_SIZE=8
GRAD_ACC=4


MODEL_PREFIX=$(echo $MODEL | sed "s/\//-/")  # change google/canine-c to google-canine-c
echo "$MODEL_PREFIX"

SAVE_DIR="$DATA_DIR/${TASK}/${LANG}_${MODEL_PREFIX}_processed_maxlen${MAXL}"
mkdir -p $SAVE_DIR
if [ -f "$SAVE_DIR/labels.txt" ]; then
    echo "preprocessed labels exist"
else
    python3 $REPO/utils_preprocess.py \
      --data_dir $DATA_DIR/$TASK/ \
      --task panx_tokenize \
      --model_name_or_path $MODEL \
      --model_prefix $MODEL_PREFIX \
      --model_type $MODEL_TYPE \
      --max_len $MAXL \
      --output_dir $SAVE_DIR \
      --languages $LANG $LC >> $SAVE_DIR/preprocess.log
      cat $SAVE_DIR/*/*.${MODEL_PREFIX} | cut -f 2 | grep -v "^$" | sort | uniq > $SAVE_DIR/labels.txt
fi

    
OUTPUT_DIR="$OUT_DIR/${TASK}/in_language/${LANG}_${MODEL_PREFIX}-LR${LR}-${MAX_STEPS}steps-MaxLen${MAXL}/"
mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=$GPU python $REPO/third_party/run_tag.py \
  --data_dir $SAVE_DIR \
  --model_type $MODEL_TYPE \
  --task_name $TASK \
  --train_langs $LANG \
  --predict_langs $LANG \
  --labels $SAVE_DIR/labels.txt \
  --model_name_or_path $MODEL \
  --output_dir $OUTPUT_DIR \
  --max_seq_length  $MAXL \
  --num_train_epochs $NUM_EPOCHS \
  --gradient_accumulation_steps $GRAD_ACC \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --per_gpu_eval_batch_size 32 \
  --save_steps 1000 \
  --logging_steps 50 \
  --learning_rate $LR \
  --do_train \
  --do_eval \
  --do_predict \
  --predict_langs $LANG \
  --train_langs $LANG \
  --log_file $OUTPUT_DIR/train.log \
  --eval_all_checkpoints \
  --eval_patience -1 \
  --overwrite_output_dir \
  --overwrite_cache --fp16
