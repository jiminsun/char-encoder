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

export CUDA_VISIBLE_DEVICES=$GPU

TASK='xnli'
LR=2e-5
EPOCH=3

LC=""
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
  MAXL=512
  FP="--fp16"
  BATCH_SIZE=8
  GRAD_ACC=4
elif [ $MODEL == "google/canine-s" ]; then
  MODEL_TYPE="canine"
  MAXL=2048
  FP="--fp16"
  BATCH_SIZE=8
  GRAD_ACC=4
elif [ $MODEL == "google/canine-c" ]; then
  MODEL_TYPE="canine"
  MAXL=2048
  FP="--fp16"
  BATCH_SIZE=8
  GRAD_ACC=4
fi




SAVE_DIR="$OUT_DIR/$TASK/in_language/${LANG}_${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}/"
mkdir -p $SAVE_DIR

CUDA_VISIBLE_DEVICES=$GPU python $PWD/third_party/run_classify.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL \
  --train_language $LANG \
  --predict_languages $LANG \
  --task_name $TASK \
  --do_train \
  --do_eval \
  --do_predict \
  --overwrite_cache \
  --overwrite_output_dir \
  --data_dir $DATA_DIR/${TASK} \
  --gradient_accumulation_steps $GRAD_ACC \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --num_train_epochs $EPOCH \
  --max_seq_length $MAXL \
  --output_dir $SAVE_DIR/ \
  --save_steps 5000 --logging_steps 50 \
  --eval_all_checkpoints \
  --log_file 'train' \
  --save_only_best_checkpoint \
  --eval_test_set $FP
