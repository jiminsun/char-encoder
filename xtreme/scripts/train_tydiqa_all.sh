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

# Script to train a model on the TyDiQA-GoldP train data of all languages.

REPO=$PWD
MODEL=${1:-bert-base-multilingual-cased}
GPU=${2:-0}
DATA_DIR=${3:-"$REPO/download/"}
OUT_DIR=${4:-"$REPO/outputs/"}

SRC=tydiqa
BATCH_SIZE=8
GRAD_ACC=4

#MAXL=384
LR=3e-5
MAX_STEPS=50000

if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
  MAXL=512
  MAX_QUERY_LEN=64
  MAX_ANSWER_LEN=30
  DOC_STRIDE=128
elif [ $MODEL == "google/canine-s" ] || [ $MODEL == "google/canine-c" ]; then
  MODEL_TYPE="canine"
  MAXL=2048
  MAX_QUERY_LEN=256
  MAX_ANSWER_LEN=100
  DOC_STRIDE=512
fi

# Model path where trained model should be stored
MODEL_PATH=$OUT_DIR/$SRC/multilingual/${MODEL}_LR${LR}_EPOCH${NUM_EPOCHS}_maxlen${MAXL}_batchsize${BATCH_SIZE}_gradacc${GRAD_ACC}
mkdir -p $MODEL_PATH

TASK_DATA_DIR=${DATA_DIR}/tydiqa
TRAIN_FILE=${TASK_DATA_DIR}/tydiqa-goldp-v1.1-train/tydiqa-goldp-v1.1-train.json
PREDICT_FILE=${TASK_DATA_DIR}/tydiqa-goldp-v1.1-dev/tydiqa.en.dev.json

# train
CUDA_VISIBLE_DEVICES=$GPU python third_party/run_squad.py \
  --model_type ${MODEL_TYPE} \
  --model_name_or_path ${MODEL} \
  --task_name ${SRC} \
  --do_train \
  --do_eval \
  --data_dir ${TASK_DATA_DIR} \
  --train_file ${TRAIN_FILE} \
  --predict_file ${PREDICT_FILE} \
  --per_gpu_train_batch_size ${BATCH_SIZE} \
  --learning_rate ${LR} \
  --max_steps ${MAX_STEPS} \
  --max_seq_length ${MAXL} \
  --max_query_length ${MAX_QUERY_LEN} \
  --max_answer_length ${MAX_ANSWER_LEN} \
  --doc_stride ${DOC_STRIDE} \
  --save_steps 10000 \
  --overwrite_output_dir --overwrite_cache \
  --gradient_accumulation_steps ${GRAD_ACC} \
  --warmup_steps 0.1 \
  --output_dir ${MODEL_PATH} \
  --weight_decay 0.0001 \
  --threads 8 \
  --local_rank -1 \
  --train_lang all \
  --eval_lang en --fp16

# predict
bash scripts/evaluate_tydiqa_all.sh $MODEL tydiqa tydiqa $GPU $DATA_DIR

