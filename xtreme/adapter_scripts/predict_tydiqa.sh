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

# Script to obtain predictions using a trained model on XQuAD, TyDi QA, and MLQA.
REPO=$PWD
MODEL=${1:-bert-base-multilingual-cased}
MODEL_PATH=${2}
TGT=${3:-xquad}
lang=${4:-en}
GPU=${5:-0}
DATA_DIR=${6:-"$REPO/download/"}

MAXL=512
MAX_QUERY_LEN=64
MAX_ANSWER_LEN=30
DOC_STRIDE=128

if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then
  MODEL_TYPE="xlm-roberta"
elif [ $MODEL == "google/canine-s" ] || [ $MODEL == "google/canine-c" ]; then
  MODEL_TYPE="canine"
  MAXL=2048
  MAX_QUERY_LEN=256
  MAX_ANSWER_LEN=100
  DOC_STRIDE=512
elif [ $MODEL == "google/mt5-small" ] || [ $MODEL == "google/mt5-base" ] || [ $MODEL == "google/mt5-large" ]; then
  MODEL_TYPE="mt5"
elif [ $MODEL == "google/byt5-small" ] || [ $MODEL == "google/byt5-base" ] || [ $MODEL == "google/byt5-large" ]; then
  MODEL_TYPE="byt5"
fi

if [ ! -d "${MODEL_PATH}" ]; then
  echo "Model path ${MODEL_PATH} does not exist."
  exit
fi

DIR=${DATA_DIR}/${TGT}/

PREDICTIONS_DIR=${MODEL_PATH}/predictions
PRED_DIR=${PREDICTIONS_DIR}/$TGT/
mkdir -p "${PRED_DIR}"

if [ $TGT == 'xquad' ]; then
  langs=( en es de el ru tr ar vi th zh hi )
elif [ $TGT == 'mlqa' ]; then
  langs=( en es de ar hi vi zh )
elif [ $TGT == 'tydiqa' ]; then
  langs=( en ar bn fi id ko ru sw te )
fi

echo "************************"
echo ${MODEL}
echo "************************"

echo
echo "Predictions on $TGT"
# for lang in ${langs[@]}; do
echo "  $lang "
if [ $TGT == 'xquad' ]; then
TEST_FILE=${DIR}/xquad.$lang.json
elif [ $TGT == 'mlqa' ]; then
TEST_FILE=${DIR}/MLQA_V1/test/test-context-$lang-question-$lang.json
elif [ $TGT == 'tydiqa' ]; then
TEST_FILE=${DIR}/tydiqa-goldp-v1.1-dev/tydiqa.$lang.dev.json
fi

CUDA_VISIBLE_DEVICES=${GPU} python $REPO/third_party/adapter_run_squad.py \
--model_type ${MODEL_TYPE} \
--model_name_or_path ${MODEL_PATH} \
--do_eval \
--eval_lang ${lang} \
--predict_file "${TEST_FILE}" \
--max_seq_length ${MAXL} \
--max_query_length ${MAX_QUERY_LEN} \
--max_answer_length ${MAX_ANSWER_LEN} \
--doc_stride ${DOC_STRIDE} \
--output_dir "${PRED_DIR}"
# done

# Rename files to test pattern
#for lang in ${langs[@]}; do
mv $PRED_DIR/predictions_${lang}_.json $PRED_DIR/test-$lang.json
#done
