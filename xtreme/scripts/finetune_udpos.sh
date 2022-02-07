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

TASK='udpos'
export CUDA_VISIBLE_DEVICES=$GPU
#LANGS='af,ar,bg,de,el,en,es,et,eu,fa,fi,fr,he,hi,hu,id,it,ja,kk,ko,mr,nl,pt,ru,ta,te,th,tl,tr,ur,vi,yo,zh,lt,pl,uk,ro'
NUM_EPOCHS=10
#MAX_LENGTH=128
LR=2e-5

LC=""
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
  MAXL=512
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
  MAXL=512
  LC=" --do_lower_case"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then
  MODEL_TYPE="xlmr"
  MAXL=512
elif [ $MODEL == "google/canine-s" ] || [ $MODEL == "google/canine-c" ]; then
  MODEL_TYPE="canine"
  MAXL=2048
elif [ $MODEL == "google/mt5-small" ] || [ $MODEL == "google/mt5-base" ] || [ $MODEL == "google/mt5-large" ]; then
  MODEL_TYPE="mt5"
  MAXL=1024
elif [ $MODEL == "google/byt5-small" ] || [ $MODEL == "google/byt5-base" ] || [ $MODEL == "google/byt5-large" ]; then
  MODEL_TYPE="byt5"
  MAXL=1024
fi

if [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-roberta-large" ]; then
  BATCH_SIZE=2
  GRAD_ACC=16
else
  BATCH_SIZE=8
  GRAD_ACC=4
fi

# preprocess
SAVE_DIR="$DATA_DIR/${TASK}/${LANG}_${MODEL}_processed_maxlen${MAXL}"
mkdir -p $SAVE_DIR
python3 $REPO/utils_preprocess.py \
  --data_dir $DATA_DIR/${TASK} \
  --task udpos_tokenize \
  --model_name_or_path $MODEL \
  --model_type $MODEL_TYPE \
  --max_len $MAXL \
  --output_dir $SAVE_DIR \
  --languages $LANG $LC >> $SAVE_DIR/process.log
if [ ! -f $SAVE_DIR/labels.txt ]; then
  echo "create label"
  cat $SAVE_DIR/*/*.${MODEL} | cut -f 2 | grep -v "^$" | sort | uniq > $SAVE_DIR/labels.txt
fi

OUTPUT_DIR="$OUT_DIR/$TASK/${LANG}_${MODEL}-LR${LR}-epoch${NUM_EPOCHS}-MaxLen${MAXL}"
mkdir -p $OUTPUT_DIR
python3 $REPO/third_party/run_tag.py \
  --data_dir $SAVE_DIR \
  --model_type $MODEL_TYPE \
  --labels $SAVE_DIR/labels.txt \
  --model_name_or_path $MODEL \
  --output_dir $OUTPUT_DIR \
  --max_seq_length $MAXL \
  --num_train_epochs $NUM_EPOCHS \
  --gradient_accumulation_steps $GRAD_ACC \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --save_steps 500 \
  --seed 1 \
  --learning_rate $LR \
  --do_train \
  --do_eval \
  --do_predict \
  --do_predict_dev \
  --evaluate_during_training \
  --train_langs $LANG \
  --predict_langs $LANG \
  --log_file $OUTPUT_DIR/train.log \
  --eval_all_checkpoints \
  --overwrite_output_dir \
  --save_only_best_checkpoint $LC
