REPO=$PWD
MODEL=$1
LANG=${2:-en}  # specific lang or all
GPU=${3:-0}

DATA_DIR=$REPO/download/
TASK=xnli

if grep -q "byt5" <<< "$MODEL"; then
    MAXL=4096
    MAX_ANSWER_LEN=8
    BATCH_SIZE=16
else
    MAXL=2048
    MAX_ANSWER_LEN=4
    BATCH_SIZE=16
fi

if [ $LANG == "all" ]; then
  LANGS=( "en" "ar" "bg" "de" "el" "es" "fr" "hi" "ru" "sw" "th" "tr" "ur" "vi" "zh" )
  for L in "${LANGS[@]}"; do
    echo "=== evaluating on $L"
    CUDA_VISIBLE_DEVICES=$GPU python third_party/run_t5_classification.py \
      --model_name_or_path ${MODEL} \
      --task ${TASK} \
      --output_dir ${MODEL} \
      --do_predict \
      --eval_lang $L \
      --dataset_name ${TASK} \
      --max_seq_length ${MAXL} \
      --output_dir ${MODEL} \
      --overwrite_cache \
      --per_device_eval_batch_size ${BATCH_SIZE} \
      --eval_accumulation_steps 4 \
      --predict_with_generate --generation_max_length ${MAX_ANSWER_LEN} \
      --generation_num_beams 5
  done
else
  echo "=== evaluating on $LANG"
    CUDA_VISIBLE_DEVICES=$GPU python third_party/run_t5_classification.py \
      --model_name_or_path ${MODEL} \
      --task ${TASK} \
      --output_dir ${MODEL} \
      --do_predict \
      --eval_lang ${LANG} \
      --dataset_name ${TASK} \
      --max_seq_length ${MAXL} \
      --output_dir ${MODEL} \
      --overwrite_cache \
      --per_device_eval_batch_size ${BATCH_SIZE} \
      --eval_accumulation_steps 4 \
      --predict_with_generate --generation_max_length ${MAX_ANSWER_LEN} \
      --generation_num_beams 5
fi

