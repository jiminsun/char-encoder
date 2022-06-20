REPO=$PWD
MODEL=$1
LANG=${2:-en}
GPU=${3:-0}
NUM_BEAM=${4:-1}

TASK=wikiann


if grep -q "byt5" <<< "$MODEL"; then
    BATCH_SIZE=2
    MAXL=2048
    MAX_ANSWER_LEN=512
    echo "max length $MAXL"
else
    BATCH_SIZE=2
    MAXL=512
    MAX_ANSWER_LEN=128
fi


LANGS=( "en" "ar" "bn" "de" "el" "es" "fi" "fr" "hi" "id" "ja" "ko" "ru" "sw" "ta" "te" "th" "tr" "ur" "zh" )


# train
if [ "${LANG}" == "all" ]; then
    for L in "${LANGS[@]}"; do
    echo "evaluating on $L"
    CUDA_VISIBLE_DEVICES=$GPU python third_party/run_t5_ner.py \
      --model_name_or_path ${MODEL} \
      --task wikiann \
      --output_dir ${MODEL} \
      --eval_lang ${L} \
      --do_predict \
      --dataset_name ${TASK} \
      --max_seq_length ${MAXL} \
      --overwrite_cache \
      --per_device_eval_batch_size ${BATCH_SIZE} \
      --eval_accumulation_steps 4 \
      --predict_with_generate --generation_max_length $MAX_ANSWER_LEN \
      --generation_num_beams $NUM_BEAM
    done
else
    CUDA_VISIBLE_DEVICES=$GPU python third_party/run_t5_ner.py \
      --model_name_or_path ${MODEL} \
      --task wikiann \
      --output_dir ${MODEL} \
      --eval_lang ${LANG} \
      --do_predict \
      --dataset_name ${TASK} \
      --max_seq_length ${MAXL} \
      --overwrite_cache \
      --per_device_eval_batch_size ${BATCH_SIZE} \
      --eval_accumulation_steps 4 \
      --predict_with_generate --generation_max_length $MAX_ANSWER_LEN \
      --generation_num_beams $NUM_BEAM
fi
