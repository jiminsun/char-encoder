REPO=$PWD
MODEL=$1
LANG=$2
GPU=$3

# base model
# MODEL=/data/private/char-encoder/xtreme/outputs/tydiqa/in_language/en_google/mt5-base_LR3e-4_EPOCH_maxlen1024_batchsize2_gradacc16/checkpoint-6000
# english model
# char-encoder/xtreme/outputs/tydiqa/in_language/en_google/mt5-small_LR3e-4_EPOCH_maxlen1024_batchsize4_gradacc8/checkpoint-15000

# multilingual model
# MODEL=/data/private/char-encoder/xtreme/outputs//tydiqa/google/mt5-base_LR1e-3_maxlen1024_batchsize2_gradacc32/checkpoint-10000/


DATA_DIR=$REPO/download
TASK=tydiqa
MAXL=2048
MAX_ANSWER_LEN=512
BATCH_SIZE=16

if [ $LANG == "all" ]; then
    LANGS=( "en" "ar" "bn" "fi" "id" "ko" "ru" "sw" "te" )
    for L in "${LANGS[@]}"; do
        echo "=== evaluating on $L"
        CUDA_VISIBLE_DEVICES=$GPU python third_party/run_t5_tydiqa.py \
          --model_name_or_path ${MODEL} \
          --output_dir $MODEL \
          --eval_lang $L \
          --validation_file $DATA_DIR/tydiqa/tydiqa-goldp-v1.1-dev-processed-t5/tydiqa.$L.dev.json \
          --task $TASK \
          --overwrite_cache \
          --do_predict \
          --context_column context \
          --question_column question \
          --answer_column answers \
          --max_seq_length $MAXL \
          --per_device_eval_batch_size ${BATCH_SIZE} \
          --eval_accumulation_steps 4 \
          --predict_with_generate --generation_max_length $MAX_ANSWER_LEN \
          --generation_num_beams 20
    done
else
    WANDB_MODE=offline
    CUDA_VISIBLE_DEVICES=$GPU python third_party/run_t5_tydiqa.py \
      --model_name_or_path ${MODEL} \
      --output_dir $MODEL \
      --train_lang $LANG \
      --eval_lang $LANG \
      --validation_file $DATA_DIR/tydiqa/tydiqa-goldp-v1.1-dev-processed-t5/tydiqa.$LANG.dev.json \
      --task $TASK \
      --do_predict \
      --overwrite_cache \
      --context_column context \
      --question_column question \
      --answer_column answers \
      --max_seq_length $MAXL \
      --per_device_eval_batch_size ${BATCH_SIZE} \
      --eval_accumulation_steps 4 \
      --predict_with_generate --generation_max_length $MAX_ANSWER_LEN \
      --generation_num_beams 20
fi




