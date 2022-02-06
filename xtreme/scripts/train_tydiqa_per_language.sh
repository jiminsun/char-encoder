REPO=$PWD
DATA_DIR="$REPO/download/"
OUT_DIR="$REPO/outputs-temp/"

MODELS=("bert-base-multilingual-cased" "google/canine-s" "google/canine-c")
TASKS=("pawsx" "xnli" "udpos" "panx")

for MODEL in "${MODELS[@]}"; do
  for TASK in "${TASKS[@]}"; do

    if [ $TASK == 'pawsx' ]; then
      LANGS=('de' 'en' 'es' 'fr' 'ja' 'ko' 'zh')
    elif [ $TASK == 'xnli' ]; then
      LANGS=('ar' 'bg' 'de' 'el' 'en' 'es' 'fr' 'hi' 'ru' 'sw' 'th' 'tr' 'ur' 'vi' 'zh')
    elif [ $TASK == 'udpos' ]; then
      LANGS=('af' 'ar' 'bg' 'de' 'el' 'en' 'es' 'et' 'eu' 'fa' 'fi' 'fr' 'he' 'hi' 'hu' 'id' 'it' 'ja' \
      'kk' 'ko' 'mr' 'nl' 'pt' 'ru' 'ta' 'te' 'th' 'tl' 'tr' 'ur' 'vi' 'yo' 'zh' 'lt' 'pl' 'uk' 'ro')
      bash $REPO/scripts/preprocess_udpos.sh $MODEL $DATA_DIR
      bash $REPO/scripts/train_udpos.sh $MODEL $GPU $DATA_DIR $OUT_DIR
    elif [ $TASK == 'panx' ]; then
      LANGS=('ar' 'he' 'vi' 'id' 'jv' 'ms' 'tl' 'eu' 'ml' 'ta' 'te' 'af' 'nl' 'en' 'de' 'el' 'bn' 'hi' \
      'mr' 'ur' 'fa' 'fr' 'it' 'pt' 'es' 'bg' 'ru' 'ja' 'ka' 'ko' 'th' 'sw' 'yo' 'my' 'zh' 'kk' 'tr' \
      'et' 'fi' 'hu' 'qu' 'pl' 'uk' 'az' 'lt' 'pa' 'gu' 'ro')
      bash $REPO/scripts/preprocess_panx.sh $MODEL $DATA_DIR
      bash $REPO/scripts/train_panx.sh $MODEL $GPU $DATA_DIR $OUT_DIR
    elif [ $TASK == 'xquad' ]; then
      bash $REPO/scripts/train_qa.sh $MODEL squad $TASK $GPU $DATA_DIR $OUT_DIR
    elif [ $TASK == 'mlqa' ]; then
      bash $REPO/scripts/train_qa.sh $MODEL squad $TASK $GPU $DATA_DIR $OUT_DIR
    elif [ $TASK == 'tydiqa' ]; then
      bash $REPO/scripts/train_qa.sh $MODEL tydiqa $TASK $GPU $DATA_DIR $OUT_DIR
    elif [ $TASK == 'bucc2018' ]; then
      bash $REPO/scripts/run_bucc2018.sh $MODEL $GPU $DATA_DIR $OUT_DIR
    elif [ $TASK == 'tatoeba' ]; then
      bash $REPO/scripts/run_tatoeba.sh $MODEL $GPU $DATA_DIR $OUT_DIR
    fi

    echo "Fine-tuning $MODEL on $TASK using GPU $GPU"
    echo "Load data from $DATA_DIR, and save models to $OUT_DIR"
#    echo "bash $REPO/scripts/train.sh $MODEL $TASK 0 $DATA_DIR $OUT_DIR"
    bash $REPO/scripts/train.sh $MODEL $TASK 0 $DATA_DIR $OUT_DIR
  done
done

LANGS=('ar' 'bn' 'en' 'fi' 'id' 'ko' 'ru' 'sw' 'te')
for LANG in "${LANGS[@]}"; do
  TASK_DATA_DIR=${DATA_DIR}/tydiqa
  TRAIN_FILE=${TASK_DATA_DIR}/tydiqa-goldp-v1.1-train/tydiqa.goldp.${LANG}.train.json
  PREDICT_FILE=${TASK_DATA_DIR}/tydiqa-goldp-v1.1-dev/tydiqa.goldp.${LANG}.dev.json

REPO=$PWD
MODEL=${1:-bert-base-multilingual-cased}
SRC=${2:-squad}
TGT=${3:-xquad}
GPU=${4:-0}
DATA_DIR=${5:-"$REPO/download/"}
OUT_DIR=${6:-"$REPO/outputs/"}

BATCH_SIZE=4
GRAD_ACC=8

MAXL=384
LR=3e-5
NUM_EPOCHS=3.0
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then
  MODEL_TYPE="xlm-roberta"
elif [ $MODEL == "google/canine-s" ] || [ $MODEL == "google/canine-c" ]; then
  MODEL_TYPE="canine"
fi

# Model path where trained model should be stored
MODEL_PATH=$OUT_DIR/$SRC/${MODEL}_LR${LR}_EPOCH${NUM_EPOCHS}_maxlen${MAXL}_batchsize${BATCH_SIZE}_gradacc${GRAD_ACC}
mkdir -p $MODEL_PATH
# Train either on the SQuAD or TyDiQa-GoldP English train file
if [ $SRC == 'squad' ]; then
  TASK_DATA_DIR=${DATA_DIR}/squad
  TRAIN_FILE=${TASK_DATA_DIR}/train-v1.1.json
  PREDICT_FILE=${TASK_DATA_DIR}/dev-v1.1.json
else
  TASK_DATA_DIR=${DATA_DIR}/tydiqa
  TRAIN_FILE=${TASK_DATA_DIR}/tydiqa-goldp-v1.1-train/tydiqa.goldp.en.train.json
  PREDICT_FILE=${TASK_DATA_DIR}/tydiqa-goldp-v1.1-dev/tydiqa.goldp.en.dev.json
fi

# train
CUDA_VISIBLE_DEVICES=$GPU python third_party/run_squad.py \
  --model_type ${MODEL_TYPE} \
  --model_name_or_path ${MODEL} \
  --do_train \
  --do_eval \
  --data_dir ${TASK_DATA_DIR} \
  --train_file ${TRAIN_FILE} \
  --predict_file ${PREDICT_FILE} \
  --per_gpu_train_batch_size ${BATCH_SIZE} \
  --learning_rate ${LR} \
  --num_train_epochs ${NUM_EPOCHS} \
  --max_seq_length $MAXL \
  --doc_stride 128 \
  --save_steps -1 \
  --overwrite_output_dir \
  --gradient_accumulation_steps ${GRAD_ACC} \
  --warmup_steps 500 \
  --output_dir ${MODEL_PATH} \
  --weight_decay 0.0001 \
  --threads 8 \
  --train_lang en \
  --eval_lang en

# predict
bash scripts/predict_qa.sh $MODEL $MODEL_PATH $TGT $GPU $DATA_DIR
