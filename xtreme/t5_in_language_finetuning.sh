# MODELS=( "bert-base-multilingual-cased" "google/canine-s" "google/canine-c" )
#TASKS=("udpos" "panx" "tydiqa")

export WANDB_API_KEY="2f0079981993b2ca38c42cb4ad8df1a168394072"

REPO=$PWD

MODEL=${1:-google/mt5-small}
TASK=${2:-tydiqa}
GPU=${3:-0}
DATA_DIR=${4:-"$REPO/download/"}
OUT_DIR=${5:-"$REPO/outputs/"}


if [ $TASK == 'tydiqa' ]; then
  LANGS=( "en" "ko" "sw" "ar" "bn" "fi" "id" "ru" "te" )
elif [ $TASK == 'xnli' ]; then
  LANGS=( "en" "ar" "bg" "de" "el" "es" "fr" "hi" "ru" "sw" "th" "tr" "ur" "vi" "zh" )
elif [ $TASK == 'ner' ]; then
  LANGS=( "en" "hi" "bn" "ur" "sw" "ar" "de" "el" "es" "fi" "fr" "ko" "ru" "te" "zh" "tr" "ta" "id" )
fi

for LANG in "${LANGS[@]}"; do
  echo "Fine-tuning $MODEL on $TASK using GPU $GPU"
  echo "Load data from $DATA_DIR, and save models to $OUT_DIR"
  bash $REPO/t5_scripts/train_${TASK}.sh $MODEL $GPU $LANG $LANG $DATA_DIR $OUT_DIR
done
