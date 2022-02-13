#MODELS=("bert-base-multilingual-cased" "google/canine-s" "google/canine-c")
#TASKS=("udpos" "panx" "tydiqa")
REPO=$PWD

MODEL=${1:-bert-base-multilingual-cased}
TASK=${2:-udpos}
GPU=${3:-0}
DATA_DIR=${4:-"$REPO/download/"}
OUT_DIR=${5:-"$REPO/outputs-in-language/"}


if [ $TASK == 'udpos' ]; then
  LANGS=(af ar bg de el en es et eu fa fi fr he hi hu id it ja kk ko mr nl pt ru ta te th tl tr ur vi yo zh lt pl uk wo ro)
elif [ $TASK == 'panx' ]; then
  LANGS=(ar he vi id jv ms tl eu ml ta te af nl en de el bn hi mr ur fa fr it pt es bg ru ja ka ko th sw yo my zh kk tr et fi hu)
elif [ $TASK == 'tydiqa' ]; then
  LANGS=("ar" "bn" "en" "fi" "id" "ko" "ru" "sw" "te")
fi

MODELS=("google/canine-s" "google/canine-c")

# for MODEL in ${MODELS[@]}; do
for LANG in "${LANGS[@]}"; do
  echo "Fine-tuning $MODEL on $TASK using GPU $GPU"
  echo "Load data from $DATA_DIR, and save models to $OUT_DIR"
  bash $REPO/scripts/finetune_${TASK}.sh $MODEL $LANG $GPU $DATA_DIR $OUT_DIR
done
# done
