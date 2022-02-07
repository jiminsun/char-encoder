REPO=$PWD
DATA_DIR="$REPO/download/"
OUT_DIR="$REPO/outputs-temp/"
GPU=$1
MODEL=$2

MODELS=("bert-base-multilingual-cased" "google/canine-s" "google/canine-c")
TASKS=("pawsx" "xnli" "panx" "tydiqa" "udpos")

# for MODEL in "${MODELS[@]}"; do
for TASK in "${TASKS[@]}"; do
    echo "Fine-tuning $MODEL on $TASK using GPU $GPU"
    echo "Load data from $DATA_DIR, and save models to $OUT_DIR"
    bash $REPO/scripts/train.sh $MODEL $TASK $GPU $DATA_DIR $OUT_DIR
    # done
done