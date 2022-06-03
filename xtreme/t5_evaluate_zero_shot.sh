MODEL=$1
TASK=${2:-tydiqa}  # tydiqa, xnli, or panx
GPU=${3:-0}

REPO=$PWD

bash $REPO/t5_scripts/test_${TASK}.sh all $GPU 