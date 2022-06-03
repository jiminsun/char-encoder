MODEL=$1
TASK=${2:-tydiqa}  # tydiqa, xnli, or panx
LANG=${3:-en}
GPU=${4:-0}

REPO=$PWD

bash $REPO/t5_scripts/test_${TASK}.sh $LANG $GPU