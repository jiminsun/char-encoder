# char-encoder

## Download data
Follow the steps described in `./xtreme` and run `./xtreme/scripts/download_data.sh`.

## Environment setup
* Follow the steps in `./xtreme`
* Install `wandb`
* For fp16 (mBERT, CANINE-S, CANINE-C), install [apex](https://github.com/NVIDIA/apex)

## Training
### Single language training
```
cd xtreme
bash in_language_finetuning.sh $MODEL $TASK $GPU
```
* `$MODEL` is either `bert-multilingual-base-cased`, `google/canine-s`, `google/canine-c`, `google/mt5-small`, or `google/byt5-small`
 * For larger t5 style models, change `small` to `base` or `large`
* `$TASK` is either `tydiqa`, `xnli`, or `panx`


### Multilingual training (multitasking)
#### mBERT, CANINE models

```
cd xtreme 
# tydiqa
bash scripts/train_tydiqa_all.sh $MODEL $GPU

# xnli
bash scripts/train_xnli.sh $MODEL $GPU

# panx
bash scripts/train_panx.sh $MODEL $GPU $LANGS
# LANGS: "en,hi,bn,ur,sw,ar,de,el,es,fi,fr,ko,ru,te,zh,tr,ta,id"
```

