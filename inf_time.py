import argparse
from transformers import (
    BertForQuestionAnswering,
    BertForTokenClassification,
    BertForSequenceClassification,
    BertTokenizer,
    CanineForQuestionAnswering,
    CanineForTokenClassification,
    CanineForSequenceClassification,
    CanineTokenizer,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import torch
import json
import numpy as np
from datasets import load_dataset
import random
import torch.autograd.profiler as profiler

import wandb

random.seed(42)
wandb.init(project="inference", entity="jiminsun")

ROOT_DIR = "/data/private/char-encoder/xtreme/outputs"

MODEL_CLASS = {
    'bert-base-multilingual-cased': {
        'tydiqa': BertForQuestionAnswering,
        'xnli': BertForSequenceClassification,
        'ner': BertForTokenClassification
    },
    'canine': {
        'tydiqa': CanineForQuestionAnswering,
        'xnli': CanineForSequenceClassification,
        'ner': CanineForTokenClassification
    }
}


max_length = {
    'tydiqa': {
        'bert-base-multilingual-cased': 512,
        'google/canine-s': 2048,
        'google/canine-c': 2048,
        'google/mt5-small': 1024,
        'google/byt5-small': 4096
    },
    'xnli': {
        'bert-base-multilingual-cased': 512,
        'google/canine-s': 2048,
        'google/canine-c': 2048,
        'google/mt5-small': 1024,
        'google/byt5-small': 2048
    },
    'ner': {
        'bert-base-multilingual-cased': 512,
        'google/canine-s': 2048,
        'google/canine-c': 2048,
        'google/mt5-small': 1024,
        'google/byt5-small': 4096
    }
}

max_answer_length = {
    'tydiqa': {
        'google/mt5-small': 512,
        'google/byt5-small': 512,
    },
    'xnli': {
        'google/mt5-small': 16,
        'google/byt5-small': 56,
    },
    'ner': {
        'google/mt5-small': 128,
        'google/byt5-small': 512,
    }
}

checkpoints = {
    'tydiqa': {
        'bert-base-multilingual-cased': f'{ROOT_DIR}/tydiqa/in_language/en_bert-base-multilingual-cased_LR3e-5_maxsteps5000_maxlen512_batchsize8_gradacc4/checkpoint-5000',
        'google/canine-s': f'{ROOT_DIR}/tydiqa/in_language/en_google/canine-s_LR3e-5_maxsteps5000_maxlen2048_batchsize8_gradacc4',
        'google/canine-c': f'{ROOT_DIR}/tydiqa/in_language/en_google/canine-c_LR3e-5_maxsteps5000_maxlen2048_batchsize8_gradacc4',
        'google/mt5-small': f'{ROOT_DIR}/tydiqa/in_language/en_google/mt5-small_epoch1e-4_maxlen1024_batchsize4_gradacc8',
        'google/byt5-small': f'{ROOT_DIR}/tydiqa/in_language/en_google/byt5-small_5000steps_LR3e-4_maxlen2048_batchsize2_gradacc16/checkpoint-5000'
    },
    'xnli': {
        'bert-base-multilingual-cased': f'{ROOT_DIR}/xnli/in_language/en_bert-base-multilingual-cased-LR2e-5-epoch5-MaxLen512/checkpoint-best',
        'google/canine-s': f'{ROOT_DIR}/xnli/in_language/en_google/canine-s-LR2e-5-epoch3-MaxLen2048',
        'google/canine-c': f'{ROOT_DIR}/xnli/in_language/en_google/canine-c-LR2e-5-epoch5-MaxLen2048',
        'google/mt5-small': f'{ROOT_DIR}/xnli/in_language/en_google/mt5-small_LR1e-4_EPOCH10_batchsize2_gradacc8',
        'google/byt5-small': f'{ROOT_DIR}/xnli/in_language/en_google/byt5-small_LR3e-4_100000steps_batchsize2_gradacc8/checkpoint-5000'
    },
    'ner': {
        'bert-base-multilingual-cased': f'{ROOT_DIR}/panx/in_language/en_bert-base-multilingual-cased-LR2e-5-5000steps-MaxLen512',
        'google/canine-s': f'{ROOT_DIR}/panx/in_language/en_google-canine-s-LR2e-5-5000steps-MaxLen2048',
        'google/canine-c': f'{ROOT_DIR}/panx/in_language/en_google-canine-c-LR2e-5-5000steps-MaxLen2048',
        'google/mt5-small': f'{ROOT_DIR}/wikiann/in_language/en_google/mt5-small_final',
        'google/byt5-small': f'{ROOT_DIR}/wikiann/in_language/en_google/byt5-small_LR1e-4_10000steps_bsz8_gradacc4_maxlen512',
    }
}


def load_pretrained(task, model_name):
    print(f"*** Loading model {model_name} for {task}")
    ckpt = checkpoints[task][model_name]
    if 't5' in model_name:
        if model_name == 'google/mt5-small':
            mt5_config = AutoConfig.from_pretrained(ckpt)
            tok = AutoTokenizer.from_pretrained(ckpt)
            model = AutoModelForSeq2SeqLM.from_pretrained(ckpt, config=mt5_config)
        else:
            byt5_config = AutoConfig.from_pretrained(ckpt)
            tok = AutoTokenizer.from_pretrained(ckpt)
            model = AutoModelForSeq2SeqLM.from_pretrained(ckpt, config=byt5_config)
        return tok, model
    else:
        if 'bert' in model_name:
            tok = BertTokenizer.from_pretrained(ckpt)
            model = MODEL_CLASS[model_name][task].from_pretrained(ckpt)
        else:
            tok = CanineTokenizer.from_pretrained(ckpt)
            model = MODEL_CLASS['canine'][task].from_pretrained(ckpt)
        return tok, model


def load_task_dataset(args):
    inputs = []
    if args.task == 'tydiqa':
        fpath = '/data/private/char-encoder/xtreme/download/tydiqa/tydiqa-goldp-v1.1-dev/tydiqa.en.dev.json'
        data = json.load(open(fpath, 'r'))
        sample = random.sample(data['data'], args.sample)
        for d in sample:
            context = d['paragraphs'][0]['context']
            question = d['paragraphs'][0]['qas'][0]['answers'][0]['text']
            if "t5" in args.model:
                input_text = 'question: ' + question + "context: " + context + "</s>"
            else:
                input_text = question + ' <pad> ' + context
            inputs.append(input_text)
    elif args.task == 'xnli':
        data = load_dataset('xnli', 'en')['test']
        sample = random.sample(list(data), args.sample)
        for d in sample:
            premise = d['premise']
            hypothesis = d['hypothesis']
            if 't5' in args.model:
                input_text = 'premise: ' + premise + 'hypothesis: ' + hypothesis + '</s>'
            else:
                input_text = premise + '<sep>' + hypothesis
            inputs.append(input_text)
    else:
        data = load_dataset('wikiann', 'en')['test']
        sample = random.sample(list(data), args.sample)
        for d in sample:
            if 't5' in args.model:
                input_text = ' '.join(["tag:"] + d['tokens']) + " </s>"
            else:
                input_text = ' '.join(d['tokens'])
            inputs.append(input_text)
    return inputs


def main(args):
    wandb_info = [args.task, args.model]
    if 't5' in args.model:
        wandb_info.append(f'beam{args.beam}')

    wandb.run.name = '_'.join(wandb_info)
    device = torch.device("cuda")
    tok, model = load_pretrained(task=args.task, model_name=args.model)

    model_name = args.model
    model.to(device)

    raw_inputs = load_task_dataset(args)
    model_inputs = []

    repetitions = 10

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    print(model_name)
    timings = np.zeros((repetitions, args.sample))

    for d in raw_inputs:
        input_tokens = tok(d, return_tensors='pt',
                           truncation=True, max_length=max_length[args.task][model_name])
        model_inputs.append(input_tokens.to(device))

    # gpu warmup
    for d in model_inputs[:10]:
        if "t5" in model_name:
            num_beams = 1 if args.task == 'xnli' else args.beam
            _ = model.generate(input_ids=d['input_ids'],
                               attention_mask=d['attention_mask'],
                               max_length=max_answer_length[args.task][model_name],
                               num_beams=num_beams, early_stopping=True)
        else:
            _ = model(input_ids=d['input_ids'],
                      token_type_ids=d['token_type_ids'],
                      attention_mask=d['attention_mask'])

    with torch.no_grad():

        for rep in range(repetitions):
            for i, d in enumerate(model_inputs[:args.sample]):
                starter.record()
                if "t5" in model_name:
                    num_beams = 1 if args.task == 'xnli' else args.beam
                    _ = model.generate(input_ids=d['input_ids'],
                                       attention_mask=d['attention_mask'],
                                       max_length=max_answer_length[args.task][model_name],
                                       num_beams=num_beams, early_stopping=True)
                else:
                    _ = model(input_ids=d['input_ids'],
                              token_type_ids=d['token_type_ids'],
                              attention_mask=d['attention_mask'])
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                # Returns the maximum GPU memory occupied by tensors in bytes for a given device
                curr_max_memory = torch.cuda.max_memory_allocated(0)
                timings[rep, i] = curr_time

    print(f"*** Average inference time for {model_name} at {args.task}: {timings.mean()} ms")
    wandb.log({f'{args.task}/inference': timings.mean()})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='bert-base-multilingual-cased')
    p.add_argument('--task', type=str, default='tydiqa')
    p.add_argument('--beam', type=int, default=1)
    p.add_argument('--sample', type=int, default=100)

    args = p.parse_args()
    main(args)