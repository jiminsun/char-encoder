import wandb

wandb.init(project="t5", entity="jiminsun")

# https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_seq2seq_qa.py
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

from torch.utils.data import Dataset

import transformers
from transformers import (
    MT5Config,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    T5Config,
    T5ForConditionalGeneration,
    ByT5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    HfArgumentParser,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    set_seed,
)
from transformers.trainer_utils import (
    EvalLoopOutput,
    get_last_checkpoint,
)
from transformers.file_utils import is_apex_available, is_torch_tpu_available
from transformers.trainer_utils import (
    EvalPrediction,
    PredictionOutput,
)

import datasets
from datasets import load_dataset, load_metric, concatenate_datasets

from processors.t5_utils import ModelArguments, DataTrainingArguments, UserArguments

languages = {
    'wikiann': ['ko', 'ja']
        # ['ace', 'af', 'als', 'am', 'an', 'ang', 'ar', 'arc', 'arz', 'as', 'ast', 'ay', 'az', 'ba', 'bar',
        #         'bat-smg', 'be', 'be-x-old', 'bg', 'bh', 'bn', 'bo', 'br', 'bs', 'ca', 'cbk-zam', 'cdo', 'ce', 'ceb',
        #         'ckb', 'co', 'crh', 'cs', 'csb', 'cv', 'cy', 'da', 'de', 'diq', 'dv', 'el', 'eml', 'en', 'eo', 'es',
        #         'et', 'eu', 'ext', 'fa', 'fi', 'fiu-vro', 'fo', 'fr', 'frr', 'fur', 'fy', 'ga', 'gan', 'gd', 'gl',
        #         'gn', 'gu', 'hak', 'he', 'hi', 'hr', 'hsb', 'hu', 'hy', 'ia', 'id', 'ig', 'ilo', 'io', 'is', 'it',
        #         'ja', 'jbo', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'ksh', 'ku', 'ky', 'la', 'lb', 'li', 'lij', 'lmo',
        #         'ln', 'lt', 'lv', 'map-bms', 'mg', 'mhr', 'mi', 'min', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'mwl',
        #         'my', 'mzn', 'nap', 'nds', 'ne', 'nl', 'nn', 'no', 'nov', 'oc', 'or', 'os', 'pa', 'pdc', 'pl', 'pms',
        #         'pnb', 'ps', 'pt', 'qu', 'rm', 'ro', 'ru', 'rw', 'sa', 'sah', 'scn', 'sco', 'sd', 'sh', 'si', 'simple',
        #         'sk', 'sl', 'so', 'sq', 'sr', 'su', 'sv', 'sw', 'szl', 'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt',
        #         'ug', 'uk', 'ur', 'uz', 'vec', 'vep', 'vi', 'vls', 'vo', 'wa', 'war', 'wuu', 'xmf', 'yi', 'yo', 'zea',
        #         'zh', 'zh-classical', 'zh-min-nan', 'zh-yue'],
}


if is_apex_available():
    from apex import amp


if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl


try:
    from torch.utils.tensorboard import SummaryWriter

    _has_tensorboard = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter

        _has_tensorboard = True
    except ImportError:
        _has_tensorboard = False


def is_tensorboard_available():
    return _has_tensorboard


class NERSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    # def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str = "eval"):
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        eval_examples=None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_examples = self.eval_examples if eval_examples is None else eval_examples
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(eval_examples, output)
            metrics = self.compute_metrics(eval_preds)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def predict(self, predict_dataset, predict_examples, ignore_keys=None, metric_key_prefix: str = "test"):
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                predict_dataloader,
                description="Prediction",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        predictions = self.post_process_function(predict_examples, output)
        metrics = self.compute_metrics(predictions)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=predictions.predictions, label_ids=predictions.label_ids, metrics=metrics)


logger = logging.getLogger(__name__)


wikiann_column_name_mapping = {
    'wikiann': ('tokens', 'ner_tags', 'langs', 'spans')
}


if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


def main():
    parser = HfArgumentParser((UserArguments, ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args, model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # wandb logging

    # Wandb experiment name configuration
    wandb.config.update(model_args)

    run_name = [data_args.dataset_name, args.train_lang]

    # append model name
    if model_args.model_name_or_path.startswith('google/'):
        model_name = model_args.model_name_or_path[7:]  # strip 'google/'
        run_name.append(model_name)  # canine-s or canine-c
    else:
        run_name.append(model_args.model_name_or_path)

    run_name.append(wandb.run.name.split('-')[-1])  # experiment number
    wandb.run.name = '-'.join(run_name)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    assert data_args.dataset_name is not None
    # Downloading and loading a dataset from the hub.
    if args.train_lang == 'all':
        langs = languages[data_args.dataset_name]
        lang_datasets = [load_dataset(data_args.dataset_name, l, cache_dir=model_args.cache_dir) for l in langs]
        train_datasets = concatenate_datasets(
            [d['train'] for d in lang_datasets]
        )
        validation_datasets = concatenate_datasets(
            [d['validation'] for d in lang_datasets]
        )
        test_datasets = concatenate_datasets(
            [d['test'] for d in lang_datasets]
        )
        raw_datasets = {
            'train': train_datasets,
            'validation': validation_datasets,
            'test': test_datasets
        }
    else:
        raw_datasets = load_dataset(data_args.dataset_name, args.train_lang, cache_dir=model_args.cache_dir)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # Preprocessing the datasets.
    # We need to generate and tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Temporarily set max_answer_length for training.
    max_answer_length = data_args.max_answer_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_wikiann_batch(
            examples,
    ) -> Tuple[List[str], List[str]]:
        """
        https://github.com/google-research/multilingual-t5/blob/625e1ca79b12299ffb7a4920041b4aa72639522d/multilingual_t5/preprocessors.py#L128
        {
          'tokens': ['Sy', 'ander', 'seun', ',', 'Swjatopolk', ',', 'was', 'die',
                     'resultaat', 'van', '’n', 'buite-egtelike', 'verhouding', '.'],
          'ner_tags': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          'langs': ['af', 'af', 'af', 'af', 'af', 'af', 'af', 'af', 'af', 'af', 'af', 'af', 'af', 'af'],
          'spans': ['PER: Swjatopolk']
        }

        """

        def generate_input(_tokens):
            return " ".join(["tag:"] + _tokens) + " </s>"

        def generate_output(_spans, delimiter=' $$ '):
            if len(_spans) > 0:
                return delimiter.join(_spans) + " </s>"
            else:
                return "None </s>"

        inputs = [generate_input(tokens) for tokens in examples['tokens']]
        targets = [generate_output(spans) for spans in examples['spans']]
        return inputs, targets

    def preprocess_function(examples):
        inputs, targets = preprocess_wikiann_batch(examples)
        model_inputs = tokenizer(
            inputs,
            max_length=max_seq_length,
            padding=padding,
            truncation=True
        )
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            # labels = tokenizer(targets, max_length=max_answer_length, padding=padding, truncation=True)
            labels = tokenizer(targets, padding=padding)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Validation preprocessing
    def preprocess_validation_function(examples):
        inputs, targets = preprocess_wikiann_batch(examples)

        model_inputs = tokenizer(
            inputs,
            max_length=max_seq_length,
            padding=padding,
            truncation=True,
            # return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                # max_length=max_answer_length,
                padding=padding,
                # truncation=True
            )
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            # We will select sample from whole data if argument is specified
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        # Create train feature from dataset
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        if data_args.max_train_samples is not None:
            # Number of samples might increase during Feature Creation, We select only specified max samples
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            # We will select sample from whole data
            max_eval_samples = min(len(eval_examples), data_args.max_eval_samples)
            eval_examples = eval_examples.select(range(max_eval_samples))
        # Validation Feature Creation
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_examples.map(
                preprocess_validation_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        if data_args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_examples = predict_examples.select(range(data_args.max_predict_samples))
        # Predict Feature Creation
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_examples.map(
                preprocess_validation_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        if data_args.max_predict_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    f1_score = load_metric("f1")

    def compute_metrics(p: EvalPrediction):
        return f1_score.compute(predictions=p.predictions, references=p.label_ids)

    def tags_to_spans(tag_sequence, delimiter=' $$ '):
        """Extract spans from IOB1 or BIO tags."""
        tag_sequence_split = [x.strip() for x in tag_sequence.split(delimiter)]
        tags_entities = []
        for tag_entity in tag_sequence_split:
            tag_entity_split = tag_entity.split(':')
            if len(tag_entity_split) != 2:
                continue
            tag = tag_entity_split[0].strip()
            entity = tag_entity_split[1].strip()
            tags_entities.append((tag, entity))
        return tags_entities

    # Post-processing:
    def post_processing_function(
            examples: datasets.Dataset,
            outputs: EvalLoopOutput,
    ):
        # Decode the predicted tokens.
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        pred_tags = [tags_to_spans(seq) for seq in decoded_preds]
        references = examples['spans']
        return EvalPrediction(predictions=pred_tags, label_ids=references)

    # Initialize our Trainer
    trainer = NERSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        post_process_function=post_processing_function,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_answer_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        results = trainer.predict(predict_dataset, predict_examples)
        metrics = results.metrics

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()