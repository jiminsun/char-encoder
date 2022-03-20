import json
import logging
import os
from functools import partial
from multiprocessing import Pool, cpu_count
import enum
import typing
import collections
import random

from tqdm import tqdm
from typing import Any, Dict, List, Mapping, Optional, Sequence, Text, Tuple

from transformers.file_utils import is_tf_available, is_torch_available
from transformers import DataProcessor
import data
import preproc
import char_splitter


if is_torch_available():
    import torch
    from torch.utils.data import TensorDataset

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text, lang='en', lang2id=None):
    """Returns tokenized answer spans that better match the annotated answer."""
    if lang2id is None:
        tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))
    else:
        tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text, lang=lang))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start: (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def tydi_convert_example_to_features(
        tydi_example,
        tokenizer,
        is_training,
        max_question_length,
        max_seq_length,
        doc_stride,
        include_unknowns,
        errors,
        debug_info=None
):
    # https://github.com/google-research/language/blob/13fd14e1b285002412252097586f8fe405ba8a24/language/canine/tydiqa/preproc.py#L406

    """Converts a single `TyDiExample` into a list of InputFeatures.
      Args:
        tydi_example: `TyDiExample` from a single JSON line in the corpus.
            attributes:
                example_id,
                question,
                contexts,
                answer,
                start_byte_offset,
                end_byte_offset,
                is_impossible=False,
                language='en'
        tokenizer: Tokenizer object that supports `tokenize` and
          `tokenize_with_offsets`.
        is_training: Are we generating these examples for training? (as opposed to
          inference).
        max_question_length: see FLAGS.max_question_length.
        max_seq_length: see FLAGS.max_seq_length.
        doc_stride: see FLAGS.doc_stride.
        include_unknowns: see FLAGS.include_unknowns.
        errors: List to be populated with error strings.
        debug_info: Dict to be populated with debugging information (e.g. how the
          strings were tokenized, etc.)
      Returns:
        List of `InputFeature`s.
    """

    features = []
    question_wordpieces = tokenizer.tokenize(tydi_example.question)

    all_doc_wp, contexts_start_offsets, contexts_end_offsets, offset_to_wp = (
        tokenizer.tokenize_with_offsets(tydi_example.contexts)
    )

    # Check invariants.
    for i in contexts_start_offsets:
        if i > 0:
            assert i < len(tydi_example.context_to_plaintext_offset), (
                "Expected {} to be in `context_to_plaintext_offset` "
                "byte_len(contexts)={}".format(i,
                                               data.byte_len(tydi_example.contexts)))
    for i in contexts_end_offsets:
        if i > 0:
            assert i < len(tydi_example.context_to_plaintext_offset), (
                "Expected {} to be in `context_to_plaintext_offset` "
                "byte_len(contexts)={}".format(i,
                                               data.byte_len(tydi_example.contexts)))

    # The offsets `contexts_start_offsets` and `contexts_end_offsets` are
    # initially in terms of `tydi_example.contexts`, but we need them with regard
    # to the original plaintext from the input corpus.
    # `wp_start_offsets` and `wp_end_offsets` are byte-wise offsets with regard
    # to the original corpus plaintext.
    wp_start_offsets, wp_end_offsets = create_mapping(
        contexts_start_offsets, contexts_end_offsets,
        tydi_example.context_to_plaintext_offset)

    if len(question_wordpieces) > max_question_length:
        # Keeps only the last `max_question_length` wordpieces of the question.
        question_wordpieces = question_wordpieces[-max_question_length:]
    # Inserts the special question marker in front of the question.
    question_wordpieces.insert(0, tokenizer.get_vocab_id("[Q]"))
    if debug_info is not None:
        debug_info["query_wp_ids"] = question_wordpieces

    # DOCUMENT PROCESSING
    # The -3 accounts for
    # 1. [CLS] -- Special BERT class token, which is always first.
    # 2. [SEP] -- Special separator token, placed after question.
    # 3. [SEP] -- Special separator token, placed after article content.
    # [CLS] [Q] question ... [SEP] ... context ... [SEP]
    max_wordpieces_for_doc = max_seq_length - len(question_wordpieces) - 3
    assert max_wordpieces_for_doc >= 0
    if debug_info is not None:
        debug_info["all_doc_wp_ids"] = all_doc_wp

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of up to our max length with a stride of `doc_stride`.
    doc_span = collections.namedtuple("DocSpan", ["start", "length"])
    doc_spans = []
    doc_span_start_wp_offset = 0

    while doc_span_start_wp_offset < len(all_doc_wp):
        length = len(all_doc_wp) - doc_span_start_wp_offset
        length = min(length, max_wordpieces_for_doc)
        doc_spans.append(doc_span(start=doc_span_start_wp_offset, length=length))
        if doc_span_start_wp_offset + length == len(all_doc_wp):
            break
        doc_span_start_wp_offset += min(length, doc_stride)

    # Answer processing
    if is_training:
        # Get start and end position
        assert tydi_example.start_byte_offset is not None
        assert tydi_example.end_byte_offset is not None

        # start_position = tydi_example.start_position
        # end_position = tydi_example.end_position

        # If the answer cannot be found in the text, then skip this example.
        # actual_text = " ".join(example.doc_tokens[start_position: (end_position + 1)])
        # actual_text = tydi_example.contexts
        # cleaned_answer_text = " ".join(whitespace_tokenize(tydi_example.answer_text))
        # if actual_text.find(cleaned_answer_text) == -1:
        #     logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
        #     return []

        has_wordpiece = False
        for i in range(tydi_example.start_byte_offset,
                       tydi_example.end_byte_offset):
            if offset_to_wp.get(i, -1) >= 0:
                has_wordpiece = True
                break
        if not has_wordpiece:
            if debug_info is not None:
                searched_offset_to_wp = []
                for i in range(tydi_example.start_byte_offset,
                               tydi_example.end_byte_offset):
                    searched_offset_to_wp.append(i)
                debug_info["offset_to_wp"] = offset_to_wp
                debug_info["searched_offset_to_wp"] = searched_offset_to_wp
            # It looks like the most likely cause of these issues is not having
            # whitespace between Latin/non-Latin scripts, which causes the tokenizer
            # to produce large chunks of non-wordpieced output. Unsurprisingly, the
            # vast majority of such problems arise in Thai and Japanese, which
            # typically do not use space between words.
            errors.append(
                "All byte indices between start/end offset have no assigned "
                "wordpiece.")
            return []

        # Find the indices of the first and last (inclusive) pieces of the answer
        # span. The end_byte_offset is exclusive, so we start our search for the
        # last piece from `end_byte_offset - 1` to ensure we do not select the piece
        # that follows the span.
        assert tydi_example.start_byte_offset <= tydi_example.end_byte_offset
        wp_start_position = find_nearest_wordpiece_index(
            tydi_example.start_byte_offset, offset_to_wp, scan_right=True)
        wp_end_position = find_nearest_wordpiece_index(
            tydi_example.end_byte_offset - 1, offset_to_wp, scan_right=False)

        # Sometimes there's no wordpiece at all for all the offsets in answer,
        # in such case, treat it as a null example.
        if wp_start_position == -1:
            errors.append("No starting wordpiece found.")
            return []
        if wp_end_position == -1:
            errors.append("No ending wordpiece found.")
            return []

    for doc_span_index, doc_span in enumerate(doc_spans):
        wps = []
        wps.append(tokenizer.get_vocab_id("[CLS]"))
        segment_ids = []
        segment_ids.append(0)
        wps.extend(question_wordpieces)
        segment_ids.extend([0] * len(question_wordpieces))
        wps.append(tokenizer.get_vocab_id("[SEP]"))
        segment_ids.append(0)

        wp_start_offset = [-1] * len(wps)
        wp_end_offset = [-1] * len(wps)

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            wp_start_offset.append(wp_start_offsets[split_token_index])
            wp_end_offset.append(wp_end_offsets[split_token_index])
            wps.append(all_doc_wp[split_token_index])
            segment_ids.append(1)
        wps.append(tokenizer.get_vocab_id("[SEP]"))
        wp_start_offset.append(-1)
        wp_end_offset.append(-1)
        segment_ids.append(1)
        assert len(wps) == len(segment_ids)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(wps)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(wps)
        padding = [0] * padding_length
        padding_offset = [-1] * padding_length
        wps.extend(padding)
        input_mask.extend(padding)
        segment_ids.extend(padding)
        wp_start_offset.extend(padding_offset)
        wp_end_offset.extend(padding_offset)

        assert len(wps) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(wp_start_offset) == max_seq_length
        assert len(wp_end_offset) == max_seq_length

        start_position = None
        end_position = None
        answer_type = None
        answer_text = ""
        wp_error = False
        if is_training:
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            contains_an_annotation = (
                    wp_start_position >= doc_start and wp_end_position <= doc_end)
            if ((not contains_an_annotation) or
                    tydi_example.answer.type == data.AnswerType.UNKNOWN):
                # If an example has unknown answer type or does not contain the answer
                # span, then we only include it with probability --include_unknowns.
                # When we include an example with unknown answer type, we set the first
                # token of the passage to be the annotated short span.
                if (include_unknowns < 0 or random.random() > include_unknowns):
                    continue
                start_position = 0
                end_position = 0
                answer_type = data.AnswerType.UNKNOWN
            else:
                doc_offset = len(question_wordpieces) + 2  # one for CLS, one for SEP.

                start_position = wp_start_position - doc_start + doc_offset
                end_position = wp_end_position - doc_start + doc_offset
                answer_type = tydi_example.answer.type
                answer_start_byte_offset = wp_start_offset[start_position]
                answer_end_byte_offset = wp_end_offset[end_position]
                answer_text = tydi_example.contexts[
                              answer_start_byte_offset:answer_end_byte_offset]

                try:
                    assert answer_start_byte_offset > -1
                    assert answer_end_byte_offset > -1
                    assert end_position >= start_position
                except AssertionError:
                    errors.append("wp_error")
                    wp_error = True
                    logging.info(wp_start_position, wp_end_position)
                    logging.info(tydi_example.start_byte_offset,
                                 tydi_example.end_byte_offset)
                    logging.info(start_position, end_position)
                    logging.info(doc_offset, doc_start)
                    logging.info(tydi_example.example_id)
                    logging.info("Error: end position smaller than start: %d",
                                 tydi_example.example_id)

        feature = TydiFeatures(
            unique_id=-1,  # this gets assigned afterwards.
            example_index=tydi_example.example_id,
            language=tydi_example.language,
            doc_span_index=doc_span_index,
            wp_start_offset=wp_start_offset,
            wp_end_offset=wp_end_offset,
            input_ids=wps,
            input_mask=input_mask,
            segment_ids=segment_ids,
            start_position=start_position,
            end_position=end_position,
            answer_text=answer_text,
            answer_type=answer_type
        )
        # unique_id: int,
        # example_index: int,
        # language: str,
        # doc_span_index: int,
        # wp_start_offset: Sequence[int],
        # wp_end_offset: Sequence[int],
        # input_ids: Sequence[int],
        # input_mask: Sequence[int],
        # context_len: int,
        # segment_ids: Sequence[int],
        # start_position: Optional[int] = None,
        # end_position: Optional[int] = None,
        # answer_text: Text = "",
        # answer_type: int = 3,  # minimal is default
        if not wp_error:
            features.append(feature)
    return features


def find_nearest_wordpiece_index(offset_index: int,
                                 offset_to_wp: Mapping[int, int],
                                 scan_right: bool = True) -> int:
    """According to offset_to_wp dictionary, find the wordpiece index for offset.

  Some offsets do not have mapping to word piece index if they are delimited.
  If scan_right is True, we return the word piece index of nearest right byte,
  nearest left byte otherwise.

  Args:
    offset_index: the target byte offset.
    offset_to_wp: a dictionary mapping from byte offset to wordpiece index.
    scan_right: When there is no valid wordpiece for the offset_index, will
      consider offset_index+i if this is set to True, offset_index-i otherwise.

  Returns:
    The index of the nearest word piece of `offset_index`
    or -1 if no match is possible.
  """
    for i in range(0, len(offset_to_wp.items())):
        next_ind = offset_index + i if scan_right else offset_index - i
        if next_ind >= 0 and next_ind in offset_to_wp:
            return_ind = offset_to_wp[next_ind]
            # offset has a match.
            if return_ind > -1:
                return return_ind
    return -1


def create_mapping(
        start_offsets: Sequence[int],
        end_offsets: Sequence[int],
        context_to_plaintext_offset: Sequence[int],
) -> Tuple[List[int], List[int]]:
    """Creates a mapping from context offsets to plaintext offsets.

  Args:
    start_offsets: List of offsets relative to a TyDi entry's `contexts`.
    end_offsets: List of offsets relative to a TyDi entry's `contexts`.
    context_to_plaintext_offset: Mapping `contexts` offsets to plaintext
      offsets.

  Returns:
    List of offsets relative to the original corpus plaintext.
  """

    plaintext_start_offsets = [
        context_to_plaintext_offset[i] if i >= 0 else -1 for i in start_offsets
    ]
    plaintext_end_offsets = [
        context_to_plaintext_offset[i] if i >= 0 else -1 for i in end_offsets
    ]
    return plaintext_start_offsets, plaintext_end_offsets


def tydi_convert_examples_to_features(
        examples,
        tokenizer,
        is_training,
        max_seq_length,
        max_question_length,
        doc_stride,
        include_unknowns,
        return_dataset=False,
        threads=1,
        lang2id=None
):
    # https://github.com/google-research/language/blob/13fd14e1b285002412252097586f8fe405ba8a24/language/canine/tydiqa/preproc.py#L312

    """
    Converts a list of examples into a list of features that can be directly given as input to a model.
    It is model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset,
            if 'tf': returns a tf.data.Dataset
        threads: multiple processing threadsa-smi


    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`

    Example::

        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)

        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """

    # Defining helper methods
    threads = min(threads, cpu_count())
    with Pool(threads) as p:
        annotate_ = partial(
            tydi_convert_example_to_features,
            tokenizer=tokenizer,
            is_training=is_training,
            max_question_length=max_question_length,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            include_unknowns=include_unknowns,
            errors=[],
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert tydiqa examples to features",
            )
        )

    # Process features
    # https://github.com/google-research/language/blob/13fd14e1b285002412252097586f8fe405ba8a24/language/canine/tydiqa/tf_io.py#L140

    new_features = []
    # unique_id = 1000000000
    # example_index = 0
    for example_features in tqdm(features, total=len(features), desc="add example index and unique id"):
        if not example_features:
            continue
        for example_feature in example_features:
            # example_feature.example_index = example_index
            example_feature.unique_id = example_feature.example_index + example_feature.doc_span_index
            new_features.append(example_feature)
        # example_index += 1
    features = new_features
    del new_features
    if return_dataset == "pt":
        if not is_torch_available():
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

        # Convert to Tensors and build dataset
        all_unique_ids = torch.tensor([f.unique_id for f in features], dtype=torch.long)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_masks = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        # all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        # all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        # all_langs = torch.tensor([f.langs for f in features], dtype=torch.long)

        if is_training:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            all_answer_types = torch.tensor([f.answer_type for f in features], dtype=torch.long)

            dataset = TensorDataset(
                all_unique_ids,
                all_input_ids,
                all_input_masks,
                all_segment_ids,
                all_start_positions,
                all_end_positions,
                all_answer_types
            )
        else:
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            all_wp_start_offset = torch.tensor([f.wp_start_offset for f in features],
                                               dtype=torch.long)
            all_wp_end_offset = torch.tensor([f.wp_end_offset for f in features],
                                             dtype=torch.long)

            dataset = TensorDataset(
                all_input_ids,
                all_input_masks,
                all_segment_ids,
                all_example_index,
                all_wp_start_offset,
                all_wp_end_offset
            )

        return features, dataset


class TyDiProcessor(DataProcessor):
    """
    Processor for the TyDiQA data set.
    Overriden by TyDiV1Processor, used by the version 1.0 of TyDiQA
    """

    train_file = 'tydiqa-v1.0-train.jsonl'
    dev_file = 'tydiqa-v1.0-dev.jsonl'

    def get_train_examples(
            self,
            data_dir,
            filename=None,
            max_passages=45,
            max_position=45,
            fail_on_invalid=False
    ):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

        """

        if data_dir is None:
            data_dir = ""

        with open(
                os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = [json.loads(line) for line in reader]

        splitter = char_splitter.CharacterSplitter()
        entries = [preproc.create_entry_from_json(json_elem,
                                                  splitter,
                                                  max_passages=max_passages,
                                                  max_position=max_position,
                                                  fail_on_invalid=fail_on_invalid) for json_elem in tqdm(input_data)]
        return self._create_examples(entries, "train")

    def get_dev_examples(
            self,
            data_dir,
            filename=None,
            max_passages=45,
            max_position=45,
            fail_on_invalid=False
    ):
        """
        Returns the evaluation example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """

        if data_dir is None:
            data_dir = ""

        with open(
                os.path.join(data_dir, self.dev_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = [json.loads(line) for line in reader]

        splitter = char_splitter.CharacterSplitter()
        entries = [preproc.create_entry_from_json(json_elem,
                                                  splitter,
                                                  max_passages=max_passages,
                                                  max_position=max_position,
                                                  fail_on_invalid=fail_on_invalid) for json_elem in input_data]
        return self._create_examples(entries, "dev")

    def _create_examples(self, input_data, set_type: str):
        # https://github.com/google-research/language/blob/bcc90d312aa355f507ed128e39b7f6ea4b709537/language/canine/tydiqa/data.py#L281
        # Converts a list of data entries to TyDiExample
        is_training = set_type == "train"
        examples = []

        for entry in tqdm(input_data):
            if is_training:
                answer = data.make_tydi_answer(entry["contexts"], entry["answer"])
                start_byte_offset = answer.offset
                end_byte_offset = answer.offset + data.byte_len(answer.text)

            else:
                answer = None
                start_byte_offset = None
                end_byte_offset = None

            example = TyDiExample(
                example_id=int(entry["id"]),
                language=entry["language"],
                question=entry["question"]["input_text"],
                contexts=entry["contexts"],
                plaintext=entry["plaintext"],
                context_to_plaintext_offset=entry["context_to_plaintext_offset"],
                answer=answer,
                start_byte_offset=start_byte_offset,
                end_byte_offset=end_byte_offset
            )
            examples.append(example)

        return examples


class TyDiExample(object):
    """
    A single training/test example for the Squad dataset, as loaded from disk.

    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
            self,
            example_id: int,
            language: str,
            question: Text,
            contexts: Text,
            plaintext: Text,
            context_to_plaintext_offset: Sequence[int],
            answer: Optional[data.Answer] = None,
            start_byte_offset: Optional[int] = None,
            end_byte_offset: Optional[int] = None):
        self.example_id = example_id
        self.language = language

        # `question` and `contexts` are the preprocessed question and plaintext
        # with special tokens appended by `create_entry_from_json`. All candidate
        # contexts have been concatenated in `contexts`.
        self.question = question
        self.contexts = contexts

        # `plaintext` is the original article plaintext from the corpus.
        self.plaintext = plaintext

        self.answer = answer
        # `context_to_plaintext_offset` gives a mapping from byte indices in
        # `context` to byte indices in `plaintext`.
        self.context_to_plaintext_offset = context_to_plaintext_offset

        # The following attributes will be `None` for non-training examples.
        # For training, the *offset attributes are derived from the TyDi entry's
        # `start_offset` attribute via `make_tydi_answer`. They are offsets within
        # the original plaintext.
        if answer is not None:
            self.answer = answer
            self.start_byte_offset = start_byte_offset
            self.end_byte_offset = end_byte_offset
            self.is_impossible = self.answer.type == data.AnswerType.UNKNOWN
        else:
            self.answer = None
            self.start_byte_offset = None
            self.end_byte_offset = None
            self.is_impossible = None


class TydiFeatures(object):
    """
    Single squad example features to be fed to a model.
    Those features are model-specific and can be crafted from :class:`~transformers.data.processors.squad.SquadExample`
    using the :method:`~transformers.data.processors.squad.squad_convert_examples_to_features` method.

    Adapted to TyDiQA InputFeatures
    https://github.com/google-research/language/blob/13fd14e1b285002412252097586f8fe405ba8a24/language/canine/tydiqa/preproc.py#L280

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        context_len: The length of the context
        token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
            If a token does not have their maximum context in this feature object, it means that another feature object
            has more information related to that token and should be prioritized over this feature for that token.
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
        start_position: start of the answer token index
        end_position: end of the answer token index
    """

    def __init__(
            self,
            unique_id: int,
            example_index: int,
            language: str,
            doc_span_index: int,
            wp_start_offset: Sequence[int],
            wp_end_offset: Sequence[int],
            input_ids: Sequence[int],
            input_mask: Sequence[int],
            # context_len: int,
            segment_ids: Sequence[int],
            start_position: Optional[int] = None,
            end_position: Optional[int] = None,
            answer_text: Text = "",
            answer_type: int = 3,  # minimal is default
    ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.language = language

        self.doc_span_index = doc_span_index
        self.wp_start_offset = wp_start_offset
        self.wp_end_offset = wp_end_offset

        self.input_ids = input_ids
        self.input_mask = input_mask
        # self.context_len = context_len  # padded to max length in TyDiQA for all samples
        self.segment_ids = segment_ids  # equivalent to type_token_ids
        # self.cls_index = cls_index
        # self.p_mask = p_mask

        # self.segment_ids = segment_ids
        self.start_position = start_position  # Index of wordpiece span start.
        self.end_position = end_position  # Index of wordpiece span end (inclusive).
        self.answer_text = answer_text
        self.answer_type = answer_type

        # self.paragraph_len = paragraph_len
        # self.token_is_max_context = token_is_max_context
        # self.tokens = tokens
        # self.token_to_orig_map = token_to_orig_map


class TyDiResult(object):
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.

    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(self, unique_id, start_logits, end_logits, answer_type_logits,
                 start_top_index=None, end_top_index=None, answer_type_index=None, cls_logits=None):
        self.unique_id = unique_id
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.answer_type_logits = answer_type_logits

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.answer_type_index = answer_type_index
            self.cls_logits = cls_logits
