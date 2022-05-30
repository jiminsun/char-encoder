# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Very heavily inspired by the official evaluation script for SQuAD version 2.0 which was modified by XLNet authors to
update `find_best_threshold` scripts for SQuAD V2.0

In addition to basic functionality, we also compute additional statistics and plot precision-recall curves if an
additional na_prob.json file is provided. This file is expected to map question ID's to the model's predicted
probability that a question is unanswerable.
"""


import collections
import json
import math
import re
import string
from tqdm import tqdm

from transformers.models.bert import BasicTokenizer
from transformers.utils import logging

from multiprocessing import Pool, cpu_count
from functools import partial

logger = logging.get_logger(__name__)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_raw_scores(examples, preds):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    exact_scores = {}
    f1_scores = {}

    for example in examples:
        qas_id = example.qas_id
        gold_answers = [answer["text"] for answer in example.answers if normalize_answer(answer["text"])]

        if not gold_answers:
            # For unanswerable questions, only correct answer is empty string
            gold_answers = [""]

        if qas_id not in preds:
            logger.info(f"Missing prediction for {qas_id}")
            continue

        prediction = preds[qas_id]
        exact_scores[qas_id] = max(compute_exact(a, prediction) for a in gold_answers)
        f1_scores[qas_id] = max(compute_f1(a, prediction) for a in gold_answers)

    return exact_scores, f1_scores


def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores.values()) / total),
                ("f1", 100.0 * sum(f1_scores.values()) / total),
                ("total", total),
            ]
        )
    else:
        total = len(qid_list)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
                ("f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
                ("total", total),
            ]
        )


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval[f"{prefix}_{k}"] = new_eval[k]


def find_best_thresh_v2(preds, scores, na_probs, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for i, qid in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]

    has_ans_score, has_ans_cnt = 0, 0
    for qid in qid_list:
        if not qid_to_has_ans[qid]:
            continue
        has_ans_cnt += 1

        if qid not in scores:
            continue
        has_ans_score += scores[qid]

    return 100.0 * best_score / len(scores), best_thresh, 1.0 * has_ans_score / has_ans_cnt


def find_all_best_thresh_v2(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    best_exact, exact_thresh, has_ans_exact = find_best_thresh_v2(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh, has_ans_f1 = find_best_thresh_v2(preds, f1_raw, na_probs, qid_to_has_ans)
    main_eval["best_exact"] = best_exact
    main_eval["best_exact_thresh"] = exact_thresh
    main_eval["best_f1"] = best_f1
    main_eval["best_f1_thresh"] = f1_thresh
    main_eval["has_ans_exact"] = has_ans_exact
    main_eval["has_ans_f1"] = has_ans_f1


def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for _, qid in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]
    return 100.0 * best_score / len(scores), best_thresh


def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)

    main_eval["best_exact"] = best_exact
    main_eval["best_exact_thresh"] = exact_thresh
    main_eval["best_f1"] = best_f1
    main_eval["best_f1_thresh"] = f1_thresh


def squad_evaluate(examples, preds, no_answer_probs=None, no_answer_probability_threshold=1.0):
    qas_id_to_has_answer = {example.qas_id: bool(example.answers) for example in examples}
    has_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if has_answer]
    no_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if not has_answer]

    if no_answer_probs is None:
        no_answer_probs = {k: 0.0 for k in preds}

    exact, f1 = get_raw_scores(examples, preds)

    exact_threshold = apply_no_ans_threshold(
        exact, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold
    )
    f1_threshold = apply_no_ans_threshold(f1, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold)

    evaluation = make_eval_dict(exact_threshold, f1_threshold)

    if has_answer_qids:
        has_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=has_answer_qids)
        merge_eval(evaluation, has_ans_eval, "HasAns")

    if no_answer_qids:
        no_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=no_answer_qids)
        merge_eval(evaluation, no_ans_eval, "NoAns")

    if no_answer_probs:
        find_all_best_thresh(evaluation, preds, exact, f1, no_answer_probs, qas_id_to_has_answer)

    return evaluation


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(f"Unable to find text: '{pred_text}' in '{orig_text}'")
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info(f"Length not equal after stripping spaces: '{orig_ns_text}' vs '{tok_ns_text}'")
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position : (orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits[1:], 1), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


class ScoreSummary(object):

  def __init__(self):
    self.predicted_label = None
    self.minimal_span_score = None
    self.cls_token_score = None
    self.answer_type_logits = None

Span = collections.namedtuple("Span", ["start_byte_offset", "end_byte_offset"])


class EvalExample(object):
  """Eval data available for a single example."""

  def __init__(self, example_id, candidates):
    self.example_id = example_id
    self.candidates = candidates
    self.results = {}
    self.features = {}


def compute_predictions_logits(
    eval_example,
    n_best_size,
    max_answer_length,
):
    # features = eval_example.features

    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    # score_null = 1000000  # large and positive
    # min_null_feature_index = 0  # the paragraph slice with min null score
    # null_start_logit = 0  # the start logit at the slice with min null score
    # null_end_logit = 0  # the end logit at the slice with min null score
    if not eval_example.results:
        return None
    if len(eval_example.features) != len(eval_example.results):
        logger.warning(
            "ERROR: len(features)=%s, but len(results)=%d for eval_example %s",
            len(eval_example.features), len(eval_example.results),
            eval_example.example_id)
        return None

    for (unique_id, result) in eval_example.results.items():
        if unique_id not in eval_example.features:
            logger.warning("No feature found with unique_id: %s", unique_id)
            return None

        wp_start_offset = (
            eval_example.features[unique_id].wp_start_offset)
        wp_end_offset = (
            eval_example.features[unique_id].wp_end_offset)
        language = (
            eval_example.features[unique_id].language
        )

        start_indexes = _get_best_indexes(result.start_logits, n_best_size)
        end_indexes = _get_best_indexes(result.end_logits, n_best_size)
        # if we could have irrelevant answers, get the min score of irrelevant
        # https://github.com/google-research/language/blob/bcc90d312aa355f507ed128e39b7f6ea4b709537/language/canine/tydiqa/postproc.py#L49

        cls_token_score = result.start_logits[0] + result.end_logits[0]

        for start_index in start_indexes:
            for end_index in end_indexes:
                # We could hypothetically create invalid predictions, e.g., predict
                # that the start of the span is in the question. We throw out all
                # invalid predictions.
                if end_index < start_index:
                    continue
                if wp_start_offset[start_index] == -1:
                    continue
                if wp_end_offset[end_index] == -1:
                    continue
                length = end_index - start_index + 1
                if length > max_answer_length:
                    continue
                summary = ScoreSummary()
                summary.minimal_span_score = (
                        result.start_logits[start_index] +
                        result.end_logits[end_index]
                )
                summary.cls_token_score = cls_token_score
                summary.answer_type_logits = result.answer_type_logits

                start_offset = wp_start_offset[start_index]
                end_offset = wp_end_offset[end_index] + 1
                score = summary.minimal_span_score - summary.cls_token_score

                prelim_predictions.append(
                    (score, summary, language, start_offset, end_offset)
                )
    if not prelim_predictions:
        logger.warning("No predictions for eval_example %s",
                        eval_example.example_id)
        return None

    score, summary, language_name, start_span, end_span = max(
        prelim_predictions, key=lambda x: x[0])

    minimal_span = Span(start_span, end_span)
    passage_span_index = 0
    for c_ind, c in enumerate(eval_example.candidates):
        start = minimal_span.start_byte_offset
        end = minimal_span.end_byte_offset
        if c['plaintext_start_byte'] <= start and c['plaintext_end_byte'] >= end:
            passage_span_index = c_ind
            break
    else:
        logger.warning("No passage predicted for eval_example %s. Choosing first.",
                        eval_example.example_id)
    summary.predicted_label = {
        "example_id": eval_example.example_id,
        "language": language_name,
        "passage_answer_index": passage_span_index,
        "passage_answer_score": score,
        "minimal_answer": {
            "start_byte_offset": minimal_span.start_byte_offset,
            "end_byte_offset": minimal_span.end_byte_offset
        },
        "minimal_answer_score": score,
        "yes_no_answer": "NONE"
    }
    return summary


def compute_pred_dict(
    candidates_dict,
    dev_features,
    raw_results,
    candidate_beam,
    max_answer_length,
    threads,
):
    # https://github.com/google-research-datasets/tydiqa/blob/43cde6d598c1cf88c1a8b9ed32e89263ffb5e03b/baseline/postproc.py#L183
    logger.info("Post-processing predictions started.")

    raw_results_by_id = [(int(res.unique_id + 1), res) for res in raw_results]
    all_candidates = candidates_dict.items()
    example_ids = [int(k) for k, _ in all_candidates]
    examples_by_id = list(zip(example_ids, all_candidates))
    if not examples_by_id:
        raise ValueError("No examples candidates found.")
    feature_ids = []
    features = []
    for f in dev_features:
        feature_ids.append(f.unique_id + 1)
        features.append(f)
    features_by_id = list(zip(feature_ids, features))

    # Join examples with features and raw results.
    eval_examples = []
    merged = sorted(
        examples_by_id + raw_results_by_id + features_by_id,
        key=lambda pair: pair[0])

    # Error counters
    num_failed_matches = 0

    ex_count = 0
    feature_count = 0
    result_count = 0

    for feature_unique_id, datum in merged:
        # if from `examples_by_id`
        if isinstance(datum, tuple):
            ex_count += 1
            eval_examples.append(
                EvalExample(example_id=datum[0], candidates=datum[1])
            )
        # if from `features_by_id`
        elif "wp_start_offset" in datum.__dict__:
            feature_count += 1
            # Join with the example that we just appended above, by
            # adding to the `EvalExample`'s `features` dict.
            if not eval_examples:
                logger.warning("Expected to already have example for this example id. "
                                "Dataset / predictions mismatch?")
                num_failed_matches += 1
                continue
            eval_examples[-1].features[feature_unique_id] = datum
        # if from `raw_results_by_id`
        else:
            result_count += 1
            # Join with the example that we just appended above, by
            # adding to the `EvalExample`'s `results` dict.
            if not eval_examples:
                logger.warning("Expected to already have example for this example id. "
                                "Dataset / predictions mismatch?")
                num_failed_matches += 1
                continue
            eval_examples[-1].results[feature_unique_id] = datum

    logger.info("  Num candidate examples found: %d", ex_count)
    logger.info("  Num candidate features found: %d", feature_count)
    logger.info("  Num results found: %d", result_count)
    logger.info("  len(merged): %d", len(merged))
    if num_failed_matches > 0:
        logger.warning("  Num failed matches: %d", num_failed_matches)

    # Construct prediction objects.

    threads = min(threads, cpu_count())
    with Pool(threads) as p:
        compute_predictions_ = partial(
            compute_predictions_logits,
            n_best_size=candidate_beam,
            max_answer_length=max_answer_length
        )
        summaries = list(
            tqdm(
                p.imap(compute_predictions_, eval_examples, chunksize=32),
                total=len(eval_examples),
                desc="computing predictions"
            )
        )

    # print(summaries)
    tydi_pred_dict = {s.predicted_label['example_id']: s.predicted_label
                      for s in summaries if s is not None}
    return tydi_pred_dict