from transformers import (
    PreTrainedTokenizerBase,
    BertTokenizer,
    AutoTokenizer,
    RobertaTokenizer
)
import collections
import unicodedata
import six
try:
    from bert.bpe_helper import BPE
except:
    from processors.bert.bpe_helper import BPE

import sentencepiece as spm
import numpy as np
from datasets import load_metric


BOUNDARY_LABELS = {
    '[PAD]': 0,
    'B-SUB': 1,
    'I-SUB': 2,
}

SUBWORD_ID2LABEL = ['[PAD]', 'B-SUB', 'I-SUB']
metric = load_metric("seqeval")


def compute_subword_pred_metrics(labels, logits, ignore_index=0):
    logits = np.array(logits)
    logits[:, :, 0] = - np.inf
    predictions = np.argmax(logits, axis=-1)
    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[SUBWORD_ID2LABEL[l] for l in label if l != ignore_index] for label in labels]
    true_predictions = [
        [SUBWORD_ID2LABEL[p] for (p, l) in zip(prediction, label) if l != ignore_index]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def make_subword_boundary_labels(text, tokens):
    tok_idx = 0
    max_idx = len(tokens) - 1
    labels = []
    for c in text:
        if _is_whitespace(c):
            labels.append('[PAD]')
        elif c in tokens[min(tok_idx, max_idx)]:
            labels.append('B-SUB')
            tok_idx += 1
        else:
            labels.append('I-SUB')
    return labels


class PretrainedTokenizer:
    tokenizer_class = None
    checkpoint_name = None

    def __init__(self):
        self.tokenizer = self.load_tokenizer()

    def load_tokenizer(self):
        tokenizer = self.tokenizer_class.from_pretrained(self.checkpoint_name)
        return tokenizer

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)
        return tokens


class KoreanTokenizer:
    def __init__(self):
        from konlpy.tag import Mecab
        self.tokenizer = Mecab()

    def tokenize(self, text):
        tokens = self.tokenizer.morphs(text)
        return tokens


class JapaneseTokenizer:
    def __init__(self):
        import MeCab
        self.tokenizer = MeCab.Tagger("-Owakati")

    def tokenize(self, text):
        tokens = self.tokenizer.parse(text).split()
        return tokens


class EnglishTokenizer:
    def __init__(self):
        pass

    def tokenize(self, text):
        return text.split()


class FinnishTokenizer(PretrainedTokenizer):
    tokenizer_class = RobertaTokenizer
    checkpoint_name = 'Finnish-NLP/roberta-large-finnish'


class ArabicTokenizer(PretrainedTokenizer):
    tokenizer_class = AutoTokenizer
    checkpoint_name = 'asafaya/bert-base-arabic'


class BengaliTokenizer(PretrainedTokenizer):
    tokenizer_class = AutoTokenizer
    checkpoint_name = "sagorsarker/bangla-bert-base"


class IndonesianTokenizer(PretrainedTokenizer):
    tokenizer_class = BertTokenizer
    checkpoint_name = 'cahya/bert-base-indonesian-522M'


class SwahiliTokenizer(PretrainedTokenizer):
    tokenizer_class = AutoTokenizer
    checkpoint_name = "flax-community/roberta-swahili"


class TeluguTokenizer(PretrainedTokenizer):
    tokenizer_class = AutoTokenizer
    checkpoint_name = "ai4bharat/indic-bert"


class ThaiTokenizer(PretrainedTokenizer):
    tokenizer_class = AutoTokenizer
    checkpoint_name = "airesearch/wangchanberta-base-att-spm-uncased"


# class ThaiTokenizer:
#     def __init__(self):
#         from pythainlp import sent_tokenize
#         self.sent_tokenize = sent_tokenize
#         self.pre_tokenizer = ThaiPreTokenizer(
#             vocab_file='bert/th.wiki.bpe.op25000.vocab',
#             spm_file='bert/th.wiki.bpe.op25000.model',
#         )
#
#     def tokenize(self, text):
#         sentences = self.sent_tokenize(text)
#         split_words = ' '.join(self.tokenizer.tokenize(' '.join(sentences)))
#         tokens = split_words.split()
#         return tokens


class RussianTokenizer:
    def __init__(self):
        from spacy.lang.ru import Russian
        self.tokenizer = Russian()

    def tokenize(self, text):
        doc = self.tokenizer(text)
        tokens = [token.text for token in doc]
        return tokens


def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicodedata):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
  vocab = collections.OrderedDict()
  index = 0
  with open(vocab_file, "r") as reader:
    while True:
      token = reader.readline()
      if token.split():
          token = token.split()[0] # to support SentencePiece vocab file
      token = convert_to_unicode(token)
      if not token:
        break
      token = token.strip()
      vocab[token] = index
      index += 1
  return vocab


def convert_by_vocab(vocab, items):
  output = []
  for item in items:
    output.append(vocab[item])
  return output


class ThaiPreTokenizer(object):
  """Tokenizes Thai texts."""

  def __init__(self, vocab_file, spm_file):
    self.vocab = load_vocab(vocab_file)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}

    self.bpe = BPE(vocab_file)
    self.s = spm.SentencePieceProcessor()
    self.s.Load(spm_file)

  def tokenize(self, text):
    bpe_tokens = self.bpe.encode(text).split(' ')
    spm_tokens = self.s.EncodeAsPieces(text)

    tokens = bpe_tokens if len(bpe_tokens) < len(spm_tokens) else spm_tokens

    split_tokens = []

    for token in tokens:
      new_token = token

      if token.startswith('_') and not token in self.vocab:
        split_tokens.append('_')
        new_token = token[1:]

      if not new_token in self.vocab:
        split_tokens.append('<unk>')
      else:
        split_tokens.append(new_token)

    return split_tokens

  def convert_tokens_to_ids(self, tokens):
    return convert_by_vocab(self.vocab, tokens)

  def convert_ids_to_tokens(self, ids):
    return convert_by_vocab(self.inv_vocab, ids)

