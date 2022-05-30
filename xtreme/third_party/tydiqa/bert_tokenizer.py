from transformers import BertTokenizer
from transformers.models.bert.tokenization_bert import *
import tydiqa.data as data
import tydiqa.tydi_tokenization_interface as tydi_tokenization_interface
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Text, Tuple


class BertTyDiTokenizer(tydi_tokenization_interface.TokenizerWithOffsets):
    def __init__(self, model_path, tydi_vocab_file_path):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        print(f'=== Loaded vocab -- size {len(self.tokenizer)}')
        special_tokens_dict = {'additional_special_tokens': self.load_tydi_vocab(tydi_vocab_file_path)}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        print(num_added_toks)
        print(f'=== Fixed vocab -- size {len(self.tokenizer)}')
        assert len(self.tokenizer.tokenize('[YES]')) == 1
        self.token2idx = OrderedDict(self.tokenizer.vocab)
        self.idx2token = list(self.token2idx.keys())

    def load_tydi_vocab(self, fpath):
        tydi_vocab = [t.rstrip() for t in open(fpath, 'r')]
        special_tokens = ['\n'] + [t for t in tydi_vocab if t not in self.tokenizer.vocab]
        return special_tokens

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def tokenize_with_offsets(self, text):
        result = tydi_tokenization_interface.TokenizedIdsWithOffsets([], [], [], {})
        byte_index = 0
        subtokens = self.tokenizer.tokenize(text)
        subtoken_ids = self.tokenizer.encode(text, add_special_tokens=False)
        subtoken_lengths = [data.byte_len(t) for t in subtokens]
        for subtoken_index, subtoken in enumerate(subtokens):
            result.subtokens.append(subtoken_ids[subtoken_index])
            result.start_bytes.append(byte_index)
            for _ in range(data.byte_len(subtoken)):
                result.offsets_to_subtoken[byte_index] = subtoken_index
                byte_index += 1
            result.limit_bytes.append(byte_index - 1)
        return result

    def get_passage_marker(self, i: int) -> Text:
        return f'[Paragraph={i}]'

    def unk_id(self):
        return self.token2idx.get('[UNK]')

    def get_vocab_id(self, key):
        """Gets the vocab id of `key`."""
        return self.token2idx.get(key, self.unk_id())

    def id_to_string(self, i):
        return self.idx2token[i]