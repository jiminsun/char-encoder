# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Character-level preprocessing for CANINE."""

from typing import Dict, Optional, Text
import processors.tydi_special_codepoints as special_codepoints
import processors.tydi_data as data
import processors.tydi_tokenization_interface as tydi_tokenization_interface


# Padding is always index zero. This means that the NULL character is
# technically not embeddable. This seems fine according to all reasonable
# interpretations of the NULL character as a past-end-of-string marker.
PAD = 0

CLS = 0xE000
SEP = 0xE001
BOS = 0xE002
MASK = 0xE003
RESERVED = 0xE004

# Maps special codepoints to human-readable names.
SPECIAL_CODEPOINTS: Dict[int, Text] = {
    # Special symbols are represented using codepoints values that are valid,
    # but designated as "Private Use", meaning that they will never by assigned
    # characters by the Unicode Consortium, and are thus safe for use here.
    #
    # NOTE: Do *NOT* add any sort of [UNK_CHAR] here. They are explicitly
    # excluded and should fail with a hard error.
    CLS: "[CLS]",
    SEP: "[SEP]",
    BOS: "[BOS]",
    MASK: "[MASK]",
    PAD: "[PAD]",
    RESERVED: "[RESERVED]",
}

# Maps special codepoint human-readable names to their codepoint values.
SPECIAL_CODEPOINTS_BY_NAME: Dict[Text, int] = {
    name: codepoint for codepoint, name in SPECIAL_CODEPOINTS.items()
}



class CharacterSplitter(tydi_tokenization_interface.TokenizerWithOffsets):
  """A character splitter that preserves byte offsets.

  This implements the `TokenizerWithOffsets` interface to demonstrate how to
  retrofit legacy tokenization code with character splitting.
  """

  def __init__(self):
    next_private_use = max(special_codepoints.SPECIAL_CODEPOINTS) + 1

    # Special symbols that should be given pseudo-codepoints from the "private
    # use area", following those in the standard CANINE SPECIAL_CODEPOINTS set.
    tydiqa_symbols = ["[Q]"]

    # Creates a mapping for looking up the IDs of special symbols.
    self._special_codepoints: Dict[Text, int] = {}
    for codepoint, name in special_codepoints.SPECIAL_CODEPOINTS.items():
      self._special_codepoints[name] = codepoint
    for codepoint, name in enumerate(tydiqa_symbols, next_private_use):
      self._special_codepoints[name] = codepoint
    next_private_use += len(tydiqa_symbols)

    self._passage_0_codepoint = next_private_use

    # Creates a mapping for looking up the string forms of special symbol IDs.
    self._special_codepoint_strings: Dict[int, Text] = {
        codepoint: name for name, codepoint in self._special_codepoints.items()
    }

  def tokenize_with_offsets(
      self, text: Text) -> tydi_tokenization_interface.TokenizedIdsWithOffsets:
    result = tydi_tokenization_interface.TokenizedIdsWithOffsets([], [], [], {})
    byte_index = 0
    for char_index, c in enumerate(text):
      result.subtokens.append(ord(c))
      result.start_bytes.append(byte_index)
      for _ in range(data.byte_len(c)):
        result.offsets_to_subtoken[byte_index] = char_index
        byte_index += 1
      result.limit_bytes.append(byte_index - 1)
    return result

  def get_passage_marker(self, i: int) -> Text:
    return chr(self._passage_0_codepoint + i)

  def get_vocab_id(self, key: Text, default: Optional[int] = None) -> int:
    """Gets the vocab id of `key`."""
    if key in self._special_codepoints:
      return self._special_codepoints[key]
    try:
      return ord(key)
    except TypeError:
      raise ValueError(f"invalid vocab key: '{key}'")

  def id_to_string(self, i: int) -> Text:
    if i in self._special_codepoint_strings:
      return self._special_codepoint_strings[i]
    try:
      return chr(i)
    except TypeError:
      raise ValueError(f"invalid id: {i}")