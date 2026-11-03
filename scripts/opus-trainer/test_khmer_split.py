"""Khmer sentence stops in the corpus splitter.

Khmer ends a sentence with the khan and a passage with the bariyoosan.
Spaces separate phrases, so a space alone must not split.
"""

from __future__ import annotations

from split_sentences import split


def test_khan_splits_khmer_sentences():
    assert split("ហាងបិទថ្ងៃអាទិត្យ។ សូមអភ័យទោស។") == ["ហាងបិទថ្ងៃអាទិត្យ។", "សូមអភ័យទោស។"]


def test_bariyoosan_ends_a_sentence():
    assert split("ចប់ហើយ៕ សូមអរគុណ។") == ["ចប់ហើយ៕", "សូមអរគុណ។"]


def test_khmer_phrase_space_is_not_a_boundary():
    one = "សូមដោះស្បែកជើង មុនពេលចូលវិហារ។"
    assert split(one) == [one]


def test_sentence_starting_with_khmer_digit_splits():
    # Any Khmer character, digits included, can open the next sentence.
    assert split("តម្លៃឡើងថ្លៃ។ ១០០ រៀលក្នុងមួយគីឡូ។") == ["តម្លៃឡើងថ្លៃ។", "១០០ រៀលក្នុងមួយគីឡូ។"]
