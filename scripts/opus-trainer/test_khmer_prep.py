"""Khmer lines in the training-text cleaner.

Khmer puts spaces between phrases, not words, and prints figures in Khmer
digits as often as in Western ones. These tests pin the cleaner against
hand-written rows.
"""

from __future__ import annotations

import pytest

from pair_spec import ascii_digits, numbers_agree


def test_khmer_digits_match_western_digits():
    assert ascii_digits("៛១២៥០០") == "៛12500"
    assert numbers_agree("តម្លៃ ១២៥០០ ៛", "Price 12500 KHR")
    assert numbers_agree("ម៉ោង ៨:០០ ដល់ ១៧:៣០", "Open 8:00 to 17:30")
    assert not numbers_agree("តម្លៃ ១២៥០ ៛", "Price 12500 KHR")


def test_number_fidelity_reads_khmer_digits():
    from number_fidelity import numeral_tokens

    assert numeral_tokens("តម្លៃ ១២,៥០០ ៛") == numeral_tokens("Price 12,500 KHR") == ["12500"]


def test_prep_counts_khmer_words_from_characters():
    pytest.importorskip("fasttext")
    import prep_data

    km_line = "សូមដោះស្បែកជើងមុនពេលចូលវិហារ"  # 28 characters, no spaces
    assert prep_data.word_count(km_line) == round(28 / prep_data.CHARS_PER_WORD)
    assert prep_data.word_count("Please take off your shoes") == 5


def test_prep_letter_ratio_accepts_khmer_marks_and_spaced_punctuation():
    pytest.importorskip("fasttext")
    import prep_data

    assert prep_data.alpha_ratio_ok("ពាក្យសម្ងាត់ គឺ « មុយរាល » ។", 0.6, 0.4)
    assert prep_data.alpha_ratio_ok("ដាក់​ស្នើ", 0.6, 0.4)


def test_prep_khmer_digits_match_ascii():
    pytest.importorskip("fasttext")
    import prep_data

    assert prep_data.num_mismatch_ok("we have 20 bacteria", "យើងមាន ២០ បាក់តេរី", 1.0)


def test_prep_numbers_touching_khmer_letters():
    pytest.importorskip("fasttext")
    import prep_data

    ok = prep_data.num_mismatch_ok
    # Khmer has no spaces: a figure sits against the letters around it
    assert ok("aged 51, accused of misconduct", "អាយុ 51ឆ្នាំត្រូវបានចោទ", 1.0)
    assert ok("at least 60 were killed", "មនុស្សយ៉ាងតិច60នាក់បានស្លាប់", 1.0)
    assert ok("on 86 minutes", "នៅ86នាទី", 1.0)
    # English ordinals ("20th") are read too when the pair has Khmer
    assert ok("the 20th century", "សតវត្សទី20", 1.0)
    assert ok("the third level, 3rd", "ទី៣", 1.0)
    # a real mismatch still fails
    assert not ok("aged 51", "អាយុ 52ឆ្នាំ", 1.0)
    assert not ok("60 were killed", "មនុស្សបានស្លាប់", 1.0)
    # a figure glued to a Latin letter, or inside a longer figure, is not split out
    assert prep_data._nums_unspaced("ម៉ូដែល A5") == set()
    assert prep_data._nums_unspaced("ចំនួន 5,000 នាក់") == {"5*000"}


def test_prep_number_check_unchanged_without_khmer():
    pytest.importorskip("fasttext")
    import prep_data

    # no Khmer in the pair: the original \b behaviour, so other languages are identical
    def old(s, t, ratio):
        a, b = prep_data._nums(s), prep_data._nums(t)
        return (not a and not b) or (len(a & b) + 1) / (len(a ^ b) + 1) >= ratio

    pairs = [
        ("the 20th century", "ს20სუკუნე"),          # Georgian letters touching a figure
        ("we have 20 bacteria", "wir haben 21 Bakterien"),
        ("we have 20 bacteria", "wir haben 20 Bakterien"),
        ("iPhone12 costs 5,000", "iPhone12 kostet 5000"),
        ("aged 51", "ასაკი 51წელი"),
        ("room 101", "otaq 101-ci"),
    ]
    for s, t in pairs:
        assert prep_data.num_mismatch_ok(s, t, 1.0) == old(s, t, 1.0), (s, t)


def test_ui_word_check_accepts_khmer():
    import registers

    assert registers.has_word("ដាក់ស្នើ")
    assert registers.ui_strip("Submit", "ដាក់ស្នើ") == ("Submit", "ដាក់ស្នើ")
    assert not registers.has_word("%02d:%02d")


def test_prep_long_khmer_phrase_is_not_a_long_word():
    pytest.importorskip("fasttext")
    import prep_data

    phrase = "ក" * 160  # one space-free phrase longer than --max-word-len
    assert prep_data.max_word_ok("Hello there", phrase, 150)
    assert not prep_data.max_word_ok("x" * 160, "សួស្តី", 150)  # Latin still guarded
