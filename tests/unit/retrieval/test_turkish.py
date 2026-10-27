"""Tests for Turkish lower-casing and BM25 tokenisation (FR-R2, FR-R3)."""

import pytest

from tubitak_rag.retrieval.turkish import STOPWORDS_TR, TurkishTokenizer, turkish_lower


def test_turkish_lower_dotted_and_dotless_i() -> None:
    assert turkish_lower("İSTANBUL IĞDIR") == "istanbul ığdır"


def test_turkish_lower_leaves_no_combining_dot() -> None:
    assert turkish_lower("İ") == "i"  # str.lower() alone gives "i̇" (i + U+0307)


def test_tokenizer_keeps_program_codes_and_percent_numbers() -> None:
    assert TurkishTokenizer()("1507'de %75") == ["1507", "75"]


def test_tokenizer_keeps_hyphenated_and_dotted_terms() -> None:
    assert TurkishTokenizer(stemming=False)("AR-GE harcaması 5.1 maddesi") == [
        "ar-ge",
        "harcaması",
        "5.1",
        "maddesi",
    ]


def test_stemming_merges_plural_forms() -> None:
    tokenize = TurkishTokenizer()
    assert tokenize("projelerin") == tokenize("projeler") == tokenize("proje") == ["proje"]


def test_stopwords_removed() -> None:
    assert TurkishTokenizer(stemming=False)("bu proje ve bir destek için") == ["proje", "destek"]


def test_stopword_list_is_lowercase_turkish() -> None:
    assert {"ve", "veya", "ile", "için", "mı", "mü"} <= STOPWORDS_TR
    assert len(STOPWORDS_TR) >= 50
    assert all(word == turkish_lower(word) for word in STOPWORDS_TR)


def test_no_stemming_option() -> None:
    tokenize = TurkishTokenizer(stemming=False)
    assert tokenize.stemming is False
    assert tokenize("Projelerin") == ["projelerin"]


def test_single_letters_dropped_but_single_digits_kept() -> None:
    assert TurkishTokenizer(stemming=False)("a) bendi 3 yıl") == ["bendi", "3", "yıl"]


@pytest.mark.parametrize(("text", "expected"), [("hâlinde", "halinde"), ("Kâr", "kar")])
def test_circumflex_letters_are_folded(text: str, expected: str) -> None:
    assert TurkishTokenizer(stemming=False)(text) == [expected]


def test_numbers_are_not_stemmed() -> None:
    assert TurkishTokenizer()("2026 yılında") == ["2026", "yıl"]


def test_empty_text_has_no_tokens() -> None:
    assert TurkishTokenizer()("  ,.;  ") == []


def test_az_is_kept_to_distinguish_minimum_from_maximum() -> None:
    assert TurkishTokenizer(stemming=False)("en az %20") == ["az", "20"]
