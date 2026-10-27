"""Tests for domain enumerations."""

from enum import StrEnum

import pytest

from tubitak_rag.domain.enums import AnswerStatus, DocType, QuestionType, SectionKind, SourceKind


@pytest.mark.parametrize("enum", [SourceKind, DocType, SectionKind, QuestionType, AnswerStatus])
def test_enums_are_str_enums(enum: type[StrEnum]) -> None:
    assert issubclass(enum, StrEnum)


def test_enum_values_match_spec() -> None:
    assert [k.value for k in SourceKind] == ["pdf", "html", "gib_json", "xlsx"]
    assert [d.value for d in DocType] == [
        "kanun",
        "yonetmelik",
        "uygulama_esaslari",
        "yonerge",
        "esaslar",
        "program_sayfasi",
        "cagri",
        "kilavuz",
        "tablo",
        "diger",
    ]
    assert [s.value for s in SectionKind] == ["madde", "bolum", "sss", "paragraf", "tablo"]
    assert [q.value for q in QuestionType] == [
        "tek_belge",
        "cok_belge",
        "cevapsiz",
        "kapsam_disi",
        "surum",
        "belirsiz",
    ]
    assert [a.value for a in AnswerStatus] == [
        "cevaplandi",
        "bilgi_bulunamadi",
        "kapsam_disi",
        "atif_reddi",
    ]


def test_str_enum_compares_to_plain_string() -> None:
    assert DocType.KANUN == "kanun"
    assert f"{AnswerStatus.KAPSAM_DISI}" == "kapsam_disi"
