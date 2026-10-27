"""Tests for answer models returned by the pipeline."""

import pytest
from pydantic import ValidationError

from tubitak_rag.domain.answer import Answer, Citation, SourceView
from tubitak_rag.domain.enums import AnswerStatus


def _answer(**overrides: object) -> Answer:
    fields: dict[str, object] = {
        "soru": "1507 destek oranı nedir?",
        "durum": "cevaplandi",
        "cevaplanabilir": True,
        "cevap": "En fazla %75 [1].",
        "atiflar": [
            Citation(
                no=1,
                chunk_id="c1",
                doc_id="mevzuat-031",
                doc_title="1507 UE",
                konum="Madde 7 (2)",
                kaynak_url="https://x",
            )
        ],
        "kaynaklar": [
            SourceView(
                no=1,
                chunk_id="c1",
                doc_title="1507 UE",
                konum="Madde 7 (2)",
                kaynak_url="https://x",
                text="…",
                score=1.2,
            )
        ],
        "uyari": "Bu cevap bilgilendirme amaçlıdır.",
        "gecikme_ms": {"total": 12.0},
        "llm_cagri_sayisi": 1,
        "gate_key": "rerank",
        "gate_score": 1.2,
    }
    fields.update(overrides)
    return Answer.model_validate(fields)


def test_answer_parses_status_enum() -> None:
    assert _answer().durum is AnswerStatus.CEVAPLANDI


def test_answer_rejects_unknown_status() -> None:
    with pytest.raises(ValidationError):
        _answer(durum="belki")


def test_answer_json_uses_turkish_field_names() -> None:
    dumped = _answer().model_dump(mode="json")
    assert {"cevaplanabilir", "cevap", "atiflar", "kaynaklar"} <= set(dumped)
    assert dumped["durum"] == "cevaplandi"
