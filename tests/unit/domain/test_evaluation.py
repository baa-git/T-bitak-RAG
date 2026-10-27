"""Tests for evaluation models (test set rows, per-question results, run metrics)."""

import json

import pytest
from pydantic import ValidationError

from tubitak_rag.domain.enums import QuestionType
from tubitak_rag.domain.evaluation import EvidenceQuote, QuestionResult, RunMetrics, TestQuestion

EXAMPLE_ROW = (
    '{"id":"q001","soru":"1507 programında destek oranı en fazla yüzde kaçtır?",'
    '"tur":"tek_belge","zorluk":"kolay","beklenen_cevap":"Her dönem için en fazla %75.",'
    '"kanit_alintilari":[{"doc_id":"mevzuat-031",'
    '"alinti":"destek oranı her dönem için en fazla %75 olarak belirlenmiştir"}],'
    '"beklenen_belgeler":["mevzuat-031"],"beklenen_sayilar":["75"],"program_kodu":["1507"],'
    '"dogrulandi":false,"review_notes":"","uretim":"llm"}'
)


def _minimal_question(**overrides: object) -> TestQuestion:
    fields: dict[str, object] = {
        "id": "q",
        "soru": "?",
        "tur": "cevapsiz",
        "zorluk": "kolay",
        "beklenen_cevap": "bilgi bulunamadı",
        "kanit_alintilari": [],
        "beklenen_belgeler": [],
    }
    fields.update(overrides)
    return TestQuestion.model_validate(fields)


def test_test_question_parses_example_row() -> None:
    q = TestQuestion.model_validate_json(EXAMPLE_ROW)
    assert q.tur is QuestionType.TEK_BELGE
    assert q.kanit_alintilari == [
        EvidenceQuote(
            doc_id="mevzuat-031",
            alinti="destek oranı her dönem için en fazla %75 olarak belirlenmiştir",
        )
    ]
    assert json.loads(q.model_dump_json()) == json.loads(EXAMPLE_ROW)


def test_test_question_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        _minimal_question(tur="bilinmeyen")


def test_test_question_rejects_unknown_difficulty() -> None:
    with pytest.raises(ValidationError):
        _minimal_question(zorluk="çok zor")


def test_test_question_defaults() -> None:
    q = _minimal_question()
    assert q.dogrulandi is False
    assert (q.beklenen_sayilar, q.program_kodu, q.review_notes, q.uretim) == ([], [], "", "manuel")


def test_question_result_round_trips() -> None:
    result = QuestionResult(
        id="q001",
        tur=QuestionType.TEK_BELGE,
        soru="?",
        retrieved=[{"chunk_id": "c1", "doc_id": "d", "rank": 1, "score": 0.5, "scores": {}}],
        evidence_hits=[1, None],
        first_hit_rank=1,
        doc_hit_at_5=True,
        doc_hit_at_10=True,
        gate_key="dense",
        gate_score=0.5,
        answer={"durum": "cevaplandi", "cevap": "x", "atiflar": ["c1"]},
        judge={"dogru": True, "sadik": None, "atif_dogru": True, "gerekce": ""},
        rule_check={"sayilar_bulundu": None},
        latency_ms={"total": 1.0},
        llm_calls=1,
    )
    assert QuestionResult.model_validate_json(result.model_dump_json()) == result


def test_run_metrics_allows_missing_reranker() -> None:
    metrics = RunMetrics(
        run_id="v0_20261015-120000",
        experiment="v0",
        started_at="2026-10-15T12:00:00",
        finished_at="2026-10-15T12:30:00",
        wall_clock_s=1800.0,
        judge_model="qwen3:4b",
        llm_model="qwen3:4b",
        embedder="google/embeddinggemma-2",
        chunker="fixed",
        retriever="dense",
        reranker=None,
        counts={"questions": 100},
        retrieval={"recall_at_5": 0.5},
        generation={},
        latency_ms={},
        by_type={},
        stage_times_s={},
    )
    assert metrics.reranker is None
