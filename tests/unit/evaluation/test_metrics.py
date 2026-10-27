"""Tests for retrieval, generation and latency metrics (FR-E6, FR-E7, NFR-PERF1)."""

import math

import pytest

from tests.eval_builders import make_metrics, make_question, make_result, refused
from tubitak_rag.domain.enums import QuestionType
from tubitak_rag.domain.evaluation import QuestionResult
from tubitak_rag.evaluation.metrics import (
    by_type,
    by_verified,
    generation_metrics,
    latency_metrics,
    mean_ignoring_nan,
    mrr,
    reached_generation,
    recall_at_k,
    recompute,
    refusal_counts,
    retrieval_metrics,
    run_counts,
)

T = QuestionType


def _eight() -> list:
    """Eight hand-built rows covering every question type and outcome."""
    return [
        # answered correctly, evidence at rank 1
        make_result("q1"),
        # answered, judge says wrong but every expected number present -> still correct
        make_result(
            "q2",
            tur=T.COK_BELGE,
            evidence_hits=[2, 7],
            first_hit_rank=2,
            judge={"dogru": False, "sadik": False, "atif_dogru": False, "gerekce": "Eksik."},
            rule_check={
                "sayilar_bulundu": True,
                "atif_kesinligi": 0.5,
                "programlara_ayrildi": None,
            },
        ),
        # answerable but refused by the gate: false rejection, evidence missed
        refused("q3", T.SURUM, "esik_alti", evidence_hits=[None], first_hit_rank=None),
        # answerable, refused by the number guardrail
        refused("q4", T.TEK_BELGE, "dayanaksiz_sayi", evidence_hits=[6], first_hit_rank=6),
        # unanswerable, correctly refused by the gate
        refused("q5", T.CEVAPSIZ, "esik_alti", evidence_hits=[], first_hit_rank=None),
        # unanswerable but answered (hallucination)
        make_result("q6", tur=T.CEVAPSIZ, evidence_hits=[], first_hit_rank=None),
        # out of scope, rejected by the scope check
        refused(
            "q7",
            T.KAPSAM_DISI,
            "kapsam_disi",
            durum="kapsam_disi",
            evidence_hits=[],
            first_hit_rank=None,
            retrieved=[],
            gate_key=None,
            gate_score=None,
        ),
        # ambiguous question answered with the programs split, numbers stripped
        make_result(
            "q8",
            tur=T.BELIRSIZ,
            answer={
                "durum": "cevaplandi",
                "cevap": "1501'de %60, 1507'de %75 [1].",
                "atiflar": ["c"],
                "red_nedeni": None,
                "sayi_ayiklandi": True,
            },
            rule_check={
                "sayilar_bulundu": None,
                "atif_kesinligi": 1.0,
                "programlara_ayrildi": True,
            },
            judge={"dogru": None, "sadik": None, "atif_dogru": True, "gerekce": "hakem_hatasi"},
        ),
    ]


def test_recall_at_k_partial() -> None:
    assert recall_at_k([1, 4, 9], 5) == pytest.approx(2 / 3, abs=1e-3)
    assert math.isnan(recall_at_k([], 5))


def test_mrr() -> None:
    assert mrr([1, 2, None, 4]) == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)
    assert math.isnan(mrr([]))


def test_mean_ignoring_nan() -> None:
    assert mean_ignoring_nan([1.0, math.nan, None, 0.0]) == 0.5
    assert math.isnan(mean_ignoring_nan([]))


def test_unanswerable_excluded() -> None:
    metrics = retrieval_metrics(_eight())
    # answerable rows: q1 [1], q2 [2,7], q3 [None], q4 [6], q8 [1]
    assert metrics["recall_at_5"] == pytest.approx((1 + 0.5 + 0 + 0 + 1) / 5)
    assert metrics["recall_at_10"] == pytest.approx((1 + 1 + 0 + 1 + 1) / 5)
    assert metrics["mrr"] == pytest.approx((1 + 0.5 + 0 + 1 / 6 + 1) / 5)


def test_keys_present() -> None:
    assert set(retrieval_metrics(_eight())) == {
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "doc_hit_at_5",
        "doc_hit_at_10",
    }


def test_metrics_without_data_are_omitted_not_nan() -> None:
    only_unanswerable = [refused("q5", T.CEVAPSIZ, "esik_alti", evidence_hits=[])]
    assert retrieval_metrics(only_unanswerable) == {}
    assert "answer_accuracy" not in generation_metrics(only_unanswerable)


def test_generation_metric_keys() -> None:
    assert set(generation_metrics(_eight())) == {
        "answer_accuracy",
        "faithfulness",
        "citation_precision",
        "correct_rejection_rate",
        "false_rejection_rate",
        "scope_rejection_rate",
        "programs_split_rate",
        "number_rejection_rate",
        "number_strip_rate",
        "error_rate",
    }


def test_answer_accuracy_counts_refusals_as_wrong_and_numbers_rule() -> None:
    metrics = generation_metrics(_eight())
    # answerable: q1 correct, q2 correct (numbers), q3 refused, q4 refused, q8 undecided
    assert metrics["answer_accuracy"] == pytest.approx(2 / 4)
    # answered: q1 True, q2 False, q6 True, q8 None
    assert metrics["faithfulness"] == pytest.approx(2 / 3)
    assert metrics["citation_precision"] == pytest.approx((1 + 0.5 + 1 + 1) / 4)


def test_false_rejection() -> None:
    assert generation_metrics(_eight())["false_rejection_rate"] == pytest.approx(2 / 5)


def test_correct_rejection() -> None:
    metrics = generation_metrics(_eight())
    assert metrics["correct_rejection_rate"] == pytest.approx(2 / 3)  # q5, q7 of q5, q6, q7
    assert metrics["scope_rejection_rate"] == 1.0
    assert metrics["programs_split_rate"] == 1.0


def test_number_guardrail_rates_over_questions_that_reached_the_llm() -> None:
    metrics = generation_metrics(_eight())
    # reached the LLM: q1, q2, q4, q6, q8 (q3, q5 gate; q7 scope)
    assert metrics["number_rejection_rate"] == pytest.approx(1 / 5)
    assert metrics["number_strip_rate"] == pytest.approx(1 / 5)


def test_refusal_counts() -> None:
    assert refusal_counts(_eight()) == {"esik_alti": 2, "dayanaksiz_sayi": 1, "kapsam_disi": 1}


def test_latency_percentiles() -> None:
    rows = [
        make_result(
            f"q{i}", latency_ms={"retrieval": float(i), "llm": 100.0 * i, "total": 10.0 * i}
        )
        for i in range(1, 11)
    ]
    rows.append(
        refused("q11", T.CEVAPSIZ, "esik_alti", latency_ms={"retrieval": 1.0, "total": 1.0})
    )
    metrics = latency_metrics(rows)
    assert metrics["retrieval_p50"] == pytest.approx(5.0)
    assert metrics["total_p95"] == pytest.approx(95.0)
    assert metrics["llm_p50"] == pytest.approx(550.0)  # the refused row did not call the LLM
    assert "rerank_p50" not in metrics


def test_by_type_groups() -> None:
    groups = by_type(_eight(), retrieval_metrics)
    assert list(groups) == [
        "tek_belge",
        "cok_belge",
        "cevapsiz",
        "kapsam_disi",
        "surum",
        "belirsiz",
    ]
    assert groups["tek_belge"]["recall_at_5"] == pytest.approx(0.5)  # q1 hit, q4 rank 6
    assert groups["cevapsiz"] == {}


def test_by_verified() -> None:
    rows = _eight()
    questions = [make_question(r.id, dogrulandi=r.id in {"q1", "q2"}) for r in rows]
    groups = by_verified(rows, questions, retrieval_metrics)
    assert groups["dogrulanmis"]["recall_at_5"] == pytest.approx(0.75)
    assert set(groups) == {"dogrulanmis", "dogrulanmamis"}


def test_llm_error_is_an_error_not_a_refusal() -> None:
    rows = [
        refused("u1", T.CEVAPSIZ, "esik_alti", evidence_hits=[]),
        refused("u2", T.CEVAPSIZ, "llm_hatasi", evidence_hits=[], llm_calls=1),
        make_result("a1"),
        refused("a2", T.TEK_BELGE, "llm_hatasi", llm_calls=1),
    ]
    metrics = generation_metrics(rows)
    assert metrics["correct_rejection_rate"] == 1.0  # u2 is not counted as a refusal
    assert metrics["false_rejection_rate"] == 0.0  # a2 is not counted as a rejection
    assert metrics["answer_accuracy"] == 0.5  # but a2 is a wrong answer
    assert metrics["error_rate"] == 0.5


def test_citation_repair_rate_only_when_reported() -> None:
    assert "citation_repair_rate" not in generation_metrics([make_result("a")])

    def with_repairs(qid: str, count: int) -> QuestionResult:
        answer = make_result(qid).answer | {"atif_onarildi": count}
        return make_result(qid, answer=answer)

    rows = [with_repairs("a", 1), with_repairs("b", 0), with_repairs("c", 2)]
    assert generation_metrics(rows)["citation_repair_rate"] == pytest.approx(2 / 3)


# --- edge cases: scope rejections, judge errors, LLM failures -------------------------------


def test_scope_false_positive_is_not_a_retrieval_miss() -> None:
    scope_refused = refused(
        "a2",
        T.TEK_BELGE,
        "kapsam_disi",
        durum="kapsam_disi",
        retrieved=[],
        evidence_hits=[None],
        first_hit_rank=None,
        doc_hit_at_5=False,
        doc_hit_at_10=False,
    )
    rows = [make_result("a1"), scope_refused]
    assert retrieval_metrics(rows)["recall_at_5"] == 1.0  # only a1 counts
    assert generation_metrics(rows)["false_rejection_rate"] == 0.5  # still a false rejection
    assert run_counts(rows)["scope_false_rejections"] == 1


def test_llm_error_did_not_reach_generation() -> None:
    error = refused("a", T.TEK_BELGE, "llm_hatasi", llm_calls=1)
    assert not reached_generation(error)
    assert reached_generation(refused("b", T.TEK_BELGE, "dayanaksiz_sayi", llm_calls=1))


def test_without_judge_accuracy_is_the_number_rule() -> None:
    no_judge = {"dogru": None, "sadik": None, "atif_dogru": None, "gerekce": ""}
    rows = [
        make_result("a", judge=no_judge),  # numbers found
        make_result(
            "b",
            judge=no_judge,
            rule_check={
                "sayilar_bulundu": False,
                "atif_kesinligi": 1.0,
                "programlara_ayrildi": None,
            },
        ),
        refused("c", T.TEK_BELGE, "esik_alti"),  # a refusal is wrong
        make_result(
            "d",
            judge=no_judge,
            rule_check={
                "sayilar_bulundu": None,
                "atif_kesinligi": 1.0,
                "programlara_ayrildi": None,
            },
        ),  # no expected number: undecided
    ]
    metrics = generation_metrics(rows, judged=False)
    assert "answer_accuracy" not in metrics
    assert metrics["number_rule_accuracy"] == pytest.approx(1 / 3)
    assert "number_rule_accuracy" not in generation_metrics(rows)


def test_run_counts_judge_errors() -> None:
    failed = {"dogru": None, "sadik": None, "atif_dogru": None, "gerekce": "hakem_hatasi"}
    rows = [make_result("a"), make_result("b", judge=failed)]
    assert run_counts(rows) == {"judge_errors": 1, "scope_false_rejections": 0}


def test_recompute_rebuilds_aggregates_from_rows() -> None:
    rows = _eight()
    stale = make_metrics(
        judge_model="yok",
        counts={"questions": 99, "docs": 120, "chunks": 4000},
        retrieval={"recall_at_5": 0.0},
    )
    fresh = recompute(stale, rows)
    assert fresh.retrieval == retrieval_metrics(rows)
    assert fresh.generation == generation_metrics(rows, judged=False)
    assert fresh.counts == {
        "questions": 8,
        "docs": 120,
        "chunks": 4000,
        "judge_errors": 1,
        "scope_false_rejections": 0,
    }
    assert fresh.refusals == refusal_counts(rows)
    assert "number_rule_accuracy" in fresh.by_type["tek_belge"]
    assert fresh.run_id == stale.run_id and fresh.started_at == stale.started_at
