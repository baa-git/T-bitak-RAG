"""Run-level metrics computed from ``per_question.jsonl`` rows (FR-E6, FR-E7, NFR-PERF1).

Every function reads :class:`QuestionResult` rows only, so a report can be recomputed from
the saved files without running anything again. A metric without data (for example
recall on a run with no answerable question) is *omitted* from the returned dict instead of
being NaN, because NaN does not survive a JSON round trip.

Refusal reasons (``answer["red_nedeni"]``) are recorded by the runner:

* before the LLM: ``kapsam_disi`` (scope check), ``sonuc_yok`` (nothing retrieved),
  ``esik_alti`` (retrieval gate);
* after the LLM: ``cevaplanamaz`` (the model said it cannot answer), ``gecersiz_cikti``,
  ``atifsiz_cevap`` / ``gecersiz_atif`` (citation check), ``dayanaksiz_sayi`` (number check),
  ``atif_reddi`` / ``bilgi_bulunamadi`` when the pipeline did not say more;
* ``llm_hatasi``: the answer LLM failed (connection, truncated output). This is an *error*,
  not a refusal: it is excluded from the correct/false rejection rates, counts as a wrong
  answer on answerable questions and is reported as ``error_rate``.
"""

import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from typing import Any

import numpy as np

from tubitak_rag.domain.enums import AnswerStatus, QuestionType
from tubitak_rag.domain.evaluation import QuestionResult, RunMetrics, TestQuestion
from tubitak_rag.evaluation.judge import JUDGE_ERROR
from tubitak_rag.evaluation.testset import ANSWERABLE
from tubitak_rag.guardrails.numbers import UNGROUNDED_NUMBER
from tubitak_rag.pipeline import BELOW_THRESHOLD, LLM_FAILURE, NO_RESULTS, OUT_OF_SCOPE

UNANSWERABLE = frozenset({QuestionType.CEVAPSIZ, QuestionType.KAPSAM_DISI})
REFUSALS = frozenset(
    {AnswerStatus.BILGI_BULUNAMADI, AnswerStatus.KAPSAM_DISI, AnswerStatus.ATIF_REDDI}
)

# The pipeline's codes, so the two can never drift apart.
REASON_OUT_OF_SCOPE = OUT_OF_SCOPE
REASON_NO_RESULTS = NO_RESULTS
REASON_BELOW_THRESHOLD = BELOW_THRESHOLD
REASON_LLM_ERROR = LLM_FAILURE
# Runner fallbacks for a pipeline whose trace has no ``rejection_reason``: the answer status.
REASON_CITATION = AnswerStatus.ATIF_REDDI.value
REASON_NOT_FOUND = AnswerStatus.BILGI_BULUNAMADI.value
PRE_GENERATION_REASONS = frozenset({REASON_OUT_OF_SCOPE, REASON_NO_RESULTS, REASON_BELOW_THRESHOLD})

LATENCY_STAGES = ("retrieval", "rerank", "llm", "total")

Metrics = dict[str, float]


# --- helpers -------------------------------------------------------------------------------


def mean_ignoring_nan(values: Iterable[float | None]) -> float:
    """Mean of the finite values; NaN when there are none."""
    finite = [v for v in values if v is not None and not math.isnan(v)]
    return sum(finite) / len(finite) if finite else math.nan


def _rate(flags: Iterable[bool | None]) -> float:
    return mean_ignoring_nan(None if flag is None else float(flag) for flag in flags)


def _finite(metrics: Mapping[str, float]) -> Metrics:
    return {key: value for key, value in metrics.items() if not math.isnan(value)}


def status(result: QuestionResult) -> AnswerStatus:
    """Answer status of a result row."""
    return AnswerStatus(result.answer["durum"])


def refusal_reason(result: QuestionResult) -> str | None:
    """Refusal reason code of a result row (None when answered)."""
    reason = result.answer.get("red_nedeni")
    return str(reason) if reason else None


def is_error(result: QuestionResult) -> bool:
    """Whether the answer LLM failed for this question (an error, not a refusal)."""
    return refusal_reason(result) == REASON_LLM_ERROR


def reached_generation(result: QuestionResult) -> bool:
    """Whether the answer LLM produced an output for this question.

    False when scope or gate stopped the question before the LLM, and when the LLM call
    failed (``llm_hatasi``): no output means no number guardrail decision either.
    """
    reason = refusal_reason(result)
    return status(result) == AnswerStatus.CEVAPLANDI or (
        reason not in PRE_GENERATION_REASONS and reason != REASON_LLM_ERROR
    )


def is_scope_false_rejection(result: QuestionResult) -> bool:
    """An answerable question that the scope check called out of scope (never retrieved)."""
    return result.tur in ANSWERABLE and refusal_reason(result) == REASON_OUT_OF_SCOPE


# --- retrieval -----------------------------------------------------------------------------


def recall_at_k(hits: Sequence[int | None], k: int) -> float:
    """Share of evidence quotes found at rank <= k; NaN when there is no quote."""
    if not hits:
        return math.nan
    return sum(1 for rank in hits if rank is not None and rank <= k) / len(hits)


def mrr(first_ranks: Sequence[int | None]) -> float:
    """Mean reciprocal rank of the first hit (a miss counts 0); NaN for no questions."""
    if not first_ranks:
        return math.nan
    return sum(1 / rank for rank in first_ranks if rank is not None) / len(first_ranks)


def retrieval_metrics(
    results: Sequence[QuestionResult], k_values: Sequence[int] = (5, 10)
) -> Metrics:
    """recall_at_k for each k, mrr, doc_hit_at_5, doc_hit_at_10 over answerable types.

    Answerable questions stopped by the scope check are left out: nothing was retrieved, so
    they are a scope error (``run_counts``), not a retrieval miss.
    """
    rows = [r for r in results if r.tur in ANSWERABLE and not is_scope_false_rejection(r)]
    metrics = {
        f"recall_at_{k}": mean_ignoring_nan(recall_at_k(r.evidence_hits, k) for r in rows)
        for k in k_values
    }
    with_quotes = [r for r in rows if r.evidence_hits]
    metrics["mrr"] = mrr([r.first_hit_rank for r in with_quotes])
    metrics["doc_hit_at_5"] = _rate(r.doc_hit_at_5 for r in rows)
    metrics["doc_hit_at_10"] = _rate(r.doc_hit_at_10 for r in rows)
    return _finite(metrics)


# --- generation ----------------------------------------------------------------------------


def is_correct(result: QuestionResult) -> bool | None:
    """Answer correctness of an answerable question (None when nothing can decide).

    A refusal is never correct. Otherwise the judge decides, and an answer that contains
    every expected number also counts as correct; with neither signal it is excluded.
    """
    if status(result) != AnswerStatus.CEVAPLANDI:
        return False
    dogru = result.judge.get("dogru")
    numbers = result.rule_check.get("sayilar_bulundu")
    if dogru is None and numbers is None:
        return None
    return bool(dogru) or bool(numbers)


def number_rule_correct(result: QuestionResult) -> bool | None:
    """Correctness by the expected-number rule alone (None without expected numbers)."""
    if status(result) != AnswerStatus.CEVAPLANDI:
        return False
    numbers = result.rule_check.get("sayilar_bulundu")
    return None if numbers is None else bool(numbers)


def generation_metrics(results: Sequence[QuestionResult], *, judged: bool = True) -> Metrics:
    """Answer quality and guardrail behaviour (FR-E7, FR-GR4, FR-GR8).

    answer_accuracy, faithfulness, citation_precision, correct_rejection_rate,
    false_rejection_rate, scope_rejection_rate, programs_split_rate, plus the number
    guardrail: number_rejection_rate (refused with ``dayanaksiz_sayi``) and
    number_strip_rate (sentences removed but answered), both over the questions that
    reached the LLM; error_rate (``llm_hatasi`` over all questions) and, when the pipeline
    reports citation repairs, citation_repair_rate (answers with an added citation).

    A run without a judge (``judged=False``) has no answer_accuracy; it reports
    number_rule_accuracy instead (expected numbers present, refusals wrong), which is a
    weaker signal and must not be compared with a judged accuracy.
    """
    answerable = [r for r in results if r.tur in ANSWERABLE]
    unanswerable = [r for r in results if r.tur in UNANSWERABLE]
    answered = [r for r in results if status(r) == AnswerStatus.CEVAPLANDI]
    generated = [r for r in results if reached_generation(r)]
    repairs = [
        r.answer["atif_onarildi"] for r in answered if r.answer.get("atif_onarildi") is not None
    ]
    metrics = {
        "answer_accuracy": _rate(is_correct(r) for r in answerable) if judged else math.nan,
        "number_rule_accuracy": (
            math.nan if judged else _rate(number_rule_correct(r) for r in answerable)
        ),
        "faithfulness": _rate(r.judge.get("sadik") for r in answered),
        "citation_precision": mean_ignoring_nan(
            r.rule_check.get("atif_kesinligi") for r in answered
        ),
        "correct_rejection_rate": _rate(
            status(r) in REFUSALS for r in unanswerable if not is_error(r)
        ),
        "false_rejection_rate": _rate(
            status(r) != AnswerStatus.CEVAPLANDI for r in answerable if not is_error(r)
        ),
        "scope_rejection_rate": _rate(
            status(r) == AnswerStatus.KAPSAM_DISI
            for r in results
            if r.tur == QuestionType.KAPSAM_DISI
        ),
        "programs_split_rate": _rate(
            r.rule_check.get("programlara_ayrildi")
            for r in results
            if r.tur == QuestionType.BELIRSIZ
        ),
        "number_rejection_rate": _rate(refusal_reason(r) == UNGROUNDED_NUMBER for r in generated),
        "number_strip_rate": _rate(bool(r.answer.get("sayi_ayiklandi")) for r in generated),
        "error_rate": _rate(is_error(r) for r in results),
        "citation_repair_rate": _rate(count > 0 for count in repairs),
    }
    return _finite(metrics)


def refusal_counts(results: Sequence[QuestionResult]) -> dict[str, int]:
    """How many questions were refused for each reason, most frequent first."""
    counts = Counter(reason for r in results if (reason := refusal_reason(r)))
    return dict(counts.most_common())


# --- latency -------------------------------------------------------------------------------


def latency_metrics(results: Sequence[QuestionResult]) -> Metrics:
    """p50/p95 in ms per stage; the LLM stage only over questions that called the LLM."""
    metrics: Metrics = {}
    for stage in LATENCY_STAGES:
        rows = [r for r in results if r.llm_calls > 0] if stage == "llm" else results
        values = [r.latency_ms[stage] for r in rows if stage in r.latency_ms]
        if values:
            p50, p95 = np.percentile(np.asarray(values, dtype=float), [50, 95])
            metrics[f"{stage}_p50"] = float(p50)
            metrics[f"{stage}_p95"] = float(p95)
    return metrics


# --- breakdowns ----------------------------------------------------------------------------


def by_type(
    results: Sequence[QuestionResult], fn: Callable[[Sequence[QuestionResult]], Metrics]
) -> dict[str, Metrics]:
    """Apply ``fn`` to the results of each question type (types without rows are skipped)."""
    groups: dict[str, list[QuestionResult]] = {}
    for result in results:
        groups.setdefault(result.tur.value, []).append(result)
    return {tur: fn(rows) for tur in (t.value for t in QuestionType) if (rows := groups.get(tur))}


def by_verified(
    results: Sequence[QuestionResult],
    questions: Sequence[TestQuestion],
    fn: Callable[[Sequence[QuestionResult]], Metrics] | None = None,
) -> dict[str, Metrics]:
    """``fn`` (default: retrieval + generation) for human-verified vs unverified questions."""
    compute = fn or quality_metrics
    verified = {q.id for q in questions if q.dogrulandi}
    groups = {
        "dogrulanmis": [r for r in results if r.id in verified],
        "dogrulanmamis": [r for r in results if r.id not in verified],
    }
    return {name: compute(rows) for name, rows in groups.items() if rows}


def quality_metrics(results: Sequence[QuestionResult], *, judged: bool = True) -> Metrics:
    """Retrieval and generation metrics in one dict (used for the per-type table)."""
    return {**retrieval_metrics(results), **generation_metrics(results, judged=judged)}


# --- whole run -----------------------------------------------------------------------------


def run_counts(results: Sequence[QuestionResult]) -> dict[str, int]:
    """Counts that qualify the metrics: judge failures and scope false rejections."""
    return {
        "judge_errors": sum(r.judge.get("gerekce") == JUDGE_ERROR for r in results),
        "scope_false_rejections": sum(is_scope_false_rejection(r) for r in results),
    }


def aggregate(
    results: Sequence[QuestionResult], *, k_values: Sequence[int] = (5, 10), judged: bool = True
) -> dict[str, Any]:
    """Every aggregated ``RunMetrics`` field that is derived from the per-question rows."""
    return {
        "retrieval": retrieval_metrics(results, k_values),
        "generation": generation_metrics(results, judged=judged),
        "latency_ms": latency_metrics(results),
        "by_type": by_type(results, partial(quality_metrics, judged=judged)),
        "refusals": refusal_counts(results),
    }


def recompute(
    metrics: RunMetrics, results: Sequence[QuestionResult], k_values: Sequence[int] = (5, 10)
) -> RunMetrics:
    """``metrics`` with every row-derived field rebuilt from ``results`` (``rag report``).

    Run facts (ids, times, models, corpus size) are kept; a run without a judge is
    recognised by ``judge_model == "yok"``.
    """
    judged = metrics.judge_model != "yok"
    counts = {**metrics.counts, "questions": len(results), **run_counts(results)}
    fields = aggregate(results, k_values=k_values, judged=judged)
    return metrics.model_copy(update={**fields, "counts": counts})
