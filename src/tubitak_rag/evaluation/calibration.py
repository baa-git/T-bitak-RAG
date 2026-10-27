"""Retrieval-gate threshold calibration from a finished run (FR-GR10).

Every result row stores the gate's score key and best score, whatever the gate decided. For
a candidate threshold ``t`` a question would be refused when ``gate_score < t``; over the
run's questions that gives

* the correct-rejection rate (CR) on unanswerable questions (``cevapsiz``, ``kapsam_disi``)
  and
* the false-rejection rate (FR) on answerable ones,

and ``F = 2·CR·(1−FR) / (CR + (1−FR))`` balances the two. The threshold is a precision/recall
trade-off: refusing more protects against invented answers and costs correct ones.

Scores of different keys live on different scales, so a sweep always uses one key: the key
of the run's last retrieval stage (``rerank`` logits roughly −11..+6, ``dense`` cosine,
``bm25``). ``rrf`` is a rank-fusion score that is not comparable between runs and is never
a threshold key.
"""

from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from tubitak_rag.domain.evaluation import QuestionResult
from tubitak_rag.evaluation.metrics import REASON_NO_RESULTS, UNANSWERABLE, refusal_reason
from tubitak_rag.evaluation.testset import ANSWERABLE
from tubitak_rag.guardrails.gate import SCORE_KEYS

# (start, stop, step) per score key; stop is inclusive.
_GRIDS = {"dense": (0.20, 0.80, 0.02), "rerank": (-11.0, 6.0, 0.5), "bm25": (0.0, 30.0, 1.0)}


class CalibrationRow(BaseModel):
    """Rates of one candidate threshold."""

    model_config = ConfigDict(frozen=True)

    threshold: float
    correct_rejection_rate: float
    false_rejection_rate: float
    f_score: float
    negatives: int = 0  # unanswerable questions in the sweep
    positives: int = 0  # answerable questions in the sweep


def default_grid(key: str) -> list[float]:
    """Thresholds to sweep for ``key``: dense 0.20..0.80/0.02, rerank −11..6/0.5, bm25 0..30/1."""
    if key not in _GRIDS:
        raise ValueError(
            f"'{key}' için eşik ızgarası yok; kalibre edilebilen anahtarlar: "
            f"{', '.join(SCORE_KEYS)} (rrf bir eşik anahtarı değildir)"
        )
    start, stop, step = _GRIDS[key]
    count = round((stop - start) / step) + 1
    return [round(start + i * step, 4) for i in range(count)]


def calibration_key(results: Sequence[QuestionResult]) -> str | None:
    """Most frequent gate key of a run (the key of its last stage), or None."""
    keys = Counter(r.gate_key for r in results if r.gate_key in _GRIDS)
    return keys.most_common(1)[0][0] if keys else None


def _f_score(cr: float, fr: float) -> float:
    kept = 1.0 - fr
    return 2 * cr * kept / (cr + kept) if cr + kept > 0 else 0.0


def calibrate(
    results: Sequence[QuestionResult], key: str, thresholds: Sequence[float]
) -> list[CalibrationRow]:
    """CR, FR and F for each threshold over the questions gated with ``key``.

    A group without questions contributes a rate of 0. Questions for which retrieval
    returned nothing (``sonuc_yok``) have no score and are refused at every threshold;
    questions stopped by the scope check never reached the gate and are not counted.
    """
    scored = [
        (r.tur, r.gate_score) for r in results if r.gate_key == key and r.gate_score is not None
    ]
    scored += [
        (r.tur, float("-inf"))
        for r in results
        if r.gate_key is None and refusal_reason(r) == REASON_NO_RESULTS
    ]
    negatives = [score for tur, score in scored if tur in UNANSWERABLE]
    positives = [score for tur, score in scored if tur in ANSWERABLE]

    def below(scores: Sequence[float], t: float) -> float:
        return sum(s < t for s in scores) / len(scores) if scores else 0.0

    table = []
    for t in thresholds:
        cr, fr = below(negatives, t), below(positives, t)
        table.append(
            CalibrationRow(
                threshold=t,
                correct_rejection_rate=cr,
                false_rejection_rate=fr,
                f_score=_f_score(cr, fr),
                negatives=len(negatives),
                positives=len(positives),
            )
        )
    return table


def suggest_threshold(
    rows: Sequence[CalibrationRow], max_false_rejection: float = 0.10
) -> CalibrationRow:
    """Best-F row whose false-rejection rate stays within the cap (lowest FR if none does).

    Ties go to the higher correct-rejection rate, then to the lower threshold.
    """
    if not rows:
        raise ValueError("Kalibrasyon tablosu boş")
    allowed = [r for r in rows if r.false_rejection_rate <= max_false_rejection]
    if not allowed:
        lowest = min(r.false_rejection_rate for r in rows)
        allowed = [r for r in rows if r.false_rejection_rate == lowest]
    return max(allowed, key=lambda r: (r.f_score, r.correct_rejection_rate, -r.threshold))
