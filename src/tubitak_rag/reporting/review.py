"""Manual answer review (Claude) used by the reports instead of the automatic judge.

The answers of a run can be reviewed one by one outside the pipeline. The review lives next
to the runs, in ``<artifacts>/judge/``:

* ``<experiment>_claude.json``: answered answerable questions,
  ``[{"id", "dogru", "sadik", "not"}]``;
* ``<experiment>_rest_claude.json``: every other question (refused answerable ones and the
  ``cevapsiz`` / ``kapsam_disi`` questions), ``[{"id", "karar", "hata_yeri", "not"}]`` with
  ``karar`` one of ``dogru_ret``, ``uydurma``, ``gereksiz_ret``;
* optionally ``<experiment>_items.json`` / ``<experiment>_rest.json``: the answers that were
  reviewed (``id``, ``cevap`` / ``durum``).

A review is used only when it covers every question of the run and, if the reviewed answers
were saved, they are the run's answers. A stale review of an older run of the same
experiment is therefore ignored. Reports never show the automatic judge's verdicts; without
a review they show no accuracy at all.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tubitak_rag.domain.enums import AnswerStatus
from tubitak_rag.domain.evaluation import QuestionResult, RunMetrics
from tubitak_rag.evaluation.metrics import UNANSWERABLE, status
from tubitak_rag.evaluation.testset import ANSWERABLE

REVIEWER = "Claude (Opus)"
REVIEWER_SHORT = "Claude"
JUDGE_DIR = "judge"
# Generation metrics that come from a judge's verdicts; never shown without a review.
JUDGE_KEYS = ("answer_accuracy", "faithfulness")
# Metrics a review replaces (correct rejection: content refusals count as correct).
REVIEW_KEYS = (*JUDGE_KEYS, "correct_rejection_rate")
CORRECT_REFUSAL = "dogru_ret"


@dataclass(frozen=True)
class Review:
    """Verdicts of one manual review, keyed by question id."""

    answers: dict[str, dict[str, Any]]  # {dogru, sadik, not}
    others: dict[str, dict[str, Any]]  # {karar, hata_yeri, not}


def review_dir(runs_dir: Path) -> Path:
    """``<artifacts>/judge`` for ``<artifacts>/runs``."""
    return runs_dir.parent / JUDGE_DIR


def _read_list(path: Path) -> list[dict[str, Any]] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, list) or not all(isinstance(x, dict) and "id" in x for x in data):
        return None
    return data


def load_review(judge_dir: Path, experiment: str) -> Review | None:
    """The review of ``experiment`` (None when either file is missing or unreadable)."""
    answers = _read_list(judge_dir / f"{experiment}_claude.json")
    others = _read_list(judge_dir / f"{experiment}_rest_claude.json")
    if answers is None or others is None:
        return None
    return Review(
        answers={str(x["id"]): x for x in answers}, others={str(x["id"]): x for x in others}
    )


def _is_answered_answerable(result: QuestionResult) -> bool:
    return result.tur in ANSWERABLE and status(result) == AnswerStatus.CEVAPLANDI


def covers(review: Review, results: Sequence[QuestionResult]) -> bool:
    """Whether ``review`` has a verdict of the right kind for every row of the run."""
    return bool(results) and all(
        r.id in (review.answers if _is_answered_answerable(r) else review.others) for r in results
    )


def _same_text(a: object, b: object) -> bool:
    return " ".join(str(a or "").split()) == " ".join(str(b or "").split())


def snapshots_match(judge_dir: Path, experiment: str, results: Sequence[QuestionResult]) -> bool:
    """Whether the saved reviewed answers (when present) are the answers of this run."""
    by_id = {r.id: r for r in results}
    checks = (("_items.json", "cevap"), ("_rest.json", "durum"))
    for suffix, field in checks:
        path = judge_dir / f"{experiment}{suffix}"
        if not path.exists():
            continue
        items = _read_list(path)
        if items is None:
            return False
        for item in items:
            result = by_id.get(str(item["id"]))
            if result is None or not _same_text(item.get(field), result.answer.get(field)):
                return False
    return True


def find_review(
    judge_dir: Path, experiment: str, results: Sequence[QuestionResult]
) -> Review | None:
    """The review that belongs to this run, or None."""
    review = load_review(judge_dir, experiment)
    if review is None or not covers(review, results):
        return None
    return review if snapshots_match(judge_dir, experiment, results) else None


def _rate(flags: Sequence[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def apply_review(
    metrics: RunMetrics, results: Sequence[QuestionResult], review: Review
) -> tuple[RunMetrics, list[QuestionResult]]:
    """``metrics`` with accuracy, faithfulness and correct rejection taken from ``review``,
    and the rows with the reviewer's verdicts and notes in ``judge``."""
    rows: list[QuestionResult] = []
    for r in results:
        if _is_answered_answerable(r):
            verdict = review.answers[r.id]
            judge = {
                **r.judge,
                "dogru": verdict.get("dogru"),
                "sadik": verdict.get("sadik"),
                "gerekce": str(verdict.get("not") or ""),
            }
        else:
            other = review.others[r.id]
            judge = {
                **r.judge,
                "karar": other.get("karar"),
                "hata_yeri": other.get("hata_yeri"),
                "gerekce": str(other.get("not") or ""),
            }
        rows.append(r.model_copy(update={"judge": judge}))
    answerable = [r for r in rows if r.tur in ANSWERABLE]
    # An undecided faithfulness verdict (null) is left out of the rate.
    faithful = [
        r.judge["sadik"] is True
        for r in rows
        if _is_answered_answerable(r) and r.judge.get("sadik") is not None
    ]
    unanswerable = [r for r in rows if r.tur in UNANSWERABLE]
    values = {
        "answer_accuracy": _rate([r.judge.get("dogru") is True for r in answerable]),
        "faithfulness": _rate(faithful),
        "correct_rejection_rate": _rate(
            [r.judge.get("karar") == CORRECT_REFUSAL for r in unanswerable]
        ),
    }
    generation = strip_keys(metrics, REVIEW_KEYS).generation
    generation |= {key: value for key, value in values.items() if value is not None}
    return metrics.model_copy(update={"generation": generation}), rows


def strip_keys(metrics: RunMetrics, keys: Sequence[str] = JUDGE_KEYS) -> RunMetrics:
    """``metrics`` without the generation metrics ``keys``."""
    generation = {k: v for k, v in metrics.generation.items() if k not in keys}
    return metrics.model_copy(update={"generation": generation})


def review_counts(results: Sequence[QuestionResult]) -> dict[str, int]:
    """Counts of the reviewer's decisions on the non-answered rows (``karar``, ``hata_yeri``)."""
    counts: dict[str, int] = {}
    for r in results:
        karar = r.judge.get("karar")
        if karar:
            counts[str(karar)] = counts.get(str(karar), 0) + 1
            place = r.judge.get("hata_yeri")
            if karar == "gereksiz_ret" and place:
                key = f"gereksiz_ret:{place}"
                counts[key] = counts.get(key, 0) + 1
    return counts
