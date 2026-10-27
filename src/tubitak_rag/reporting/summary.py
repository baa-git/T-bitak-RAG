"""``runs/ozet.md``: the whole experiment ladder in one table and a few sentences (FR-X8).

Accuracy is shown only when every run in the table has a manual review (``review.py``); the
automatic judge's accuracy is never shown.
"""

from collections.abc import Sequence
from pathlib import Path

from tubitak_rag.domain.evaluation import RunMetrics
from tubitak_rag.reporting.report import (
    SPECS,
    describe_method,
    find_previous_run,
    fmt,
    load_run,
    reviewed,
)
from tubitak_rag.reporting.review import JUDGE_KEYS, REVIEWER_SHORT, review_dir, strip_keys

LADDER = ("v0", "v0r", "v1", "v2", "v3", "v3b")
SUMMARY_FILE = "ozet.md"
COLUMNS = (
    ("recall_at_5", "R@5"),
    ("recall_at_10", "R@10"),
    ("mrr", "MRR"),
    ("answer_accuracy", "Doğruluk"),  # only with a manual review of every run
    ("correct_rejection_rate", "Doğru ret"),
    ("false_rejection_rate", "Yanlış ret"),
    ("total_p50", "Gecikme p50"),
)
HIGHLIGHTS = (
    ("recall_at_5", "en yüksek Recall@5"),
    ("answer_accuracy", "en yüksek cevap doğruluğu"),
    ("correct_rejection_rate", "en yüksek doğru ret"),
    ("total_p50", "en düşük gecikme"),
)
REVIEWED_COLUMNS = frozenset({"answer_accuracy", "correct_rejection_rate"})


def _value(metrics: RunMetrics, key: str) -> float | None:
    values: dict[str, float] = getattr(metrics, SPECS[key].group)
    return values.get(key)


def _best(runs: Sequence[RunMetrics], key: str) -> tuple[float, str] | None:
    spec = SPECS[key]
    scored = [(v, m.experiment) for m in runs if (v := _value(m, key)) is not None]
    if not scored:
        return None
    return max(scored) if spec.higher_is_better else min(scored)


def render_summary(runs: Sequence[RunMetrics], reviewer: str | None = None) -> str:
    """Summary of ``runs`` in ladder order.

    ``reviewer``: every run carries that manual review's metrics (``apply_review``); without
    it the judge-based metrics are dropped and there is no accuracy column.
    """
    lines = ["# Deney özeti", ""]
    if not runs:
        return "\n".join([*lines, "Henüz tamamlanmış koşu yok.", ""])
    if reviewer is None:
        runs = [strip_keys(m, JUDGE_KEYS) for m in runs]
    questions = sorted({m.counts.get("questions", 0) for m in runs})
    intro = (
        f"Her satır bir yapılandırmadır; hepsi aynı test setiyle ({'/'.join(map(str, questions))} "
        "soru) ve aynı dil modeliyle koşuldu. Arama metrikleri (R@5, R@10, MRR) kanıt "
        "alıntısıyla ölçülür."
    )
    if reviewer:
        intro += (
            f" Doğruluk ve doğru ret, cevapların {reviewer} ile tek tek değerlendirilmesinden "
            "gelir."
        )
    else:
        intro += " Cevaplar elle değerlendirilmediği için doğruluk sütunu yok."
    lines += [intro, ""]
    columns = [(k, h) for k, h in COLUMNS if reviewer or k != "answer_accuracy"]
    header = (
        "Koşu",
        "Yöntem",
        *(
            f"{h} ({REVIEWER_SHORT})" if reviewer and k in REVIEWED_COLUMNS else h
            for k, h in columns
        ),
    )
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for m in runs:
        cells = [fmt(_value(m, key), SPECS[key].kind) for key, _ in columns]
        lines.append("| " + " | ".join((m.experiment, describe_method(m), *cells)) + " |")
    highlights = []
    for key, label in HIGHLIGHTS:
        best = _best(runs, key)
        if best is not None:
            value, name = best
            highlights.append(f"{label} {name} ({fmt(value, SPECS[key].kind)})")
    if highlights:
        sentence = "; ".join(highlights)
        lines += ["", sentence[0].upper() + sentence[1:] + "."]
    return "\n".join(lines).rstrip() + "\n"


def write_summary(runs_dir: Path, experiment_order: Sequence[str] = LADDER) -> Path:
    """Write ``ozet.md`` from one run per experiment (``find_previous_run``: full runs first,
    newest of those), with metrics recomputed from its rows. The manual reviews are used
    only when every run has one."""
    raw: list[RunMetrics] = []
    shown: list[RunMetrics] = []
    reviewers: set[str | None] = set()
    for name in experiment_order:
        run_dir = find_previous_run(runs_dir, name)
        if run_dir is not None:
            metrics, results = load_run(run_dir)
            with_review, _, reviewer = reviewed(metrics, results, review_dir(runs_dir))
            raw.append(metrics)
            shown.append(with_review)
            reviewers.add(reviewer)
    # Only some runs reviewed: show no review at all, so every column stays comparable.
    reviewer = next(iter(reviewers)) if len(reviewers) == 1 else None
    runs = shown if reviewer is not None else raw
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / SUMMARY_FILE
    path.write_text(render_summary(runs, reviewer), encoding="utf-8", newline="\n")
    return path
