"""Run metrics: headline summary (``/api/runs``) and two-run comparison (demo reports)."""

from dataclasses import dataclass
from typing import Literal

from tubitak_rag.domain.evaluation import RunMetrics
from tubitak_rag.interfaces.web.schemas import RunSummary

Group = Literal["retrieval", "generation", "latency_ms"]
Kind = Literal["oran", "ms"]


@dataclass(frozen=True)
class MetricSpec:
    """How one metric is labelled, grouped, formatted and judged."""

    key: str
    label: str
    group: Group
    kind: Kind
    higher_is_better: bool


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("recall_at_5", "Recall@5", "retrieval", "oran", True),
    MetricSpec("recall_at_10", "Recall@10", "retrieval", "oran", True),
    MetricSpec("mrr", "MRR", "retrieval", "oran", True),
    MetricSpec("doc_hit_at_5", "Belge isabeti@5", "retrieval", "oran", True),
    MetricSpec("doc_hit_at_10", "Belge isabeti@10", "retrieval", "oran", True),
    MetricSpec("answer_accuracy", "Cevap doğruluğu", "generation", "oran", True),
    MetricSpec("faithfulness", "Kaynağa sadakat", "generation", "oran", True),
    MetricSpec("citation_precision", "Atıf kesinliği", "generation", "oran", True),
    MetricSpec("correct_rejection_rate", "Doğru ret oranı", "generation", "oran", True),
    MetricSpec("false_rejection_rate", "Yanlış ret oranı", "generation", "oran", False),
    MetricSpec("scope_rejection_rate", "Kapsam dışı ret oranı", "generation", "oran", True),
    MetricSpec("retrieval_p50", "Arama p50", "latency_ms", "ms", False),
    MetricSpec("retrieval_p95", "Arama p95", "latency_ms", "ms", False),
    MetricSpec("rerank_p50", "Yeniden sıralama p50", "latency_ms", "ms", False),
    MetricSpec("llm_p50", "Dil modeli p50", "latency_ms", "ms", False),
    MetricSpec("total_p50", "Toplam p50", "latency_ms", "ms", False),
    MetricSpec("total_p95", "Toplam p95", "latency_ms", "ms", False),
)
GROUP_LABELS: dict[Group, str] = {
    "retrieval": "Arama",
    "generation": "Cevap",
    "latency_ms": "Gecikme",
}
# Columns of the runs table, in order.
HEADLINE: tuple[str, ...] = (
    "recall_at_5",
    "mrr",
    "answer_accuracy",
    "correct_rejection_rate",
    "total_p50",
)
SPECS: dict[str, MetricSpec] = {spec.key: spec for spec in METRICS}


@dataclass(frozen=True)
class ComparedMetric:
    """One metric of run A next to run B; ``improved`` is None when equal or missing."""

    spec: MetricSpec
    a: float | None
    b: float | None
    delta: float | None
    improved: bool | None


@dataclass(frozen=True)
class SettingRow:
    """A configuration field of both runs and whether it changed."""

    label: str
    a: str
    b: str
    changed: bool


@dataclass(frozen=True)
class Comparison:
    """Everything the comparison partial renders."""

    a: RunMetrics
    b: RunMetrics
    settings: list[SettingRow]
    groups: list[tuple[str, list[ComparedMetric]]]


def _value(metrics: RunMetrics, spec: MetricSpec) -> float | None:
    values: dict[str, float] = getattr(metrics, spec.group)
    return values.get(spec.key)


def summarize(metrics: RunMetrics) -> RunSummary:
    """Build the runs-table row of one run from its ``metrics.json``."""
    values = {key: _value(metrics, SPECS[key]) for key in HEADLINE}
    return RunSummary(
        run_id=metrics.run_id,
        deney=metrics.experiment,
        tarih=metrics.started_at,
        durum="tamam" if metrics.finished_at else "suruyor",
        metrikler={key: value for key, value in values.items() if value is not None},
    )


def _compare_metric(spec: MetricSpec, a: float | None, b: float | None) -> ComparedMetric:
    if a is None or b is None:
        return ComparedMetric(spec, a, b, None, None)
    delta = b - a
    improved = None if delta == 0 else (delta > 0) == spec.higher_is_better
    return ComparedMetric(spec, a, b, delta, improved)


def _settings(a: RunMetrics, b: RunMetrics) -> list[SettingRow]:
    fields = (
        ("Deney", "experiment"),
        ("Parçalama", "chunker"),
        ("Embedding modeli", "embedder"),
        ("Arama", "retriever"),
        ("Yeniden sıralayıcı", "reranker"),
        ("Dil modeli", "llm_model"),
        ("Hakem", "judge_model"),
    )
    rows = []
    for label, attr in fields:
        va, vb = getattr(a, attr) or "yok", getattr(b, attr) or "yok"
        rows.append(SettingRow(label, va, vb, va != vb))
    return rows


def compare(a: RunMetrics, b: RunMetrics) -> Comparison:
    """Compare run ``b`` against run ``a`` metric by metric (delta = b − a)."""
    groups: list[tuple[str, list[ComparedMetric]]] = []
    for group, label in GROUP_LABELS.items():
        rows = [
            _compare_metric(spec, _value(a, spec), _value(b, spec))
            for spec in METRICS
            if spec.group == group
        ]
        rows = [row for row in rows if row.a is not None or row.b is not None]
        if rows:
            groups.append((label, rows))
    return Comparison(a=a, b=b, settings=_settings(a, b), groups=groups)
