"""``report.md``: a short, plain Markdown report of one run (FR-X3 … FR-X7, FR-E8, FR-GR10).

The report says what the configuration is (one paragraph), shows the key metrics next to the
previous rung's values and gives one good and two bad example questions. Everything is
generated from the run folder, so ``rag report`` can rebuild it at any time.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from tubitak_rag.config import AppConfig
from tubitak_rag.domain.enums import QuestionType
from tubitak_rag.domain.evaluation import QuestionResult, RunMetrics
from tubitak_rag.domain.reports import ComparisonRow
from tubitak_rag.evaluation.calibration import (
    CalibrationRow,
    calibrate,
    calibration_key,
    default_grid,
    suggest_threshold,
)
from tubitak_rag.evaluation.metrics import recompute, refusal_reason, status
from tubitak_rag.evaluation.runner import RunPaths, is_run_of
from tubitak_rag.guardrails.numbers import UNGROUNDED_NUMBER
from tubitak_rag.reporting.concepts import Concept, concept_for
from tubitak_rag.reporting.examples import pick_examples
from tubitak_rag.reporting.review import (
    JUDGE_KEYS,
    REVIEW_KEYS,
    REVIEWER,
    REVIEWER_SHORT,
    apply_review,
    find_review,
    review_counts,
    review_dir,
    strip_keys,
)

Group = Literal["retrieval", "generation", "latency_ms"]
Kind = Literal["oran", "ms"]
MAX_FALSE_REJECTION = 0.10
ANSWER_CHARS = 400  # longer answers are cut in the examples


@dataclass(frozen=True)
class MetricSpec:
    """Label, group, unit and direction of one metric."""

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
    MetricSpec("number_rule_accuracy", "Sayı kuralı doğruluğu", "generation", "oran", True),
    MetricSpec("faithfulness", "Kaynağa sadakat", "generation", "oran", True),
    MetricSpec("citation_precision", "Atıf kesinliği", "generation", "oran", True),
    MetricSpec("correct_rejection_rate", "Doğru ret oranı", "generation", "oran", True),
    MetricSpec("false_rejection_rate", "Yanlış ret oranı", "generation", "oran", False),
    MetricSpec("scope_rejection_rate", "Kapsam dışı ret oranı", "generation", "oran", True),
    MetricSpec("programs_split_rate", "Programlara ayırma oranı", "generation", "oran", True),
    MetricSpec("number_rejection_rate", "Sayı kontrolü ret oranı", "generation", "oran", False),
    MetricSpec("number_strip_rate", "Sayı kontrolü ayıklama oranı", "generation", "oran", False),
    MetricSpec("citation_repair_rate", "Atıf onarım oranı", "generation", "oran", False),
    MetricSpec("error_rate", "Hata oranı (dil modeli)", "generation", "oran", False),
    MetricSpec("retrieval_p50", "Arama p50", "latency_ms", "ms", False),
    MetricSpec("retrieval_p95", "Arama p95", "latency_ms", "ms", False),
    MetricSpec("rerank_p50", "Yeniden sıralama p50", "latency_ms", "ms", False),
    MetricSpec("llm_p50", "Dil modeli p50", "latency_ms", "ms", False),
    MetricSpec("total_p50", "Toplam p50", "latency_ms", "ms", False),
    MetricSpec("total_p95", "Toplam p95", "latency_ms", "ms", False),
)
SPECS: dict[str, MetricSpec] = {spec.key: spec for spec in METRICS}

# The report's table: retrieval, answer quality, refusals and latency, in this order.
KEY_METRICS: tuple[str, ...] = (
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "answer_accuracy",  # only with a manual review (Claude)
    "faithfulness",  # only with a manual review (Claude)
    "number_rule_accuracy",  # only present in runs without a judge
    "correct_rejection_rate",
    "false_rejection_rate",
    "total_p50",
)

TYPE_LABELS: dict[str, str] = {
    QuestionType.TEK_BELGE: "tek belge",
    QuestionType.COK_BELGE: "çok belge",
    QuestionType.CEVAPSIZ: "cevapsız",
    QuestionType.KAPSAM_DISI: "kapsam dışı",
    QuestionType.SURUM: "sürüm",
    QuestionType.BELIRSIZ: "belirsiz",
}
REASON_LABELS: dict[str, str] = {
    "kapsam_disi": "kapsam dışı",
    "sonuc_yok": "arama sonuç döndürmedi",
    "esik_alti": "skor eşiğin altında",
    "llm_hatasi": "dil modeli hatası",
    "cevaplanamaz": "model 'kaynaklarda yok' dedi",
    "gecersiz_cikti": "geçersiz JSON",
    "atifsiz_cevap": "atıfsız cevap",
    "gecersiz_atif": "geçersiz atıf",
    UNGROUNDED_NUMBER: "kaynakta olmayan sayı",
    "atif_reddi": "atıf reddi",
    "bilgi_bulunamadi": "bilgi bulunamadı",
}
STATUS_LABELS: dict[str, str] = {
    "cevaplandi": "cevaplandı",
    "bilgi_bulunamadi": "bilgi bulunamadı",
    "kapsam_disi": "kapsam dışı",
    "atif_reddi": "atıf denetiminden geçmedi",
}
CHUNKER_LABELS: dict[str, str] = {
    "fixed": "sabit pencere",
    "structural": "yapı bazlı parçalama",
    "structural_ctx": "yapı bazlı parçalama + bağlam başlığı",
}
RETRIEVER_LABELS: dict[str, str] = {
    "dense": "dense arama",
    "bm25": "BM25 arama",
    "hybrid": "hibrit arama (dense + BM25, RRF)",
}
EMBEDDER_LABELS: dict[str, str] = {
    "google/embeddinggemma-2": "EmbeddingGemma 2",
    "BAAI/bge-m3": "bge-m3",
}
SECTIONS = ("## Ana metrikler", "## Örnekler")


# --- formatting ----------------------------------------------------------------------------


def fmt(value: float | None, kind: Kind = "oran") -> str:
    """Turkish number format: "0,725" for rates, "1 250 ms" for latencies, "—" for none."""
    if value is None:
        return "—"
    if kind == "ms":
        return f"{value:,.0f} ms".replace(",", " ")
    return f"{value:.3f}".replace(".", ",")


def fmt_delta(delta: float | None, kind: Kind = "oran") -> str:
    """Signed difference ("+0,110", "−35 ms")."""
    if delta is None:
        return "—"
    sign = "+" if delta >= 0 else "−"
    return sign + fmt(abs(delta), kind)


def _count(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def _threshold(value: float) -> str:
    return f"{value:g}".replace(".", ",")


def _duration(seconds: float) -> str:
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes} dk {secs} sn" if minutes else f"{secs} sn"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def _short(text: str, limit: int = ANSWER_CHARS) -> str:
    text = _cell(text)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def describe_method(metrics: RunMetrics) -> str:
    """The configuration in words, e.g. "yapı bazlı parçalama + bağlam başlığı, hibrit arama
    (dense + BM25, RRF), reranker, embedding EmbeddingGemma 2"."""
    parts = [
        CHUNKER_LABELS.get(metrics.chunker, metrics.chunker),
        RETRIEVER_LABELS.get(metrics.retriever, metrics.retriever),
    ]
    if metrics.reranker:
        parts.append("reranker")
    embedder = EMBEDDER_LABELS.get(metrics.embedder, metrics.embedder.rsplit("/", 1)[-1])
    parts.append(f"embedding {embedder}")
    return ", ".join(parts)


# --- loading -------------------------------------------------------------------------------


def _k_values(paths: RunPaths) -> tuple[int, ...]:
    try:
        data = yaml.safe_load(paths.config_yaml.read_text(encoding="utf-8"))
        return tuple(int(k) for k in data["eval"]["k_values"])
    except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError):
        return (5, 10)


def load_run(run_dir: Path) -> tuple[RunMetrics, list[QuestionResult]]:
    """Read a run folder; the aggregated metrics are recomputed from ``per_question.jsonl``.

    Without ``per_question.jsonl`` the saved metrics are returned as they are. Recomputing
    keeps every report consistent with the current metric definitions even for
    runs written by an older version (``rag report --all`` rewrites ``metrics.json``).
    """
    paths = RunPaths(run_dir)
    metrics = RunMetrics.model_validate_json(paths.metrics_json.read_text(encoding="utf-8"))
    if not paths.per_question_jsonl.is_file():  # metrics only (e.g. copied summaries)
        return metrics, []
    lines = paths.per_question_jsonl.read_text(encoding="utf-8").splitlines()
    results = [QuestionResult.model_validate_json(line) for line in lines if line.strip()]
    return recompute(metrics, results, _k_values(paths)), results


def reviewed(
    metrics: RunMetrics, results: Sequence[QuestionResult], judge_dir: Path
) -> tuple[RunMetrics, list[QuestionResult], str | None]:
    """The run with its manual review applied when one belongs to it, and the reviewer."""
    review = find_review(judge_dir, metrics.experiment, results)
    if review is None:
        return metrics, list(results), None
    shown, rows = apply_review(metrics, results, review)
    return shown, rows, REVIEWER


def load_reviewed_run(
    run_dir: Path, runs_dir: Path
) -> tuple[RunMetrics, list[QuestionResult], str | None]:
    """``load_run`` plus the manual review from ``<artifacts>/judge`` when it fits the run."""
    metrics, results = load_run(run_dir)
    return reviewed(metrics, results, review_dir(runs_dir))


def _question_count(run_dir: Path) -> int:
    text = RunPaths(run_dir).metrics_json.read_text(encoding="utf-8")
    return RunMetrics.model_validate_json(text).counts.get("questions", 0)


def find_previous_run(runs_dir: Path, experiment_name: str | None) -> Path | None:
    """The run of ``experiment_name`` to compare with (None if there is none).

    Partial runs (``--limit``, ``--ids``) are not representative, so the runs with the most
    questions win and the newest of those is taken.
    """
    if not experiment_name or not runs_dir.is_dir():
        return None
    runs = [
        d
        for d in runs_dir.iterdir()
        if d.is_dir() and is_run_of(d.name, experiment_name) and RunPaths(d).metrics_json.is_file()
    ]
    if not runs:
        return None
    return max(runs, key=lambda d: (_question_count(d), d.name))


# --- comparison ----------------------------------------------------------------------------


def _value(metrics: RunMetrics, spec: MetricSpec) -> float | None:
    values: dict[str, float] = getattr(metrics, spec.group)
    return values.get(spec.key)


def comparison_rows(
    current: RunMetrics, previous: RunMetrics | None, keys: Sequence[str] = KEY_METRICS
) -> list[ComparisonRow]:
    """The metrics ``keys`` of ``current`` (when present) with the previous run's value and
    the difference."""
    rows = []
    for key in keys:
        spec = SPECS[key]
        value = _value(current, spec)
        if value is None:
            continue
        before = _value(previous, spec) if previous is not None else None
        delta = None if before is None else value - before
        rows.append(ComparisonRow(metric=spec.key, current=value, previous=before, delta=delta))
    return rows


# --- sections ------------------------------------------------------------------------------


def _header(metrics: RunMetrics, concept: Concept, reviewer: str | None) -> list[str]:
    counts = metrics.counts
    facts = [f"koşu `{metrics.run_id}`", f"{counts.get('questions', 0)} soru"]
    if "docs" in counts and "chunks" in counts:
        facts.append(f"{_count(counts['docs'])} belge kaydı, {_count(counts['chunks'])} parça")
    facts.append(f"dil modeli {metrics.llm_model}")
    if reviewer:
        facts.append(f"cevaplar {reviewer} ile tek tek değerlendirildi")
    facts.append(f"süre {_duration(metrics.wall_clock_s)}")
    return [
        f"# Deney raporu: {metrics.experiment} – {describe_method(metrics)}",
        "",
        concept.summary,
        "",
        "Künye: " + " · ".join(facts) + ".",
    ]


def _label(spec: MetricSpec, reviewer: str | None) -> str:
    if reviewer and spec.key in REVIEW_KEYS:
        return f"{spec.label} ({REVIEWER_SHORT})"
    return spec.label


ERROR_PLACES = {"retrieval": "arama", "uretim": "üretim", "belirsiz": "belirsiz"}


def _review_line(results: Sequence[QuestionResult], reviewer: str | None) -> str | None:
    if not reviewer:
        return None
    counts = review_counts(results)
    if not counts:
        return None
    where = [
        f"{label} {counts[f'gereksiz_ret:{code}']}"
        for code, label in ERROR_PLACES.items()
        if counts.get(f"gereksiz_ret:{code}")
    ]
    line = (
        f"{REVIEWER_SHORT} değerlendirmesi: uydurma {counts.get('uydurma', 0)} · gereksiz ret "
        f"{counts.get('gereksiz_ret', 0)}"
    )
    if where:
        line += f" (hatanın yeri: {', '.join(where)})"
    return line + "."


def _refusal_line(metrics: RunMetrics) -> str | None:
    if not metrics.refusals:
        return None
    parts = [f"{REASON_LABELS.get(code, code)} {n}" for code, n in metrics.refusals.items()]
    return "Retler (soru sayısı): " + " · ".join(parts) + "."


def _calibration_line(calibration: Sequence[CalibrationRow] | None, key: str | None) -> str | None:
    if not calibration:
        return None
    best = suggest_threshold(calibration, MAX_FALSE_REJECTION)
    return (
        f"Eşik kalibrasyonu: `{key or '?'}` skoru için yanlış ret ≤ {fmt(MAX_FALSE_REJECTION)} "
        f"koşuluyla önerilen eşik **{_threshold(best.threshold)}** (doğru ret "
        f"{fmt(best.correct_rejection_rate)}, yanlış ret {fmt(best.false_rejection_rate)})."
    )


def _metrics_section(
    metrics: RunMetrics,
    results: Sequence[QuestionResult],
    previous: RunMetrics | None,
    calibration: Sequence[CalibrationRow] | None,
    key: str | None,
    reviewer: str | None,
    previous_reviewed: bool,
) -> list[str]:
    prev_name = previous.experiment if previous is not None else "yok"
    rows = []
    for row in comparison_rows(metrics, previous):
        spec = SPECS[row.metric]
        rows.append(
            (
                _label(spec, reviewer),
                fmt(row.previous, spec.kind),
                fmt(row.current, spec.kind),
                fmt_delta(row.delta, spec.kind),
            )
        )
    lines = [SECTIONS[0], ""]
    if previous is None:
        lines += ["Karşılaştırılacak önceki basamak yok; bu koşu taban çizgisidir.", ""]
    else:
        before, now = previous.counts.get("questions", 0), metrics.counts.get("questions", 0)
        if before != now:
            lines += [
                f"**Dikkat:** önceki koşu {before} soru, bu koşu {now} soru. Karşılaştırma "
                "eşit değil.",
                "",
            ]
        if reviewer and not previous_reviewed:
            lines += [
                f"**Dikkat:** önceki koşunun cevapları {reviewer} ile değerlendirilmedi; "
                "doğruluk, sadakat ve doğru ret karşılaştırılmadı.",
                "",
            ]
    if previous is None:
        lines += _table(("Metrik", "Bu koşu"), [(label, now) for label, _, now, _ in rows])
    else:
        lines += _table(("Metrik", f"Önceki ({prev_name})", "Bu koşu", "Fark"), rows)
    lines += [
        "",
        "Recall@k: kanıt alıntılarının ilk k parçada bulunma oranı. MRR: ilk doğru parçanın "
        "sırasının tersinin ortalaması (1. sıra = 1, 2. sıra = 0,5). Doğru ret: cevapsız ve "
        "kapsam dışı sorularda cevap vermeme oranı. Yanlış ret: cevabı olan sorularda gereksiz "
        "ret oranı."
        + (
            f" Doğruluk, sadakat ve doğru ret {reviewer} ile tek tek yapılan değerlendirmeden "
            "gelir; cevap metninde 'kaynakta yok' diyen cevap da doğru ret sayılır."
            if reviewer
            else ""
        ),
    ]
    notes = [_refusal_line(metrics), _review_line(results, reviewer)]
    scope_misses = metrics.counts.get("scope_false_rejections", 0)
    if scope_misses:
        notes.append(f"Kapsam kontrolü {scope_misses} cevaplanabilir soruyu kapsam dışı saydı.")
    errors = metrics.refusals.get("llm_hatasi", 0)
    if errors:
        notes.append(
            f"**Dikkat:** {errors} soruda dil modeli hata verdi; bunlar ret değil hatadır."
        )
    notes.append(_calibration_line(calibration, key))
    present = [f"- {note}" for note in notes if note]
    if present:
        lines += ["", *present]
    return lines


def _example(kind: str, result: QuestionResult, reviewer: str | None) -> list[str]:
    rank = result.first_hit_rank
    reason = refusal_reason(result)
    durum = STATUS_LABELS.get(status(result).value, status(result).value)
    if reason:
        durum += f" ({REASON_LABELS.get(reason, reason)})"
    if result.tur in (QuestionType.CEVAPSIZ, QuestionType.KAPSAM_DISI):
        evidence = "beklenmiyor (corpus'ta cevabı yok)"
    elif not result.retrieved:
        evidence = "arama yapılmadı"
    elif rank is None:
        evidence = f"{len(result.retrieved)} adayda yok"
    else:
        evidence = f"{rank}. sırada"
    lines = [
        f"### {kind}: {result.id} ({TYPE_LABELS.get(result.tur, result.tur)})",
        "",
        f"**Soru:** {_cell(result.soru)}",
        "",
        f"- Kanıt: {evidence} · durum: {durum}",
    ]
    answer = str(result.answer.get("cevap", "")).strip()
    if status(result).value == "cevaplandi" and answer:
        lines.append(f"- Cevap: {_short(answer)}")
    note = str(result.judge.get("gerekce") or "").strip()
    if reviewer and note:  # the automatic judge's reasoning is never shown
        lines.append(f"- Değerlendirme notu ({REVIEWER_SHORT}): {_short(note, 220)}")
    return lines


def _examples_section(results: Sequence[QuestionResult], reviewer: str | None) -> list[str]:
    good, bad = pick_examples(results)
    lines = [SECTIONS[1]]
    for result in good:
        lines += ["", *_example("İyi örnek", result, reviewer)]
    for result in bad:
        lines += ["", *_example("Kötü örnek", result, reviewer)]
    if not good and not bad:
        lines += ["", "Bu koşuda gösterilecek örnek yok."]
    return lines


def render_report(
    metrics: RunMetrics,
    results: Sequence[QuestionResult],
    previous: RunMetrics | None,
    concept: Concept,
    calibration: Sequence[CalibrationRow] | None,
    calibration_key: str | None = None,
    *,
    reviewer: str | None = None,
    previous_reviewer: str | None = None,
) -> str:
    """The whole ``report.md`` text.

    ``reviewer`` names the manual review whose verdicts ``metrics`` and ``results`` carry
    (``apply_review``). Without one, the automatic judge's accuracy, faithfulness and notes
    are left out. A previous run without the same review is compared without the reviewed
    metrics.
    """
    if reviewer is None:
        metrics = strip_keys(metrics, JUDGE_KEYS)
        if previous is not None:
            previous = strip_keys(previous, JUDGE_KEYS)
    previous_reviewed = previous_reviewer is not None and previous_reviewer == reviewer
    if reviewer and previous is not None and not previous_reviewed:
        previous = strip_keys(previous, REVIEW_KEYS)
    blocks = [
        _header(metrics, concept, reviewer),
        _metrics_section(
            metrics, results, previous, calibration, calibration_key, reviewer, previous_reviewed
        ),
        _examples_section(results, reviewer),
    ]
    return "\n\n".join("\n".join(block).rstrip() for block in blocks) + "\n"


def write_report(run_dir: Path, runs_dir: Path, cfg: AppConfig) -> Path:
    """(Re)write ``report.md`` of ``run_dir``, comparing with the previous rung's newest run."""
    metrics, results = load_run(run_dir)
    paths = RunPaths(run_dir)
    paths.metrics_json.write_text(
        json.dumps(metrics.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    previous: RunMetrics | None = None
    previous_reviewer: str | None = None
    previous_dir = find_previous_run(runs_dir, cfg.experiment.previous)
    if previous_dir is not None:
        previous, _, previous_reviewer = load_reviewed_run(previous_dir, runs_dir)
    key = calibration_key(results)
    calibration = calibrate(results, key, default_grid(key)) if key else None
    shown, shown_results, reviewer = reviewed(metrics, results, review_dir(runs_dir))
    path = RunPaths(run_dir).report_md
    path.write_text(
        render_report(
            shown,
            shown_results,
            previous,
            concept_for(cfg.experiment.concept),
            calibration,
            key,
            reviewer=reviewer,
            previous_reviewer=previous_reviewer,
        ),
        encoding="utf-8",
        newline="\n",  # LF on Windows too: the reports are copied into the repository
    )
    return path
