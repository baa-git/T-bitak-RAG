"""Plain descriptions of a pipeline and of the corpus, derived from configs and manifests.

The query page shows, next to the configuration selector, which methods and models the
selected config uses; the Hakkında page lists the corpus and the components. Everything here
is read from ``AppConfig`` objects and manifest records, so a changed config file changes the
page without touching a template.
"""

import re
from collections import Counter
from collections.abc import Iterable, Sequence

from tubitak_rag.config import AppConfig
from tubitak_rag.domain.source import SourceEntry
from tubitak_rag.interfaces.web.schemas import CorpusGroup, MethodLine

SEPARATOR = " · "
REPO_URL = "https://github.com/baa-git/T-bitak-RAG"
DOCS_URL = f"{REPO_URL}/tree/main/docs"

# Display name, where it comes from and a link, per manifest source group.
_GROUPS: dict[str, tuple[str, str, str | None]] = {
    "tubitak_mevzuat": (
        "TÜBİTAK mevzuatı",
        "tubitak.gov.tr/tr/mevzuat",
        "https://tubitak.gov.tr/tr/mevzuat",
    ),
    "tubitak_program_sayfalari": (
        "TÜBİTAK destek programı sayfaları",
        "tubitak.gov.tr/tr/destekler (sayfalar ve ekli dosyalar)",
        "https://tubitak.gov.tr/tr/destekler",
    ),
    "sanayi_mevzuati": (
        "Ar-Ge ve teknoloji geliştirme bölgesi kanunları",
        "Gelir İdaresi Başkanlığı (GİB) mevzuat sayfası",
        "https://www.gib.gov.tr/mevzuat",
    ),
}
_FILE_LABELS = {"pdf": "PDF", "xlsx": "Excel"}
_STORE_LABELS = {"qdrant": "Qdrant", "numpy": "NumPy (bellek içi, salt okunur)"}
_LAW_NUMBER_RE = re.compile(r"^kanun-(\d+)$")


def methods_text(lines: Iterable[MethodLine]) -> str:
    """``"Parçalama: … · Embedding: …"``: the lines on one row."""
    return SEPARATOR.join(f"{line.etiket}: {line.deger}" for line in lines)


def _chunking(cfg: AppConfig) -> str:
    chunker = cfg.chunker
    if chunker.name == "fixed":
        text = f"sabit pencere ({chunker.fixed.size} token, {chunker.fixed.overlap} örtüşme)"
    else:
        text = "yapı bazlı"
    return text + (" + bağlam başlığı" if chunker.context_header else "")


def _bm25(cfg: AppConfig) -> str:
    return "BM25 (Türkçe kök bulma)" if cfg.bm25.stemming else "BM25"


def _search(cfg: AppConfig) -> str:
    mode = cfg.retriever.mode
    if mode == "dense":
        return "dense (vektör)"
    if mode == "bm25":
        return _bm25(cfg)
    return f"hibrit (BM25 + dense, RRF k={cfg.retriever.rrf_k})"


def describe_methods(cfg: AppConfig) -> list[MethodLine]:
    """Chunking, embedding, search, reranker and LLM of one config, in plain Turkish."""
    return [
        MethodLine(etiket="Parçalama", deger=_chunking(cfg)),
        MethodLine(
            etiket="Embedding", deger=f"{cfg.embedder.model} ({cfg.embedder.dimension} boyut)"
        ),
        MethodLine(etiket="Arama", deger=_search(cfg)),
        MethodLine(etiket="Reranker", deger=cfg.reranker.model if cfg.reranker.enabled else "yok"),
        MethodLine(etiket="LLM", deger=cfg.llm.model_name),
    ]


def _unique(values: Iterable[str]) -> str:
    return ", ".join(dict.fromkeys(values))


def component_lines(configs: Sequence[AppConfig]) -> list[MethodLine]:
    """Every model and index used by at least one of ``configs`` (each named once)."""
    rerankers = [c.reranker.model for c in configs if c.reranker.enabled]
    providers = [c.vector_store.provider for c in configs]
    hybrid = [c for c in configs if c.retriever.mode == "hybrid"]
    lines = [
        MethodLine(
            etiket="Embedding",
            deger=_unique(f"{c.embedder.model} ({c.embedder.dimension} boyut)" for c in configs),
        ),
        MethodLine(
            etiket="Vektör deposu",
            deger=_unique(_STORE_LABELS.get(p, p.capitalize()) for p in providers),
        ),
        MethodLine(etiket="Kelime araması", deger=_unique(_bm25(c) for c in configs)),
    ]
    if hybrid:
        lines.append(
            MethodLine(
                etiket="Birleştirme",
                deger=_unique(f"RRF (k={c.retriever.rrf_k})" for c in hybrid),
            )
        )
    if rerankers:
        lines.append(MethodLine(etiket="Reranker", deger=_unique(rerankers)))
    lines.append(MethodLine(etiket="LLM", deger=_unique(c.llm.model_name for c in configs)))
    return lines


_SCOPE_MODES = {
    "lexicon": "terim sözlüğü",
    "llm": "dil modeliyle sınıflandırma",
    "both": "terim sözlüğü, gerekirse dil modeliyle sınıflandırma",
    "off": "kapalı",
}
_NUMBER_MODES = {
    "off": "kapalı",
    "strip": "kaynakta geçmeyen sayı içeren cümle çıkarılır",
    "reject": "kaynakta geçmeyen sayı varsa cevap verilmez",
}


def _number(value: float) -> str:
    return f"{value:g}".replace(".", ",")


def guardrail_lines(cfg: AppConfig) -> list[MethodLine]:
    """The guards around the LLM, with the values of ``cfg``."""
    guard = cfg.guardrails
    thresholds = ", ".join(f"{key} ≥ {_number(v)}" for key, v in guard.thresholds.items())
    return [
        MethodLine(
            etiket="Kapsam kontrolü",
            deger=f"{_SCOPE_MODES[guard.scope_mode]}; kapsam dışı soru aranmaz",
        ),
        MethodLine(
            etiket="Arama eşiği",
            deger=f"en iyi skor eşiğin altındaysa cevap üretilmez ({thresholds})",
        ),
        MethodLine(
            etiket="Atıf doğrulama",
            deger="atıfsız cevap ya da verilmeyen kaynağa atıf reddedilir",
        ),
        MethodLine(
            etiket="Sayı doğrulama",
            deger=f"{_NUMBER_MODES[guard.number_check]}; sayı başka bir kaynakta birimiyle "
            "geçiyorsa atıf onarımı o kaynağı ekler",
        ),
    ]


def _group_summary(kinds: Counter[str]) -> str:
    parts = []
    if kinds["html"]:
        parts.append(f"{kinds['html']} sayfa")
    if kinds["gib_json"]:
        parts.append(f"{kinds['gib_json']} kanun")
    files = {k: n for k, n in kinds.items() if k not in {"html", "gib_json"} and n}
    by_kind = ", ".join(f"{n} {_FILE_LABELS.get(k, k)}" for k, n in sorted(files.items()))
    if files and parts:
        parts.append(f"{sum(files.values())} dosya ({by_kind})")
    elif files:
        parts.append(by_kind)
    return " + ".join(parts)


def _details(entries: Sequence[SourceEntry]) -> list[str]:
    """Law numbers and titles (only for groups made of laws)."""
    details = []
    for entry in entries:
        match = _LAW_NUMBER_RE.match(entry.doc_id)
        if match:
            details.append(f"{match.group(1)} sayılı {entry.title}")
    return details


def corpus_groups(entries: Iterable[SourceEntry]) -> list[CorpusGroup]:
    """Manifest records grouped by source, in manifest order, with per-kind counts."""
    grouped: dict[str, list[SourceEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.kaynak_grubu, []).append(entry)
    groups = []
    for key, members in grouped.items():
        name, origin, url = _GROUPS.get(key, (key, key, None))
        groups.append(
            CorpusGroup(
                kaynak_grubu=key,
                ad=name,
                kaynak=origin,
                kaynak_url=url,
                belge_sayisi=len(members),
                ozet=_group_summary(Counter(e.kind.value for e in members)),
                ayrinti=_details(members),
            )
        )
    return groups
