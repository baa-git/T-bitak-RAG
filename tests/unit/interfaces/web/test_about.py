"""Method lines and corpus summary derived from configs and manifests (Hakkında, selector)."""

from datetime import date
from pathlib import Path

import pytest

from tubitak_rag.config import AppConfig, VectorStoreConfig, load_config
from tubitak_rag.domain.enums import DocType, SourceKind
from tubitak_rag.domain.source import SourceEntry
from tubitak_rag.interfaces.web.about import (
    component_lines,
    corpus_groups,
    describe_methods,
    guardrail_lines,
    methods_text,
)


def _experiment(repo_root: Path, name: str) -> AppConfig:
    return load_config(repo_root / "configs" / "experiments" / f"{name}.yaml")


def test_v3_method_line_is_read_from_the_config_file(repo_root: Path) -> None:
    assert methods_text(describe_methods(_experiment(repo_root, "v3"))) == (
        "Parçalama: yapı bazlı + bağlam başlığı · Embedding: google/embeddinggemma-2 (768 boyut)"
        " · Arama: hibrit (BM25 + dense, RRF k=60) · Reranker: BAAI/bge-reranker-v2-m3"
        " · LLM: qwen3:4b"
    )


def test_v0_has_fixed_windows_dense_search_and_no_reranker(repo_root: Path) -> None:
    lines = dict((m.etiket, m.deger) for m in describe_methods(_experiment(repo_root, "v0")))
    assert lines["Parçalama"] == "sabit pencere (512 token, 64 örtüşme)"
    assert lines["Arama"] == "dense (vektör)"
    assert lines["Reranker"] == "yok"


def test_v3b_names_its_own_embedding_model(repo_root: Path) -> None:
    lines = dict((m.etiket, m.deger) for m in describe_methods(_experiment(repo_root, "v3b")))
    assert lines["Embedding"] == "BAAI/bge-m3 (1024 boyut)"


def test_method_line_follows_config_values() -> None:
    cfg = AppConfig.model_validate(
        {
            "retriever": {"mode": "bm25", "rrf_k": 30},
            "chunker": {"context_header": False},
            "llm": {"provider": "anthropic", "anthropic": {"model": "claude-x"}},
        }
    )
    lines = dict((m.etiket, m.deger) for m in describe_methods(cfg))
    assert lines["Parçalama"] == "yapı bazlı"
    assert lines["Arama"] == "BM25 (Türkçe kök bulma)"
    assert lines["LLM"] == "claude-x"


def test_no_marketing_phrase_in_any_method_line(repo_root: Path) -> None:
    for path in sorted((repo_root / "configs" / "experiments").glob("*.yaml")):
        text = methods_text(describe_methods(load_config(path)))
        assert "standard" not in text.casefold()


def _entry(doc_id: str, group: str, kind: SourceKind, title: str = "Başlık") -> SourceEntry:
    return SourceEntry(
        doc_id=doc_id,
        title=title,
        url=f"https://example.org/{doc_id}",
        local_path=f"{doc_id}.bin",
        kind=kind,
        program_kodu=None,
        doc_turu=DocType.DIGER,
        surum_tarihi=date(2026, 1, 1),
        dil="tr",
        sha256=None,
        kaynak_grubu=group,
        kaynak_sayfa_url=None,
    )


def test_corpus_groups_count_documents_by_source_and_kind() -> None:
    entries = [
        *(_entry(f"mevzuat-{i:03d}", "tubitak_mevzuat", SourceKind.PDF) for i in range(3)),
        *(_entry(f"sayfa-{i}", "tubitak_program_sayfalari", SourceKind.HTML) for i in range(2)),
        _entry("dosya-a", "tubitak_program_sayfalari", SourceKind.PDF),
        _entry("dosya-b", "tubitak_program_sayfalari", SourceKind.XLSX),
        _entry("kanun-5746", "sanayi_mevzuati", SourceKind.GIB_JSON, "Ar-Ge Kanunu"),
    ]
    groups = {g.kaynak_grubu: g for g in corpus_groups(entries)}
    assert groups["tubitak_mevzuat"].belge_sayisi == 3
    assert groups["tubitak_mevzuat"].ozet == "3 PDF"
    assert groups["tubitak_program_sayfalari"].ozet == "2 sayfa + 2 dosya (1 PDF, 1 Excel)"
    assert groups["sanayi_mevzuati"].ozet == "1 kanun"
    assert groups["sanayi_mevzuati"].ayrinti == ["5746 sayılı Ar-Ge Kanunu"]
    assert groups["tubitak_mevzuat"].kaynak_url == "https://tubitak.gov.tr/tr/mevzuat"


def test_component_lines_list_every_model_once(repo_root: Path) -> None:
    configs = [
        load_config(p) for p in sorted((repo_root / "configs" / "experiments").glob("*.yaml"))
    ]
    text = methods_text(component_lines(configs))
    assert "google/embeddinggemma-2" in text and "BAAI/bge-m3" in text
    assert text.count("BAAI/bge-reranker-v2-m3") == 1
    assert "qwen3:4b" in text and "Qdrant" in text
    assert "Türkçe kök bulma" in text and "RRF" in text


def test_component_lines_name_the_numpy_store_of_the_space() -> None:
    cfg = AppConfig().model_copy(update={"vector_store": VectorStoreConfig(provider="numpy")})
    stores = {line.etiket: line.deger for line in component_lines([cfg])}["Vektör deposu"]
    assert stores == "NumPy (bellek içi, salt okunur)"


@pytest.mark.parametrize(
    "fragment",
    ["Kapsam kontrolü", "Arama eşiği", "Atıf doğrulama", "Sayı doğrulama", "atıf onarımı"],
)
def test_guardrail_lines_name_each_guard(fragment: str) -> None:
    assert fragment in methods_text(guardrail_lines(AppConfig()))
