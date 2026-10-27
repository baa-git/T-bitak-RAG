"""Tests for the one-paragraph descriptions of the experiment ladder (FR-X3)."""

from pathlib import Path

import yaml

from tubitak_rag.reporting.concepts import CONCEPTS, DEFAULT_CONCEPT, concept_for

KEYS = {
    "fixed_chunking",
    "reranker",
    "structural_context",
    "hybrid_rrf",
    "full_pipeline",
    "embedding_comparison",
}


def test_concepts_cover_all_experiment_keys(repo_root: Path) -> None:
    assert set(CONCEPTS) == KEYS
    for path in (repo_root / "configs" / "experiments").glob("*.yaml"):
        concept = yaml.safe_load(path.read_text(encoding="utf-8"))["experiment"]["concept"]
        assert concept in CONCEPTS, path.name


def test_summaries_are_one_short_turkish_paragraph() -> None:
    for concept in CONCEPTS.values():
        words = len(concept.summary.split())
        assert 30 <= words <= 90, (concept.key, words)
        assert "\n" not in concept.summary
        assert any(ch in concept.summary for ch in "çğıöşüİ")
        assert "endüstri" not in concept.summary.lower()


def test_unknown_concept_falls_back_to_full_pipeline() -> None:
    assert concept_for("yok-boyle-bir-kavram").key == DEFAULT_CONCEPT
    assert concept_for("reranker").key == "reranker"
