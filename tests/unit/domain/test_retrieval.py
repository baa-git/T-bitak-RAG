"""Tests for retrieval result models."""

from tests.builders import make_chunk, make_scored
from tubitak_rag.domain.retrieval import MetadataFilter, ScoredChunk


def test_metadata_filter_defaults_to_no_constraints() -> None:
    flt = MetadataFilter()
    assert flt.program_kodu is None and flt.doc_turu is None


def test_scored_chunk_keeps_component_scores() -> None:
    scored = make_scored(make_chunk(), 0.9, dense=0.9, bm25=3.2)
    assert scored.scores == {"dense": 0.9, "bm25": 3.2}
    assert ScoredChunk.model_validate_json(scored.model_dump_json()) == scored
