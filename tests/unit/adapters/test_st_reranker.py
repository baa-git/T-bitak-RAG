"""Tests for the cross-encoder reranker with a fake ``sentence_transformers`` module."""

import sys
import types
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from tests.builders import make_chunk, make_scored
from tubitak_rag.adapters.st_reranker import CrossEncoderReranker
from tubitak_rag.domain.retrieval import ScoredChunk
from tubitak_rag.ports import Reranker

# Logit per passage; anything unknown scores -10.
_LOGITS = {"madde 7": 4.5, "madde 8": -1.25, "madde 9": 0.5}


class _FakeCrossEncoder:
    """Records constructor and ``predict`` arguments; scores passages from ``_LOGITS``."""

    instances: list["_FakeCrossEncoder"] = []

    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.kwargs = kwargs
        self.predict_calls: list[tuple[list[tuple[str, str]], dict[str, Any]]] = []
        _FakeCrossEncoder.instances.append(self)

    def predict(self, pairs: Sequence[tuple[str, str]], **kwargs: Any) -> np.ndarray:
        self.predict_calls.append((list(pairs), kwargs))
        return np.array([_LOGITS.get(p.split("\n")[-1], -10.0) for _, p in pairs], np.float32)


@pytest.fixture(autouse=True)
def fake_st(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = _FakeCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    _FakeCrossEncoder.instances = []


def _model() -> _FakeCrossEncoder:
    return _FakeCrossEncoder.instances[-1]


def _candidates() -> list[ScoredChunk]:
    return [
        make_scored(
            make_chunk(f"c{n}", text=f"madde {n}", embed_text=f"1507 UE › Madde {n}\nmadde {n}"),
            0.1 * n,
            dense=0.1 * n,
            rrf=0.01 * n,
        )
        for n in (8, 9, 7)
    ]


def test_is_a_reranker_with_model_name() -> None:
    reranker = CrossEncoderReranker("BAAI/bge-reranker-v2-m3", device="cpu")
    assert isinstance(reranker, Reranker)
    assert reranker.model_name == "BAAI/bge-reranker-v2-m3"


def test_loads_model_with_device_max_length_and_raw_logits() -> None:
    CrossEncoderReranker("BAAI/bge-reranker-v2-m3", device="cpu", max_length=1024)
    model = _model()
    assert model.name == "BAAI/bge-reranker-v2-m3"
    assert model.kwargs["device"] == "cpu"
    assert model.kwargs["max_length"] == 1024
    # sentence-transformers applies a sigmoid to 1-label models unless told otherwise.
    activation = model.kwargs["activation_fn"]
    assert activation(np.array([-3.0, 2.5])).tolist() == [-3.0, 2.5]


def test_st_reranker_uses_predict_pairs_and_keeps_scores() -> None:
    reranker = CrossEncoderReranker(device="cpu", batch_size=4)
    results = reranker.rerank("1507 destek oranı", _candidates(), top_n=2)
    (pairs, kwargs), *_ = _model().predict_calls
    assert pairs == [("1507 destek oranı", c.chunk.embed_text) for c in _candidates()]
    assert kwargs["batch_size"] == 4
    assert [r.chunk.chunk_id for r in results] == ["c7", "c9"]
    assert results[0].score == 4.5
    assert results[0].scores == {"dense": pytest.approx(0.7), "rrf": 0.07, "rerank": 4.5}
    assert all(isinstance(r.scores["rerank"], float) for r in results)


def test_empty_candidates_skip_the_model() -> None:
    reranker = CrossEncoderReranker(device="cpu")
    assert reranker.rerank("q", [], top_n=5) == []
    assert _model().predict_calls == []


def test_auto_device_is_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tubitak_rag.adapters.st_reranker.resolve_device", lambda _d: "cuda")
    CrossEncoderReranker()
    assert _model().kwargs["device"] == "cuda"
