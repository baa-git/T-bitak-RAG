"""Tests for the dense (embedding) retriever (FR-R1, FR-R5)."""

from collections.abc import Sequence

from tests.builders import make_chunk
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.vector_store import InMemoryVectorStore
from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.domain.retrieval import MetadataFilter, ScoredChunk
from tubitak_rag.ports import Retriever
from tubitak_rag.retrieval.dense import DenseRetriever

COLLECTION = "structural_ctx__fake"


class _RecordingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return super().embed_query(text)


class _RecordingStore(InMemoryVectorStore):
    def __init__(self) -> None:
        super().__init__()
        self.search_calls: list[tuple[str, int, MetadataFilter | None]] = []

    def search(
        self,
        name: str,
        vector: Sequence[float],
        k: int,
        filters: MetadataFilter | None = None,
    ) -> list[ScoredChunk]:
        self.search_calls.append((name, k, filters))
        return super().search(name, vector, k, filters)


def _chunks() -> list[Chunk]:
    return [
        make_chunk("a", text="destek oranı", program_kodu="1507"),
        make_chunk("b", text="bütçe sınırı", program_kodu="1501"),
        make_chunk("c", text="proje süresi", program_kodu="1507"),
        make_chunk("d", text="başvuru koşulları", program_kodu="1501"),
    ]


def _setup() -> tuple[_RecordingEmbedder, _RecordingStore, DenseRetriever]:
    embedder, store = _RecordingEmbedder(), _RecordingStore()
    chunks = _chunks()
    store.ensure_collection(COLLECTION, embedder.dimension)
    store.upsert(COLLECTION, chunks, embedder.embed_documents([c.embed_text for c in chunks]))
    return embedder, store, DenseRetriever(embedder, store, COLLECTION)


def test_is_a_retriever_named_dense() -> None:
    _, _, retriever = _setup()
    assert isinstance(retriever, Retriever)
    assert retriever.name == "dense"


def test_returns_k_results_sorted_desc() -> None:
    _, _, retriever = _setup()
    results = retriever.retrieve("destek oranı", k=3)
    assert len(results) == 3
    assert results[0].chunk.chunk_id == "a"  # identical text -> cosine 1.0
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_scores_dict_has_dense() -> None:
    _, _, retriever = _setup()
    for result in retriever.retrieve("destek oranı", k=4):
        assert result.scores == {"dense": result.score}


def test_query_is_normalized_before_embedding() -> None:
    embedder, _, retriever = _setup()
    retriever.retrieve("  destek  oranı\n", k=1)
    assert embedder.queries == ["destek oranı"]


def test_filter_passed_to_store() -> None:
    _, store, retriever = _setup()
    only_1507 = MetadataFilter(program_kodu="1507")
    results = retriever.retrieve("bütçe sınırı", k=4, filters=only_1507)
    assert store.search_calls == [(COLLECTION, 4, only_1507)]
    assert {r.chunk.chunk_id for r in results} == {"a", "c"}


def test_k_larger_than_index() -> None:
    _, _, retriever = _setup()
    assert len(retriever.retrieve("destek", k=50)) == 4
