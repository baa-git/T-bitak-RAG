"""Evidence matching: did retrieval bring back the passages that prove the answer? (FR-E5)

A retrieved chunk is a hit for an evidence quote when it comes from the quote's document and
contains the whole quote (after :func:`normalize_for_match`). Counting quotes instead of
chunk ids keeps the measurement independent of the chunking strategy: a fixed 512-token
window and a structural "Madde 7" chunk are judged by the same question, "is the proof in
there?". A quote split over two chunks is a miss on purpose — the generator would then see
only half of the proof in any single source.
"""

from collections.abc import Sequence

from tubitak_rag.domain.evaluation import TestQuestion
from tubitak_rag.domain.retrieval import ScoredChunk
from tubitak_rag.evaluation.testset import normalize_for_match


def quote_in_chunk(quote: str, chunk_text: str) -> bool:
    """Whether the whole ``quote`` occurs in ``chunk_text`` (normalised substring)."""
    needle = normalize_for_match(quote)
    return bool(needle) and needle in normalize_for_match(chunk_text)


def evidence_hits(question: TestQuestion, retrieved: Sequence[ScoredChunk]) -> list[int | None]:
    """1-based rank of the first retrieved chunk holding each quote, or None if none does."""
    texts = [(r.chunk.doc_id, normalize_for_match(r.chunk.text)) for r in retrieved]
    hits: list[int | None] = []
    for quote in question.kanit_alintilari:
        needle = normalize_for_match(quote.alinti)
        rank = next(
            (
                position
                for position, (doc_id, text) in enumerate(texts, start=1)
                if doc_id == quote.doc_id and needle and needle in text
            ),
            None,
        )
        hits.append(rank)
    return hits


def first_hit_rank(hits: Sequence[int | None]) -> int | None:
    """Best (lowest) rank over all quotes, or None when no quote was found."""
    ranks = [rank for rank in hits if rank is not None]
    return min(ranks) if ranks else None


def doc_hit(question: TestQuestion, retrieved: Sequence[ScoredChunk], k: int) -> bool:
    """Whether any expected document appears among the first ``k`` retrieved chunks."""
    expected = set(question.beklenen_belgeler)
    return any(r.chunk.doc_id in expected for r in retrieved[:k])
