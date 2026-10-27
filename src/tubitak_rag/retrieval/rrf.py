"""Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, 2009) — the project's own code (FR-R4).

RRF fuses ranked lists by rank alone, so scores on different scales (cosine similarity and
BM25) never have to be compared: ``score(id) = Σ_r 1 / (k + rank_r(id))`` with 1-based ranks.
"""

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Fuse ``rankings`` (ids, best first) into ``(id, score)`` pairs, best first.

    Ties keep the order in which the ids were first seen (first ranking first). ``k`` must
    be positive; larger values flatten the advantage of top ranks.
    """
    if k <= 0:
        raise ValueError(f"RRF k pozitif olmalıdır, verilen: {k}")
    scores: dict[str, float] = {}  # insertion order = first-seen order
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    # sorted() is stable, so equal scores stay in first-seen order.
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
