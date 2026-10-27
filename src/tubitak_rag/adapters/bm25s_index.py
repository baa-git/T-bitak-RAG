"""Lexical index on ``bm25s`` with the Turkish tokenizer; persisted to a directory.

Directory layout (FR-I6)::

    <dir>/bm25s/          bm25s sparse score matrix + vocabulary
    <dir>/chunk_ids.json  row i of the matrix -> chunk_id
    <dir>/meta.json       {"k1", "b", "stemming", "n"}
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.retrieval.turkish import TurkishTokenizer

_BM25S_DIR = "bm25s"
_IDS_FILE = "chunk_ids.json"
_META_FILE = "meta.json"


class Bm25sIndex:
    """:class:`~tubitak_rag.ports.LexicalIndex` scoring ``embed_text`` with Okapi BM25."""

    def __init__(self, tokenizer: TurkishTokenizer, k1: float = 1.5, b: float = 0.75) -> None:
        import bm25s

        self._bm25s = bm25s
        self._tokenizer = tokenizer
        self._k1 = k1
        self._b = b
        self._ids: list[str] = []
        self._retriever: Any = None  # bm25s.BM25; None while the index is empty

    def build(self, chunks: Sequence[Chunk]) -> None:
        """Index ``chunks`` from scratch (the previous contents are discarded)."""
        self._ids = [chunk.chunk_id for chunk in chunks]
        self._retriever = None
        if not chunks:  # bm25s cannot index an empty corpus
            return
        self._retriever = self._bm25s.BM25(k1=self._k1, b=self._b)
        corpus_tokens = [self._tokenizer(chunk.embed_text) for chunk in chunks]
        self._retriever.index(corpus_tokens, show_progress=False)

    def save(self, directory: Path) -> None:
        """Write the index, the row -> chunk id list and the parameters to ``directory``."""
        directory.mkdir(parents=True, exist_ok=True)
        if self._retriever is not None:
            self._retriever.save(str(directory / _BM25S_DIR))
        meta = {"k1": self._k1, "b": self._b, "stemming": self._tokenizer.stemming}
        _write_json(directory / _IDS_FILE, self._ids)
        _write_json(directory / _META_FILE, {**meta, "n": len(self._ids)})

    def load(self, directory: Path) -> None:
        """Load an index written by :meth:`save` with the same stemming setting."""
        meta = json.loads((directory / _META_FILE).read_text(encoding="utf-8"))
        if meta["stemming"] != self._tokenizer.stemming:
            raise ValueError(
                f"BM25 indeksi kök bulma={meta['stemming']} ile kurulmuş, tokenizer "
                f"kök bulma={self._tokenizer.stemming}; indeksi yeniden kurun ({directory})."
            )
        self._k1, self._b = meta["k1"], meta["b"]
        self._ids = json.loads((directory / _IDS_FILE).read_text(encoding="utf-8"))
        self._retriever = self._bm25s.BM25.load(str(directory / _BM25S_DIR)) if self._ids else None

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(chunk_id, score)`` pairs with a positive score, best first."""
        tokens = self._tokenizer(query)
        if self._retriever is None or not tokens:
            return []
        docs, scores = self._retriever.retrieve(
            [tokens], k=min(k, len(self._ids)), show_progress=False
        )
        return [
            (self._ids[int(row)], float(score))
            for row, score in zip(docs[0], scores[0], strict=True)
            if score > 0
        ]

    def __len__(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._ids)


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
