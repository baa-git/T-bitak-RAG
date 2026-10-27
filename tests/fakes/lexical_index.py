"""Term-count lexical index for unit tests (stands in for BM25)."""

import json
import re
from collections.abc import Sequence
from pathlib import Path

from tubitak_rag.domain.chunk import Chunk

_WORD = re.compile(r"\w+")
_FILE_NAME = "fake_lexical_index.json"


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class FakeLexicalIndex:
    """Score = number of occurrences of query terms in the chunk's ``embed_text``."""

    def __init__(self) -> None:
        self._docs: dict[str, list[str]] = {}

    def build(self, chunks: Sequence[Chunk]) -> None:
        self._docs = {chunk.chunk_id: _words(chunk.embed_text) for chunk in chunks}

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / _FILE_NAME).write_text(json.dumps(self._docs), encoding="utf-8")

    def load(self, directory: Path) -> None:
        self._docs = json.loads((directory / _FILE_NAME).read_text(encoding="utf-8"))

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        terms = set(_words(query))
        hits = [
            (chunk_id, float(sum(1 for word in words if word in terms)))
            for chunk_id, words in self._docs.items()
        ]
        hits = [hit for hit in hits if hit[1] > 0]
        hits.sort(key=lambda hit: hit[1], reverse=True)
        return hits[:k]

    def __len__(self) -> int:
        return len(self._docs)
