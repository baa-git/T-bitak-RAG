"""Rebuild and persist the lexical (BM25) index after every ingest (FR-I6).

BM25 statistics (document frequencies, average length) depend on the whole corpus, so the
index is rebuilt from all chunks of the collection instead of being patched per document.
"""

from collections.abc import Sequence
from pathlib import Path

from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.ports import LexicalIndex


def rebuild_bm25(chunks: Sequence[Chunk], index: LexicalIndex, directory: Path) -> int:
    """Build ``index`` from ``chunks``, save it to ``directory`` and return its size."""
    index.build(chunks)
    index.save(directory)
    return len(index)


def load_bm25(index: LexicalIndex, directory: Path) -> LexicalIndex:
    """Load the index saved in ``directory`` into ``index`` and return it."""
    index.load(directory)
    return index
