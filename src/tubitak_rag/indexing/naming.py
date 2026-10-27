"""Names derived from configuration: vector collections and BM25 directories (FR-I1).

A collection holds vectors of one chunker *and* one embedding model, so both are part of
its name; switching either one never mixes incompatible vectors.
"""

import re
from pathlib import Path

_NOT_SLUG = re.compile(r"[^a-z0-9-]")


def slug(s: str) -> str:
    """Lowercase ``s``, turn ``/`` into ``-`` and any other unsafe character into ``_``."""
    return _NOT_SLUG.sub("_", s.lower().replace("/", "-"))


def collection_name(chunker_key: str, embedder_model: str) -> str:
    """Return the vector collection name for a chunker/embedding-model pair."""
    return f"{chunker_key}__{slug(embedder_model)}"


def bm25_dir(artifacts_dir: Path, chunker_key: str) -> Path:
    """Return the BM25 index directory; BM25 depends on the chunks, not on the embedder."""
    return artifacts_dir / "cache" / "bm25" / chunker_key
