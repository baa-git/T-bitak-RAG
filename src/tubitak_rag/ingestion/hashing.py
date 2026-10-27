"""SHA-256 of files: the identity of a corpus file version (fetch, parsers, caches)."""

import hashlib
from pathlib import Path

_BLOCK = 1 << 20  # 1 MiB


def sha256_of(path: Path) -> str:
    """Hex SHA-256 digest of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()
