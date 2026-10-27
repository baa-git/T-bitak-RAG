"""Content-addressed embedding cache in SQLite: ``(model, sha256(text)) -> float32 vector``.

Because the key is the text itself (hashed), an unchanged chunk is never embedded twice,
even when it moves to another document or the collection is rebuilt (FR-I5, FR-X9).
"""

import hashlib
import sqlite3
from array import array
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from pathlib import Path

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS embeddings("
    "model TEXT, text_hash TEXT, dim INTEGER, vector BLOB, PRIMARY KEY(model, text_hash))"
)
# Stay well below SQLite's limit on "?" parameters per statement.
_MAX_PARAMS = 900


def _to_blob(vector: Sequence[float]) -> bytes:
    return array("f", vector).tobytes()


def _from_blob(blob: bytes) -> list[float]:
    values = array("f")
    values.frombytes(blob)
    return values.tolist()


class EmbeddingCache:
    """Stores document vectors per model; vectors are kept as float32 blobs."""

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    @staticmethod
    def text_hash(text: str) -> str:
        """Return the cache key of a text: hex SHA-256 of its UTF-8 bytes."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_many(self, model: str, texts: Sequence[str]) -> dict[str, list[float]]:
        """Return cached vectors of ``texts`` for ``model``, keyed by text hash (hits only)."""
        hashes = list(dict.fromkeys(self.text_hash(text) for text in texts))
        found: dict[str, list[float]] = {}
        with self._connect() as conn:
            for start in range(0, len(hashes), _MAX_PARAMS):
                batch = hashes[start : start + _MAX_PARAMS]
                marks = ",".join("?" * len(batch))
                rows = conn.execute(
                    "SELECT text_hash, vector FROM embeddings "
                    f"WHERE model = ? AND text_hash IN ({marks})",
                    [model, *batch],
                )
                found.update((text_hash, _from_blob(blob)) for text_hash, blob in rows)
        return found

    def put_many(self, model: str, items: Mapping[str, Sequence[float]]) -> None:
        """Insert or replace vectors of ``model`` keyed by text hash."""
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings(model, text_hash, dim, vector) "
                "VALUES (?, ?, ?, ?)",
                [(model, key, len(vector), _to_blob(vector)) for key, vector in items.items()],
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # One short-lived connection per operation: commits on success and never keeps the
        # file locked (Windows cannot delete an open SQLite file).
        with closing(sqlite3.connect(self._path)) as conn, conn:
            yield conn
