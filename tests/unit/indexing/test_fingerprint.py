"""Tests for chunk digests and per-document index fingerprints."""

from tests.builders import make_chunk
from tubitak_rag.indexing.fingerprint import chunk_digest, doc_fingerprint, fingerprints


def test_chunk_digest_is_stable_hex_sha256() -> None:
    digest = chunk_digest(make_chunk())
    assert digest == chunk_digest(make_chunk())
    assert len(digest) == 64 and int(digest, 16) >= 0


def test_chunk_digest_changes_with_text_or_metadata() -> None:
    base = chunk_digest(make_chunk())
    assert chunk_digest(make_chunk(embed_text="Başlık\n\nDestek oranı")) != base
    assert chunk_digest(make_chunk(hiyerarsi="Dördüncü Bölüm")) != base
    assert chunk_digest(make_chunk(sayfa=7)) != base


def test_doc_fingerprint_ignores_order() -> None:
    assert doc_fingerprint(["b", "a"]) == doc_fingerprint(["a", "b"])
    assert doc_fingerprint(["a"]) != doc_fingerprint(["a", "b"])


def test_same_raw_sha_with_changed_chunks_gives_new_fingerprint() -> None:
    old = [make_chunk("d:0001:00", doc_id="d", text="eski", embed_text="eski")]
    new = [make_chunk("d:0001:00", doc_id="d", text="yeni", embed_text="yeni")]
    assert old[0].doc_sha256 == new[0].doc_sha256
    assert fingerprints({"d": old}) != fingerprints({"d": new})


def test_fingerprints_per_document() -> None:
    chunks = [make_chunk("d:0001:00", doc_id="d"), make_chunk("d:0002:00", doc_id="d", sira=2)]
    assert fingerprints({"d": chunks}) == {
        "d": doc_fingerprint(chunk_digest(c) for c in reversed(chunks))
    }
