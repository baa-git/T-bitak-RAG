"""Answer post-processing: announcement link (FR-GR6) and missing part (FR-GR5).

The disclaimer is not added to the text; it travels in ``Answer.uyari`` (FR-G5).
"""

import re
from collections.abc import Sequence

from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.domain.enums import DocType
from tubitak_rag.retrieval.turkish import turkish_lower

# Matched against Turkish-lower-cased chunk text: "… çağrı duyurusunda belirtilir".
ANNOUNCEMENT_RE = re.compile(
    r"çağrı (?:duyurusunda|duyurularında|metninde|dokümanında) (?:belirtil|yer al|açıklan)"
)


def attach_announcement_link(text: str, cited: Sequence[Chunk]) -> str:
    """Append the announcement page link when a cited source defers to the call text.

    The link is the cited programme page if there is one (it lists the open calls), otherwise
    the URL of the deferring source itself.
    """
    deferring = [c for c in cited if ANNOUNCEMENT_RE.search(turkish_lower(c.text))]
    if not deferring:
        return text
    pages = [c for c in cited if c.doc_turu == DocType.PROGRAM_SAYFASI]
    url = (pages or deferring)[0].kaynak_url
    if url in text:
        return text
    return f"{text}\n\nİlgili duyuru/çağrı sayfası: {url}"


def append_missing_part(text: str, eksik_kisim: str | None) -> str:
    """State the part of the question the sources could not answer (FR-GR5)."""
    if not eksik_kisim or not eksik_kisim.strip():
        return text
    return f"{text}\n\nCevaplanamayan kısım: {eksik_kisim.strip()}"
