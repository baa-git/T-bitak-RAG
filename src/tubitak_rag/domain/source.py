"""Source manifest entry: one raw document to fetch, parse and index."""

from datetime import date

from tubitak_rag.domain.base import DomainModel
from tubitak_rag.domain.enums import DocType, SourceKind


class SourceEntry(DomainModel):
    """A single document listed in ``data/manifests``."""

    doc_id: str  # "mevzuat-031" | "sayfa-<slug>" | "dosya-<sha256[:12]>" | "kanun-5746"
    title: str
    url: str
    local_path: str  # relative to data/raw
    kind: SourceKind
    program_kodu: str | None
    doc_turu: DocType
    surum_tarihi: date | None
    dil: str  # "tr" | "en"
    sha256: str | None  # expected digest from the manifest, if known
    kaynak_grubu: str  # "tubitak_mevzuat" | "tubitak_program_sayfalari" | "sanayi_mevzuati"
    kaynak_sayfa_url: str | None
