"""Enumerations shared across the domain (values are the Turkish keys fixed by the spec)."""

from enum import StrEnum


class SourceKind(StrEnum):
    """File format of a raw source document."""

    PDF = "pdf"
    HTML = "html"
    GIB_JSON = "gib_json"
    XLSX = "xlsx"


class DocType(StrEnum):
    """Legal/organisational type of a document."""

    KANUN = "kanun"
    YONETMELIK = "yonetmelik"
    UYGULAMA_ESASLARI = "uygulama_esaslari"
    YONERGE = "yonerge"
    ESASLAR = "esaslar"
    PROGRAM_SAYFASI = "program_sayfasi"
    CAGRI = "cagri"
    KILAVUZ = "kilavuz"
    TABLO = "tablo"
    DIGER = "diger"


class SectionKind(StrEnum):
    """Structural unit produced by a parser."""

    MADDE = "madde"
    BOLUM = "bolum"
    SSS = "sss"
    PARAGRAF = "paragraf"
    TABLO = "tablo"


class QuestionType(StrEnum):
    """Category of an evaluation question."""

    TEK_BELGE = "tek_belge"
    COK_BELGE = "cok_belge"
    CEVAPSIZ = "cevapsiz"
    KAPSAM_DISI = "kapsam_disi"
    SURUM = "surum"
    BELIRSIZ = "belirsiz"


class AnswerStatus(StrEnum):
    """Outcome of a pipeline question."""

    CEVAPLANDI = "cevaplandi"
    BILGI_BULUNAMADI = "bilgi_bulunamadi"
    KAPSAM_DISI = "kapsam_disi"
    ATIF_REDDI = "atif_reddi"
