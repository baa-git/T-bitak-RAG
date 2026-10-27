"""Text normalisation shared by every parser (FR-P2).

``normalize_text`` makes text *comparable* without changing what it says: Unicode NFC,
removal of combining marks that survive NFC, invisible characters and repeated spaces.
``repair_legacy_glyphs`` fixes PDFs typeset with old Turkish fonts whose glyphs map to the
wrong Unicode characters ("TÜB‹TAK taraf›ndan" instead of "TÜBİTAK tarafından").
"""

import re
import unicodedata

_INVISIBLE = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\u2060\ufeff"))
_SPACES_RE = re.compile(r"[ \t\f\v\u00a0\u2000-\u200a\u202f\u205f\u3000]+")
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """NFC-normalise, drop stray combining marks/invisible chars, tidy spaces, keep newlines."""
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.translate(_INVISIBLE)
    lines = (_SPACES_RE.sub(" ", line).strip() for line in text.split("\n"))
    return "\n".join(lines).strip("\n")


def collapse_ws(text: str) -> str:
    """Collapse every run of whitespace (including newlines) to one space."""
    return _WS_RE.sub(" ", text).strip()


def tr_lower(text: str) -> str:
    """Lower-case with Turkish rules: I -> ı and İ -> i (``str.lower`` gets both wrong)."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def tr_title(text: str) -> str:
    """Title-case each word with Turkish rules ("ON İKİNCİ" -> "On İkinci")."""
    return " ".join(word[:1] + tr_lower(word[1:]) for word in text.split())


# --- legacy font repair ---------------------------------------------------------------

# Old Macintosh Turkish fonts: the PDF maps Turkish glyphs to these code points.
_MAC_SIGNAL_RE = re.compile(r"[‹¤⁄]|\w›|›\w")
_MAC_SINGLE = str.maketrans({"›": "ı", "‹": "İ", "¤": "ğ", "⁄": "Ğ", "ß": "ş", "Þ": "Ş"})
_CAPITAL = "A-ZÇĞİÖŞÜ‹⁄"
_MAC_FI_RE = re.compile(rf"(?<=[{_CAPITAL}])fi|fi(?=[{_CAPITAL}])")
# Fonts that put Ġ/ġ/Ģ (Maltese letters, never used in Turkish) where İ/Ş/ş belong.
_G_DOT_SIGNAL = frozenset("ĠġĢ")
_G_DOT = str.maketrans({"Ġ": "İ", "ġ": "Ş", "Ģ": "ş"})


def repair_legacy_glyphs(text: str) -> str:
    """Undo wrong glyph mappings of legacy Turkish fonts; other text is returned unchanged."""
    if _MAC_SIGNAL_RE.search(text):
        text = _MAC_FI_RE.sub("Ş", text.replace("fl", "ş")).translate(_MAC_SINGLE)
    if _G_DOT_SIGNAL & set(text):
        text = text.translate(_G_DOT)
    return text


_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:])")


def tidy_punctuation(text: str) -> str:
    """Remove the space left before punctuation after text was cut out ("x ." -> "x.")."""
    return _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
