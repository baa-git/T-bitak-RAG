"""Turkish text processing for keyword search: lower-casing, tokens, stopwords, stems.

Used for both sides of BM25 (chunks at index time, questions at query time), so the same
word always becomes the same token, e.g. "Projelerin" and "projeler" both become "proje".
"""

import re
from typing import Any

from tubitak_rag.ingestion.normalize import tr_lower

# Circumflex vowels are optional in modern spelling ("hâlinde" == "halinde").
_CIRCUMFLEX = str.maketrans({"â": "a", "î": "i", "û": "u"})

TOKEN_RE = re.compile(r"[0-9a-zçğıöşüâîû]+(?:[./-][0-9a-zçğıöşüâîû]+)*")

# "az" is kept on purpose: "en az" (at least) vs "en fazla" (at most) matters in regulations.
# fmt: off
STOPWORDS_TR: frozenset[str] = frozenset({
    "acaba", "ama", "ancak", "aslında", "bazı", "belki", "biri", "birkaç", "birşey", "biz",
    "bu", "bunu", "buna", "bunda", "bundan", "bunlar", "bunların", "çok", "çünkü", "da", "daha",
    "de", "defa", "diye", "dolayı", "eğer", "en", "gibi", "hem", "hep", "hepsi", "her", "hiç",
    "için", "ile", "ise", "kadar", "ki", "kim", "kimi", "mı", "mi", "mu", "mü", "nasıl", "ne",
    "neden", "nerede", "nereye", "niçin", "niye", "o", "olan", "olarak", "oldu", "olduğu", "olur",
    "şey", "şu", "şunu", "tüm", "ve", "veya", "ya", "yani", "yine", "yoksa", "bir", "diğer",
})
# fmt: on


def turkish_lower(s: str) -> str:
    """Lower-case with Turkish rules: ``"İ" -> "i"``, ``"I" -> "ı"`` (FR-R2).

    The same rule as the parsers' :func:`~tubitak_rag.ingestion.normalize.tr_lower`
    (``str.lower`` maps "I" to "i" and "İ" to "i" plus a combining dot).
    """
    return tr_lower(s)


class TurkishTokenizer:
    """``text -> tokens``: lower-case, split, drop stopwords and 1-letter words, stem.

    Stemming uses the Snowball Turkish stemmer (FR-R3). Tokens with digits ("1507",
    "5.1") are kept as they are; program codes and amounts must match exactly.
    """

    def __init__(self, stemming: bool = True) -> None:
        self.stemming = stemming
        self._stems: dict[str, str] = {}  # the vocabulary is small; stemming is pure Python
        self._stemmer: Any = None  # snowballstemmer has no type hints
        if stemming:
            import snowballstemmer

            self._stemmer = snowballstemmer.stemmer("turkish")

    def __call__(self, text: str) -> list[str]:
        """Return the tokens of ``text`` in order (duplicates kept for term frequency)."""
        words = TOKEN_RE.findall(turkish_lower(text))
        return [
            self._stem(word.translate(_CIRCUMFLEX))
            for word in words
            if word not in STOPWORDS_TR and (len(word) > 1 or word.isdigit())
        ]

    def _stem(self, word: str) -> str:
        if self._stemmer is None or any(ch.isdigit() for ch in word):
            return word
        if word not in self._stems:
            self._stems[word] = self._stemmer.stemWord(word)
        return self._stems[word]
