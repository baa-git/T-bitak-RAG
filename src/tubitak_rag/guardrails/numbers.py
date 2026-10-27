"""Number grounding: every number in an answer must appear in the question or a source.

Why a deterministic check: a small model asked "rate *and* duration?" with sources that only
give the rate wrote "… en fazla 24 aydır [1]" in every run; no prompt wording stopped it (see
the RAG guide, ``docs/RAG-GELISTIRME-REHBERI.md`` §2.4). Numbers are the most harmful thing to
invent in a legislation assistant and the easiest to verify mechanically.

How: numbers are extracted from both sides and normalised to canonical tokens (``"%75"``,
``"yüzde 75"`` → ``"75"``; ``"3,5 milyon"``, ``"3.500.000"`` → ``"3500000"``; dates → ISO plus
year; Turkish number words → digits). Token *sets* are compared, never substrings, so ``75``
is not grounded by ``1750``.

Citation repair: the cited sources are checked first. A number missing from them can be
grounded by another *prompt* source (``sources``, in prompt order), but more strictly: the
answer must write it with a unit and the source's text (or title) must contain the same
``value|unit`` pair (``extract_quantities``: "18 aydır" ↔ "18 ay"); chunk metadata (madde no,
version date, hierarchy) never repairs. The smallest such source number is added to the
sentence's inline marker ("[5]" → "[1, 5]", or " [1]" inserted), to ``atiflar`` and to
``repaired_citations``, so the text and the list cannot diverge. Reason: in a smoke test
``qwen3:4b`` answered "… en fazla 18 aydır [5]" correctly, with "18 ay" in source [1];
small models under-cite, and stripping the correct sentence was a false refusal. The strict
matching matters: bare-token repair would legitimise "3 yıl" by an unrelated "Madde 3",
"2026 yılında" by another chunk's version date and "15 ay" by "15 kişi".

Modes (``guardrails.number_check``): ``off`` (no check, no repair); ``strip`` drops the
sentences with ungrounded numbers and notes the removal in ``eksik_kisim`` (the invented
number is never echoed) — repairs that only served a dropped sentence are dropped too; if no
cited sentence remains the answer is rejected; ``reject`` rejects on any ungrounded number.

Call site (``RagPipeline.ask``): after ``citations.validate`` passes and before the
post-processing steps; ``cited`` are the chunks of the output's citations, ``sources`` all
prompt chunks. A rejected verdict (``reason="dayanaksiz_sayi"``) becomes "bilgi bulunamadı"
with no sources (FR-GR2).
"""

import re
from collections.abc import Callable, Iterable, Sequence
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import BaseModel, ConfigDict

from tubitak_rag.domain.chunk import Chunk
from tubitak_rag.generation.schema import (
    CITE_RE,
    LlmOutput,
    cited_numbers,
    inline_citation_numbers,
)
from tubitak_rag.retrieval.turkish import turkish_lower

NumberCheckMode = Literal["off", "strip", "reject"]

UNGROUNDED_NUMBER = "dayanaksiz_sayi"
STRIPPED_NOTE = "Cevabın sayısal bir ifadesi kaynaklarda doğrulanamadığı için çıkarıldı."

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_LIST_MARKER_RE = re.compile(r"^\s*(?:\(?\d{1,2}[.)]|\(?[a-zçğıöşü]\))\s+", re.MULTILINE)
_MONTHS = {
    name: index
    for index, name in enumerate(
        (
            "ocak",
            "şubat",
            "mart",
            "nisan",
            "mayıs",
            "haziran",
            "temmuz",
            "ağustos",
            "eylül",
            "ekim",
            "kasım",
            "aralık",
        ),
        start=1,
    )
}
_DMY_RE = re.compile(r"(?<!\d)(\d{1,2})[./](\d{1,2})[./](\d{4})(?!\d)")
_ISO_RE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_TEXT_DATE_RE = re.compile(rf"(?<!\d)(\d{{1,2}})\s+({'|'.join(_MONTHS)})\w*\s+(\d{{4}})(?!\d)")
_NUMERAL = r"\d+(?:[.,]\d+)*"
_MULTIPLIERS = {"bin": 1_000, "milyon": 1_000_000, "milyar": 1_000_000_000}
_MULTIPLIED_RE = re.compile(rf"(?<![\d.,])({_NUMERAL})\s*({'|'.join(_MULTIPLIERS)})\b")
_NUMERAL_RE = re.compile(rf"(?<![\d.,])({_NUMERAL})(?!\d)")

_WORD_VALUES = {
    "bir": 1,
    "iki": 2,
    "üç": 3,
    "dört": 4,
    "dörd": 4,  # before a vowel suffix: "yirmi dördü"
    "beş": 5,
    "altı": 6,
    "yedi": 7,
    "sekiz": 8,
    "dokuz": 9,
    "on": 10,
    "yirmi": 20,
    "otuz": 30,
    "kırk": 40,
    "elli": 50,
    "altmış": 60,
    "yetmiş": 70,
    "seksen": 80,
    "doksan": 90,
    "yüz": 100,
    **_MULTIPLIERS,
}
_WORD = "|".join(sorted(_WORD_VALUES, key=len, reverse=True))
# A suffix on the last number word: copula ("beştir") or possessive with an optional case
# ending ("doksan beşi", "ellisini", "beşinde"). Without it "yüzde doksan beşi" would stop
# at "doksan" and read as 90. "yüzde" itself never matches ("de" is not one of these).
_WORD_SUFFIX = r"(?:[dt][ıiuü]r|s?[ıiuü](?:n[ıiuü]|n[ae]|n[dt][ae]n?)?)?"
# Group 1 is the number words without the suffix.
_WORD_RUN_RE = re.compile(rf"\b((?:{_WORD})(?:\s+(?:{_WORD}))*){_WORD_SUFFIX}\b")
_UNITS = r"ay|yıl|gün|tl|lira|yüzde|kişi|personel|kez|puan|saat|adet"
# Answer side: a number word counts only next to a unit ("on sekiz ay", "yüzde yetmiş beş").
_UNIT_AFTER_RE = re.compile(
    rf"\s*(?:{_UNITS})(?:[dt][ıiuü]r|l[ıiuü]k|[dt][ae]n?|[ıiuü]|[ae]|l[ae]r[ıi]?)?\b"
)
_UNIT_BEFORE_RE = re.compile(r"(?:yüzde|%)\s*$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[^\d\s][.!?])\s+|\n+")
_SENTENCE_SPLIT_KEEP_RE = re.compile(f"({_SENTENCE_SPLIT_RE.pattern})")
# Citation repair: the unit word after a number, with common Turkish suffixes
# ("aydır", "yılında", "günü", "kişiye", "TL'dir").
_UNIT_WORD_RE = re.compile(
    rf"\s*({_UNITS})"
    r"(?:[ıiuü]?n?[dt][ae]n?|[dt][ıiuü]r|l[ıiuü]k|l[ae]r[ıi]?(?:n?[dt][ae])?|[ıiuü]n?|y?[ae])?\b"
)
_UNIT_ALIASES = {"lira": "tl"}


class NumberVerdict(BaseModel):
    """Outcome of the number check, with the (possibly shortened) answer to use."""

    model_config = ConfigDict(frozen=True)

    ok: bool
    reason: str | None
    cevap: str
    eksik_kisim: str | None
    atiflar: list[int]
    ungrounded: list[str]  # canonical tokens, for traces and logs only
    repaired_citations: list[int] = []  # source numbers added to atiflar by citation repair


def _canonical(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _parse_numeral(text: str) -> Decimal:
    """Turkish numerals: dots group thousands, comma is the decimal separator."""
    if "," in text:
        try:
            return Decimal(text.replace(".", "").replace(",", "."))
        except InvalidOperation:
            return Decimal(re.sub(r"\D", "", text) or "0")
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", text):
        return Decimal(text.replace(".", ""))
    try:
        return Decimal(text)
    except InvalidOperation:
        # Outline numbers such as "5.2.2" are not quantities; joining the digits maps the same
        # string to the same token on the answer and the source side, so matching stays exact.
        return Decimal(re.sub(r"\D", "", text) or "0")


def _words_value(words: Iterable[str]) -> int:
    total = current = 0
    for word in words:
        value = _WORD_VALUES[word]
        if value == 100:
            current = (current or 1) * 100
        elif value >= 1_000:
            total += (current or 1) * value
            current = 0
        else:
            current += value
    return total + current


def _blank(text: str, start: int, end: int) -> str:
    return text[:start] + " " * (end - start) + text[end:]


def _take(
    pattern: re.Pattern[str], text: str, convert: Callable[[re.Match[str]], set[str]]
) -> tuple[str, set[str]]:
    """Collect tokens for every match of ``pattern`` and blank the matched spans out."""
    tokens: set[str] = set()
    for match in reversed(list(pattern.finditer(text))):
        found = convert(match)
        if found:
            tokens |= found
            text = _blank(text, match.start(), match.end())
    return text, tokens


def _iso(year: str, month: int, day: int) -> set[str]:
    return {f"{year}-{month:02d}-{day:02d}", year}


def extract_numbers(text: str, *, answer: bool) -> set[str]:
    """Canonical number tokens of ``text``; ``answer=True`` applies the stricter answer rules."""
    text = CITE_RE.sub(" ", _URL_RE.sub(" ", text))
    if answer:
        text = _LIST_MARKER_RE.sub(" ", text)
    text = turkish_lower(text)
    tokens: set[str] = set()
    for pattern, convert in (
        (_DMY_RE, lambda m: _iso(m[3], int(m[2]), int(m[1]))),
        (_ISO_RE, lambda m: _iso(m[1], int(m[2]), int(m[3]))),
        (_TEXT_DATE_RE, lambda m: _iso(m[3], _MONTHS[m[2]], int(m[1]))),
        (
            _MULTIPLIED_RE,
            lambda m: {_canonical(_parse_numeral(m[1]) * _MULTIPLIERS[m[2]])},
        ),
    ):
        text, found = _take(pattern, text, convert)
        tokens |= found

    def word_number(match: re.Match[str]) -> set[str]:
        words = match[1].split()
        if answer and not (
            _UNIT_AFTER_RE.match(match.string, match.end())
            or _UNIT_BEFORE_RE.search(match.string, 0, match.start())
        ):
            return set()
        if answer and words == ["bir"]:
            return set()
        return {str(_words_value(words))}

    text, found = _take(_WORD_RUN_RE, text, word_number)
    tokens |= found
    tokens |= {_canonical(_parse_numeral(m[1])) for m in _NUMERAL_RE.finditer(text)}
    return tokens


def _unit_of(match: re.Match[str]) -> str | None:
    after = _UNIT_WORD_RE.match(match.string, match.end())
    if after:
        return _UNIT_ALIASES.get(after[1], after[1])
    if _UNIT_BEFORE_RE.search(match.string, 0, match.start()):
        return "yüzde"
    return None


def extract_quantities(text: str) -> set[str]:
    """``"value|unit"`` tokens for numbers written with a unit ("18 aydır" -> ``"18|ay"``,
    "%75" -> ``"75|yüzde"``, "3,5 milyon TL" -> ``"3500000|tl"``); dates and bare numbers
    (article numbers, list markers) are not quantities."""
    text = turkish_lower(CITE_RE.sub(" ", _URL_RE.sub(" ", text)))
    for date in (_DMY_RE, _ISO_RE, _TEXT_DATE_RE):
        text = date.sub(lambda m: " " * len(m[0]), text)
    tokens: set[str] = set()
    values: tuple[tuple[re.Pattern[str], Callable[[re.Match[str]], Decimal]], ...] = (
        (_MULTIPLIED_RE, lambda m: _parse_numeral(m[1]) * _MULTIPLIERS[m[2]]),
        (_WORD_RUN_RE, lambda m: Decimal(_words_value(m[1].split()))),
        (_NUMERAL_RE, lambda m: _parse_numeral(m[1])),
    )
    for pattern, value in values:

        def quantity(
            match: re.Match[str], value: Callable[[re.Match[str]], Decimal] = value
        ) -> set[str]:
            unit = _unit_of(match)
            return {f"{_canonical(value(match))}|{unit}"} if unit else set()

        text, found = _take(pattern, text, quantity)
        tokens |= found
    return tokens


def grounding_tokens(question: str, cited: Sequence[Chunk]) -> set[str]:
    """Numbers a grounded answer may use: the question plus the cited chunks and metadata."""
    parts = [question]
    for chunk in cited:
        parts += [chunk.text, chunk.doc_title, chunk.hiyerarsi]
        parts += [p for p in (chunk.program_kodu, chunk.surum_tarihi, chunk.madde_no) if p]
    return set().union(*(extract_numbers(part, answer=False) for part in parts))


def _verdict(
    output: LlmOutput, *, ok: bool = True, ungrounded: Iterable[str] = ()
) -> NumberVerdict:
    return NumberVerdict(
        ok=ok,
        reason=None if ok else UNGROUNDED_NUMBER,
        cevap=output.cevap,
        eksik_kisim=output.eksik_kisim,
        atiflar=output.atiflar,
        ungrounded=sorted(ungrounded),
    )


def _repairs(cevap: str, missing: set[str], sources: Sequence[Chunk]) -> dict[str, int]:
    """Number token -> smallest 1-based prompt source whose text holds the same quantities."""
    wanted: dict[str, set[str]] = {}
    for quantity in extract_quantities(cevap):
        token = quantity.partition("|")[0]
        if token in missing:
            wanted.setdefault(token, set()).add(quantity)
    found: dict[str, int] = {}
    for number, chunk in enumerate(sources, start=1):
        have = extract_quantities(chunk.text) | extract_quantities(chunk.doc_title)
        for token, quantities in wanted.items():
            if token not in found and quantities <= have:
                found[token] = number
    return found


def _mark(sentence: str, repairs: dict[str, int]) -> str:
    """Add the repaired source numbers of ``sentence``'s numbers to its last inline marker."""
    used = extract_numbers(sentence, answer=True)
    numbers = {n for token, n in repairs.items() if token in used}
    if not numbers:
        return sentence
    markers = list(CITE_RE.finditer(sentence))
    if markers:
        last = markers[-1]
        merged = ", ".join(map(str, sorted(inline_citation_numbers(last[0]) | numbers)))
        return f"{sentence[: last.start()]}[{merged}]{sentence[last.end() :]}"
    body = sentence.rstrip()
    end = len(body) - 1 if body and body[-1] in ".!?" else len(body)
    marker = ", ".join(map(str, sorted(numbers)))
    return f"{body[:end]} [{marker}]{body[end:]}{sentence[len(body) :]}"


def check_numbers(
    output: LlmOutput,
    question: str,
    cited: Sequence[Chunk],
    *,
    mode: NumberCheckMode = "strip",
    sources: Sequence[Chunk] = (),
) -> NumberVerdict:
    """Check that every number of an answerable output is grounded (see module docstring)."""
    if mode == "off" or not output.cevaplanabilir:
        return _verdict(output)
    grounded = grounding_tokens(question, cited)
    missing = extract_numbers(output.cevap, answer=True) - grounded
    repairs = _repairs(output.cevap, missing, sources)
    ungrounded = missing - set(repairs)
    before = cited_numbers(output)
    if not ungrounded:
        pieces = _SENTENCE_SPLIT_KEEP_RE.split(output.cevap)
        cevap = "".join(
            piece if index % 2 else _mark(piece, repairs) for index, piece in enumerate(pieces)
        )
        atiflar = inline_citation_numbers(cevap) | set(output.atiflar)
        return NumberVerdict(
            ok=True,
            reason=None,
            cevap=cevap,
            eksik_kisim=output.eksik_kisim,
            atiflar=sorted(atiflar),
            ungrounded=[],
            repaired_citations=sorted(atiflar - before),
        )
    if mode == "reject":
        return _verdict(output, ok=False, ungrounded=ungrounded)
    kept = [
        _mark(sentence, repairs)
        for sentence in _SENTENCE_SPLIT_RE.split(output.cevap.strip())
        if sentence and not extract_numbers(sentence, answer=True) & ungrounded
    ]
    cevap = " ".join(kept)
    atiflar = inline_citation_numbers(cevap)
    if not CITE_RE.search(output.cevap):  # cited only through the list: keep the list
        atiflar |= set(output.atiflar)
    if not cevap.strip() or not atiflar:
        return _verdict(output, ok=False, ungrounded=ungrounded)
    note = f"{output.eksik_kisim} {STRIPPED_NOTE}" if output.eksik_kisim else STRIPPED_NOTE
    return NumberVerdict(
        ok=True,
        reason=None,
        cevap=cevap,
        eksik_kisim=note,
        atiflar=sorted(atiflar),
        ungrounded=sorted(ungrounded),
        repaired_citations=sorted(atiflar - before),
    )
