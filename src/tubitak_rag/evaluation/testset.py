"""Evaluation test set: JSONL schema, loading, validation and the human review queue.

The test set (``eval/testset_v1.jsonl``) is the yardstick of every experiment, so it is
checked mechanically before it is trusted:

* schema rules per question type (answerable types carry 1–3 evidence quotes, unanswerable
  types carry none and a fixed expected answer);
* every evidence quote must occur *verbatim* in the parsed text of its document, compared
  after :func:`normalize_for_match` (case, whitespace, typographic punctuation and line-end
  hyphenation are ignored, the words are not).

Retrieval metrics later count a hit when a retrieved chunk contains a quote, which makes the
measurement independent of the chunking strategy (FR-E5). ``normalize_for_match`` lives here
and is reused by ``evaluation/matching.py``.

``write_review_queue`` renders the questions as a Markdown checklist for human review.
"""

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import ValidationError

from tubitak_rag.domain.document import Document, Section
from tubitak_rag.domain.enums import QuestionType, SectionKind
from tubitak_rag.domain.evaluation import TestQuestion
from tubitak_rag.ingestion.normalize import collapse_ws, normalize_text, tr_lower

ANSWERABLE = frozenset(
    {QuestionType.TEK_BELGE, QuestionType.COK_BELGE, QuestionType.SURUM, QuestionType.BELIRSIZ}
)
MULTI_DOCUMENT = frozenset({QuestionType.COK_BELGE, QuestionType.BELIRSIZ})
UNANSWERABLE_ANSWERS: Mapping[QuestionType, str] = {
    QuestionType.CEVAPSIZ: "bilgi bulunamadı",
    QuestionType.KAPSAM_DISI: "kapsam dışı",
}
MAX_QUOTES = 3

_TYPOGRAPHY = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "´": "'",
        "`": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "«": '"',
        "»": '"',
        "–": "-",
        "—": "-",
        "‒": "-",
        "―": "-",
        "−": "-",
        "…": "...",
    }
)
MIN_QUOTE_WORDS = 4  # shorter quotes match too easily; reported as warnings, not errors

_LOWER = "a-zçğıöşüâîû"
_UPPER = "A-ZÇĞİÖŞÜÂÎÛ"
# A line-end hyphen between two letters of the same case is a hyphenation ("proje-\nlerin",
# "TÜBİ-\nTAK"); a change of case keeps the hyphen ("Ar-\nGe").
_SOFT_HYPHEN_RE = re.compile(rf"(?<=[{_LOWER}])-\n(?=[{_LOWER}])|(?<=[{_UPPER}])-\n(?=[{_UPPER}])")
_HARD_HYPHEN_RE = re.compile(r"-[ \t]*\n[ \t]*")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?)\]])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([(\[])\s+")


class TestsetError(ValueError):
    """A test-set file cannot be read (message names the line and the field)."""

    __test__ = False  # not a pytest test class despite the name


def normalize_for_match(text: str) -> str:
    """Canonical form for verbatim-quote matching.

    Unicode-normalises, unifies typographic quotes and dashes, re-joins words hyphenated at a
    line end ("proje-\\nlerin" -> "projelerin", "TÜBİ-\\nTAK" -> "TÜBİTAK"; "Ar-\\nGe" ->
    "Ar-Ge"), lower-cases with Turkish rules, collapses whitespace and removes spaces before
    closing punctuation.

    Deliberately *not* equivalent (the words and digits of a quote must be copied as they
    are): "% 60" vs "%60", "dâhil" vs "dahil", "1.350.000" vs "1350000", "on sekiz" vs "18".
    An all-capitals compound broken at a line end ("AR-\\nGE") is joined like a hyphenation.
    """
    text = normalize_text(text).translate(_TYPOGRAPHY)
    text = _HARD_HYPHEN_RE.sub("-", _SOFT_HYPHEN_RE.sub("", text))
    text = collapse_ws(tr_lower(text))
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return _SPACE_AFTER_OPEN_RE.sub(r"\1", text)


def contains_number(text: str, number: str) -> bool:
    """Whether ``number`` occurs in ``text`` as a whole number, not inside a longer one.

    "2" is found in "aylık %2" but not in "24 ay"; "350" is not found in "1.350.000"; a
    sentence-final period ("30.") does not block a match. Also used by the rule-based
    number check of the judge (``judge.numbers_present``).
    """
    needle = re.escape(normalize_for_match(number))
    pattern = rf"(?<!\d)(?<!\d[.,]){needle}(?![.,]?\d)"
    return re.search(pattern, normalize_for_match(text)) is not None


# --- loading and saving --------------------------------------------------------------------


def load_testset(path: Path) -> list[TestQuestion]:
    """Read a JSONL test set; a bad line raises ``TestsetError`` with its number and field."""
    questions: list[TestQuestion] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            questions.append(TestQuestion.model_validate(json.loads(line)))
        except json.JSONDecodeError as exc:
            raise TestsetError(f"{path.name} satır {number}: geçersiz JSON ({exc.msg})") from exc
        except ValidationError as exc:
            first = exc.errors()[0]
            field = ".".join(str(part) for part in first["loc"]) or "?"
            message = f"{path.name} satır {number}: alan '{field}': {first['msg']}"
            raise TestsetError(message) from exc
    return questions


def save_testset(path: Path, questions: Sequence[TestQuestion]) -> None:
    """Write questions as UTF-8 JSONL, one question per line, Turkish text unescaped."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (json.dumps(q.model_dump(mode="json"), ensure_ascii=False) + "\n" for q in questions)
    path.write_text("".join(lines), encoding="utf-8", newline="\n")


# --- validation ----------------------------------------------------------------------------


def _structure_errors(q: TestQuestion) -> list[str]:
    """Rules that need no corpus: quotes, expected documents and answers per type."""
    errors: list[str] = []
    quote_docs = [quote.doc_id for quote in q.kanit_alintilari]
    if q.tur in ANSWERABLE:
        if not q.kanit_alintilari:
            errors.append("cevaplanabilir türde en az 1 kanıt alıntısı gerekir")
        if len(q.kanit_alintilari) > MAX_QUOTES:
            errors.append(f"en fazla {MAX_QUOTES} kanıt alıntısı olmalı")
        if not q.beklenen_belgeler:
            errors.append("beklenen_belgeler boş olamaz")
        if any(not quote.alinti.strip() for quote in q.kanit_alintilari):
            errors.append("boş kanıt alıntısı")
        for doc_id in dict.fromkeys(quote_docs):
            if q.beklenen_belgeler and doc_id not in q.beklenen_belgeler:
                errors.append(f"alıntı belgesi {doc_id} beklenen_belgeler içinde değil")
        if q.tur in MULTI_DOCUMENT and len(set(quote_docs)) < 2:
            errors.append("bu türde en az iki farklı belgeden alıntı gerekir")
    else:
        expected = UNANSWERABLE_ANSWERS[q.tur]
        if q.kanit_alintilari:
            errors.append("bu türde kanıt alıntısı olmamalı")
        if q.beklenen_belgeler:
            errors.append("bu türde beklenen_belgeler boş olmalı")
        if q.beklenen_cevap != expected:
            errors.append(f"beklenen_cevap '{expected}' olmalı")
    for number in q.beklenen_sayilar:
        if not contains_number(q.beklenen_cevap, number):
            errors.append(f"beklenen sayı '{number}' beklenen_cevap içinde geçmiyor")
    return errors


def _corpus_errors(
    q: TestQuestion, docs: Mapping[str, Document], texts: dict[str, str]
) -> list[str]:
    """Every expected document exists and every quote occurs verbatim in its document."""
    errors = [f"belge yok: {doc_id}" for doc_id in q.beklenen_belgeler if doc_id not in docs]
    for quote in q.kanit_alintilari:
        doc = docs.get(quote.doc_id)
        if doc is None:
            if quote.doc_id in q.beklenen_belgeler:
                continue  # already reported above
            errors.append(f"belge yok: {quote.doc_id}")
            continue
        if quote.doc_id not in texts:
            texts[quote.doc_id] = normalize_for_match(doc.full_text())
        if normalize_for_match(quote.alinti) not in texts[quote.doc_id]:
            errors.append(f"alıntı belgede bulunamadı ({quote.doc_id}): '{quote.alinti[:60]}'")
    return errors


def validate_testset(
    questions: Sequence[TestQuestion], docs: Mapping[str, Document] | None = None
) -> list[str]:
    """Human-readable problems of a test set (empty list = valid); each names the question.

    Without ``docs`` only the structural rules run; with ``docs`` (doc_id -> parsed document)
    every evidence quote is also verified against the parsed text.
    """
    errors: list[str] = []
    seen: set[str] = set()
    texts: dict[str, str] = {}  # normalised full texts, computed once per document
    for q in questions:
        problems = _structure_errors(q)
        if q.id in seen:
            problems.insert(0, "kimlik tekrar ediyor")
        seen.add(q.id)
        if docs is not None:
            problems += _corpus_errors(q, docs, texts)
        errors += [f"{q.id}: {problem}" for problem in problems]
    return errors


def quote_warnings(questions: Sequence[TestQuestion]) -> list[str]:
    """Soft problems that do not invalidate a question: quotes shorter than four words."""
    warnings: list[str] = []
    for q in questions:
        for quote in q.kanit_alintilari:
            words = len(quote.alinti.split())
            if words < MIN_QUOTE_WORDS:
                warnings.append(f"{q.id}: kısa alıntı ({words} kelime): '{quote.alinti}'")
    return warnings


def verify_testset(path: Path, docs: Mapping[str, Document]) -> list[str]:
    """Load a test-set file and validate it against the parsed corpus."""
    return validate_testset(load_testset(path), docs)


def type_distribution(questions: Sequence[TestQuestion]) -> dict[str, int]:
    """Question count per type; every type is present, in enum order."""
    counts = {t.value: 0 for t in QuestionType}
    for q in questions:
        counts[q.tur.value] += 1
    return counts


# --- review queue --------------------------------------------------------------------------


def _section_location(section: Section) -> str:
    parts: list[str] = []
    if section.madde_no:
        parts.append(f"Madde {section.madde_no}")
    elif section.kind == SectionKind.SSS and section.title:
        parts.append(f"SSS: {section.title}")
    elif section.title:
        parts.append(f"Bölüm: {section.title}")
    if section.page_start is not None:
        end = section.page_end
        pages = f"{section.page_start}–{end}" if end and end != section.page_start else ""
        parts.append(f"s. {pages or section.page_start}")
    return ", ".join(parts) or section.section_id


def locate_quote(doc: Document, quote: str) -> str:
    """Article/page/heading of the section that contains ``quote``, for reviewers."""
    needle = normalize_for_match(quote)
    for section in doc.sections:
        if needle in normalize_for_match(section.text):
            return _section_location(section)
    if needle in normalize_for_match(doc.full_text()):
        return "birden fazla bölüme yayılıyor"
    return "konum bulunamadı"


def _type_order(q: TestQuestion) -> tuple[int, str]:
    return list(QuestionType).index(q.tur), q.id


def _batches(ordered: Sequence[TestQuestion], size: int) -> list[list[TestQuestion]]:
    """Consecutive runs of one question type, each split into chunks of ``size``."""
    batches: list[list[TestQuestion]] = []
    for q in ordered:
        if not batches or batches[-1][0].tur != q.tur or len(batches[-1]) == size:
            batches.append([])
        batches[-1].append(q)
    return batches


def _render_question(q: TestQuestion, docs: Mapping[str, Document]) -> list[str]:
    lines = [f"- [ ] **{q.id}** ({q.tur.value}/{q.zorluk}) {q.soru}"]
    lines.append(f"  - Beklenen: {q.beklenen_cevap}")
    if q.beklenen_sayilar:
        lines.append(f"  - Beklenen sayılar: {', '.join(q.beklenen_sayilar)}")
    for quote in q.kanit_alintilari:
        doc = docs.get(quote.doc_id)
        if doc is None:
            where = f"{quote.doc_id}, belge bulunamadı"
        else:
            title = doc.title.strip() or doc.kaynak_url.rstrip("/").rsplit("/", 1)[-1]
            where = f"{title} [{quote.doc_id}], {locate_quote(doc, quote.alinti)}"
        lines.append(f'  - Kanıt: "{quote.alinti}" ({where})')
    if q.review_notes:
        lines.append(f"  - Yazar notu: {q.review_notes}")
    lines.append("  - İnceleme notu: ___")
    return lines


def write_review_queue(
    questions: Sequence[TestQuestion],
    docs: Mapping[str, Document],
    path: Path,
    batch_size: int = 10,
) -> None:
    """Write a Markdown review checklist: questions grouped by type, in batches.

    A batch holds at most ``batch_size`` questions of a single type; a new batch starts at
    each type change, so reviewers can work through one question type at a time.
    """
    ordered = sorted(questions, key=_type_order)
    lines = [
        "# Test seti inceleme kuyruğu",
        "",
        "Her soruyu kaynak belgeye bakarak kontrol et: soru doğal mı, beklenen cevap doğru ve",
        "tam mı, alıntı cevabı gerçekten kanıtlıyor mu? Onaylanan soruda kutuyu işaretle ve",
        "`eval/testset_v1.jsonl` içinde `dogrulandi: true` yap; düzeltme gerekiyorsa",
        "`review_notes` alanına yaz.",
        "",
        "| Tür | Soru sayısı |",
        "|---|---|",
    ]
    lines += [f"| {tur} | {count} |" for tur, count in type_distribution(ordered).items()]
    lines.append(f"| **Toplam** | **{len(ordered)}** |")
    for number, batch in enumerate(_batches(ordered, batch_size), start=1):
        header = f"## Parti {number} — {batch[0].tur.value} ({batch[0].id} … {batch[-1].id})"
        lines += ["", header, ""]
        for q in batch:
            lines += _render_question(q, docs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
