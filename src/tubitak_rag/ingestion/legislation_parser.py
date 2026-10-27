"""Legislation parser: KISIM/BÖLÜM hierarchy, articles (madde), clauses and amendments.

Input is the cleaned page text of a regulation PDF. The parser walks the lines once:

* ``BİRİNCİ KISIM`` / ``İKİNCİ BÖLÜM`` lines open a hierarchy level; the next line is its name.
* ``MADDE 7 –`` starts an article; the short line just before it is the article title
  ("Destek oranı ve sınırı").
* ``(1)``, ``a)``, ``1)`` split an article into fıkra / bent / alt bent paragraphs.
* ``(Değişik: …)``, ``(Ek: …)``, ``(Mülga: …)`` notes are moved out of the text into
  ``amendment_notes`` (FR-P5); an article whose text is only a Mülga note is ``repealed``.
* After the last article, ``EK-1`` / ``EK-I`` annexes become separate sections and the
  amendment-history table ("Kabul edildiği Bilim Kurulu Kararı | Yürürlüğe Giriş Tarihi …")
  is dropped together with its capitalised title ("… ESASLARINA İLİŞKİN CETVEL").
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from tubitak_rag.domain.document import Paragraph, Section
from tubitak_rag.domain.enums import SectionKind
from tubitak_rag.ingestion.document_builder import join_lines, make_section
from tubitak_rag.ingestion.normalize import collapse_ws, tidy_punctuation, tr_lower, tr_title
from tubitak_rag.ingestion.pdf_text import PageText

ARTICLE_RE = re.compile(
    r"^\s*(?:(?P<prefix>GEÇİCİ|Geçici|EK|Ek)\s+)?(?:MADDE|Madde)\s*"
    r"(?P<no>\d+(?:/[A-ZÇĞİÖŞÜ])?)\s*(?:[-–—]|\.(?!\d))\s*(?P<rest>.*)$"
)
_ORDINALS = "BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ|ALTINCI|YEDİNCİ|SEKİZİNCİ|DOKUZUNCU"
HIER_RE = re.compile(
    rf"^\s*(?:(?P<ord>(?:(?:ON|YİRMİ)\s?)?(?:{_ORDINALS})|ONUNCU|YİRMİNCİ)\s+(?P<lvl>KISIM|BÖLÜM)"
    r"|(?P<lvl2>KISIM|BÖLÜM)\s+(?P<roman>[IVXL]+))\s*$"
)
# fıkra "(1)": the clause text starts with a capital, digit, quote or parenthesis, so a wrapped
# "… en az bir" + "(1) ay …" line is not mistaken for a new clause.
PARA_RE = re.compile(r"^\s*\((?P<n>\d{1,2})\)\s*(?=[A-ZÇĞİÖŞÜ0-9(“\"‘'\[]|$)")
BENT_RE = re.compile(r"^\s*(?P<l>[a-zçğıöşü]{1,3})\)\s")  # bent "a)", "ğ)", "ıı)"
ALT_BENT_RE = re.compile(r"^\s*(?P<n>\d{1,2})\)\s")  # alt bent "1)"
# Header lines of the amendment-history table, matched on the Turkish-lower-cased line.
HISTORY_RE = re.compile(
    r"^(?:değişikliğin\s+)?kabul\s+edil(?:diği|en)\s+(?:bilim|yönetim|başkanlık)"
    r"|^(?:\S+\s+){0,3}değişiklik\s+yapılan\s+(?:bilim|yönetim|başkanlık)"
    r"|yürürlüğe\s+giriş\s+tarihi$|yürürlük\s+tarihi$"
    r"|^(?:yönetmeliğin\s+)?yay[ıi][mn]landığı\s+resm"
    r"|(?:ilişkin|ait)\s+cetve[lt]$"
)
# A lone "Yürürlüğe (Giriş) (Tarihi)" column header; must start with a capital letter so a
# wrapped body line "yürürlüğe giriş" is not mistaken for it.
_LONE_HEADER_RE = re.compile(r"^yürürlüğe(?:\s+giriş(?:\s+tarihi)?)?$")
ANNEX_RE = re.compile(r"^\s*EK\s*[-–]?\s*(?:\d{1,2}|[IVX]{1,4})\b")
_BARE_ANNEX_RE = re.compile(r"EK\s*[-–]?\s*(?:\d{1,2}|[IVX]{1,4})\s*[:.-]?")
# A note opens with a keyword (optionally followed by up to five words) and a colon, e.g.
# "(Değişik:", "(Ek iki cümle:", or is a law-amendment remark "(6676 sayılı Kanunun …".
_NOTE_START_RE = re.compile(
    r"\(\s*(?:(?:Değişik|Ek|Mülga|Yeniden\s+düzenleme)(?:\s+[^\W\d_]+){0,5}\s*:"
    r"|(?P<law>\d{4}\s+sayılı\s+kanun))",
    re.IGNORECASE,
)
_LAW_NOTE_RE = re.compile(r"maddesi(?:yle|ile)\s.*(?:değiş|ekle|yürürlük|mülga|kaldır)", re.I)
_FOOTNOTE_DIGIT_RE = re.compile(r"(?<=[^\W\d_])\d{1,2}(?:,\s*\d{1,2})*$")
_MAX_TITLE_LEN = 120
_MAX_TITLE_TAIL_LEN = 60


def _closing_paren(text: str, start: int) -> int | None:
    """Index of the parenthesis closing the one at ``start`` (nesting aware)."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return None


def extract_amendment_notes(text: str) -> tuple[str, list[str]]:
    """Move "(Değişik: …)"-style notes out of ``text``; return (clean text, notes)."""
    notes: list[str] = []
    parts: list[str] = []
    pos = 0
    while match := _NOTE_START_RE.search(text, pos):
        end = _closing_paren(text, match.start())
        if end is None:
            break
        note = text[match.start() : end + 1]
        if match["law"] and not _LAW_NOTE_RE.search(note):
            parts.append(text[pos : end + 1])  # a plain reference to a law, keep it
        else:
            parts.append(text[pos : match.start()])
            notes.append(note)
        pos = end + 1
    if not notes:
        return text, []
    clean = collapse_ws("".join(parts) + text[pos:])
    return tidy_punctuation(clean), notes


def footnote_markers(title: str) -> set[int]:
    """Footnote numbers glued to the end of a title: "… süreci2, 3" -> {2, 3}."""
    match = _FOOTNOTE_DIGIT_RE.search(title)
    return {int(n) for n in re.findall(r"\d+", match.group())} if match else set()


def strip_footnote_digit(title: str) -> str:
    """Drop footnote numbers glued to a title: "Projelerin sunulması2" -> "… sunulması"."""
    return _FOOTNOTE_DIGIT_RE.sub("", title)


def _label_of(line: str) -> tuple[str | None, str]:
    for regex, fmt in ((PARA_RE, "({})"), (BENT_RE, "{})"), (ALT_BENT_RE, "{})")):
        if match := regex.match(line):
            return fmt.format(match.group(1)), line[match.end() :].strip()
    return None, line


def build_paragraphs(
    lines: Sequence[tuple[int | None, str]],
) -> tuple[list[Paragraph], list[str]]:
    """Split article lines at fıkra/bent markers; return paragraphs and amendment notes."""
    groups: list[tuple[str | None, int | None, list[str]]] = []
    for page, line in lines:
        label, rest = _label_of(line)
        if label is not None or not groups:
            groups.append((label, page, [rest]))
        else:
            groups[-1][2].append(line)
    paragraphs: list[Paragraph] = []
    notes: list[str] = []
    for label, page, parts in groups:
        text, found = extract_amendment_notes(join_lines(parts))
        notes += found
        if text:
            paragraphs.append(Paragraph(label=label, text=text, page=page))
    return paragraphs, notes


def is_repealed(text: str, notes: Sequence[str]) -> bool:
    """An article is repealed when nothing but a "(Mülga: …)" note is left of it."""
    return not text and any(n[1:].lstrip().casefold().startswith("mülga") for n in notes)


def paragraphs_text(paragraphs: Sequence[Paragraph]) -> str:
    """Section text: one paragraph per line, each prefixed by its label."""
    return "\n".join(f"{p.label} {p.text}" if p.label else p.text for p in paragraphs)


def _is_title(line: str | None) -> bool:
    return bool(
        line
        and len(line) <= _MAX_TITLE_LEN
        and (line[0].isupper() or line[0].isdigit())
        and not line.endswith((",", ";", ":", "."))
    )


def _is_caps_line(line: str) -> bool:
    letters = [ch for ch in line if ch.isalpha()]
    return len(letters) >= 3 and all(ch.isupper() for ch in letters)


def _hierarchy_label(match: re.Match[str], name: str | None) -> tuple[str, str]:
    """Return (level, label) such as ("KISIM", "İkinci Kısım – Başvuru")."""
    if match["lvl"]:
        level = match["lvl"]
        ordinal = re.sub(rf"^(ON|YİRMİ)(?=(?:{_ORDINALS}))", r"\1 ", match["ord"])
        label = tr_title(f"{ordinal} {level}")
    else:
        level = match["lvl2"]
        label = f"{tr_title(level)} {match['roman']}"
    return level, f"{label} – {name}" if name else label


@dataclass
class _Article:
    madde_no: str
    title: str | None
    hierarchy: list[str]
    lines: list[tuple[int, str]] = field(default_factory=list)
    markers: set[int] = field(default_factory=set)  # footnote numbers glued to the title
    footnotes: list[str] = field(default_factory=list)

    def to_section(self, doc_id: str, index: int) -> Section:
        paragraphs, notes = build_paragraphs(self.lines)
        title = self.title
        if title:
            title, title_notes = extract_amendment_notes(title)
            notes = title_notes + notes
        notes += self.footnotes
        text = paragraphs_text(paragraphs)
        return make_section(
            doc_id,
            index,
            SectionKind.MADDE,
            text=text,
            paragraphs=paragraphs,
            title=title or None,
            hierarchy=self.hierarchy,
            madde_no=self.madde_no,
            page_start=self.lines[0][0],
            page_end=self.lines[-1][0],
            amendment_notes=notes,
            repealed=is_repealed(text, notes),
        )


def is_history_line(line: str) -> bool:
    """Does ``line`` belong to the header of an amendment-history table?"""
    text = line.strip()
    lowered = tr_lower(text)
    if text[:1].isupper() and _LONE_HEADER_RE.match(lowered):
        return True
    return bool(HISTORY_RE.search(lowered))


def history_start(lines: Sequence[tuple[int, str]], after: int) -> int:
    """Index where the trailing amendment-history table begins (``len(lines)`` if none).

    Only lines after index ``after`` (the last article) are candidates, so the same words in
    an article body never cut the document. The table's capitalised title is included.
    """
    start = next(
        (i for i in range(after + 1, len(lines)) if is_history_line(lines[i][1])), len(lines)
    )
    if start < len(lines):
        while start - 1 > after and _is_caps_line(lines[start - 1][1]):
            start -= 1
    return start


def _is_structural(line: str) -> bool:
    return bool(ARTICLE_RE.match(line) or HIER_RE.match(line) or is_history_line(line))


def _annex_title(block: Sequence[tuple[int, str]]) -> tuple[str, list[str], int]:
    """Title of the annex starting at ``block[0]``: (title, amendment notes, lines used)."""
    used, title = 1, block[0][1]
    while title.count("(") > title.count(")") and used < min(len(block), 4):
        title = join_lines([title, block[used][1]])
        used += 1
    title, notes = extract_amendment_notes(title)
    title = title.strip()
    if _BARE_ANNEX_RE.fullmatch(title) and used < len(block) and _is_caps_line(block[used][1]):
        title = f"{title} {block[used][1]}"
        used += 1
    return title, notes, used


def _annex_sections(
    lines: Sequence[tuple[int, str]], doc_id: str, first_index: int
) -> list[Section]:
    """One BOLUM section per ``EK-n`` annex that follows the last article."""
    starts = [i for i, (_, line) in enumerate(lines) if ANNEX_RE.match(line)]
    sections: list[Section] = []
    for n, start in enumerate(starts):
        block = lines[start : starts[n + 1] if n + 1 < len(starts) else len(lines)]
        title, notes, used = _annex_title(block)
        paragraphs, body_notes = build_paragraphs(block[used:])
        sections.append(
            make_section(
                doc_id,
                first_index + n,
                SectionKind.BOLUM,
                text=paragraphs_text(paragraphs),
                paragraphs=paragraphs,
                title=title,
                page_start=block[0][0],
                page_end=block[-1][0],
                amendment_notes=notes + body_notes,
            )
        )
    return sections


def parse_sections(pages: Sequence[PageText], doc_id: str) -> list[Section]:
    """Parse the articles (and trailing annexes) of a legislation document.

    Returns an empty list when the document has no articles.
    """
    lines = [(page.number, line) for page in pages for line in page.lines]
    article_rows = [i for i, (_, line) in enumerate(lines) if ARTICLE_RE.match(line)]
    if not article_rows:
        return []
    end = history_start(lines, article_rows[-1])
    annex_at = next(
        (i for i in range(article_rows[-1] + 1, end) if ANNEX_RE.match(lines[i][1])), end
    )
    articles = _walk_articles(lines[:annex_at])
    _attach_footnotes(articles, pages)
    sections = [article.to_section(doc_id, n) for n, article in enumerate(articles, start=1)]
    return sections + _annex_sections(lines[annex_at:end], doc_id, len(sections) + 1)


def _is_title_tail(line: str) -> bool:
    """Second line of a wrapped article title ("izleme süreci3", "tarihi")."""
    return (
        bool(line)
        and line[0].islower()
        and len(line) <= _MAX_TITLE_TAIL_LEN
        and not line.endswith((",", ";", ":", "."))
    )


def _take_title(recent: list[str], articles: list[_Article]) -> str | None:
    """Title line(s) just before an article marker, removed from the previous article.

    A title is the last line before ``MADDE`` or, when it wraps, the last two lines (the
    second starting in lower case).
    """
    if recent and _is_title(recent[-1]):
        title_lines = recent[-1:]
    elif len(recent) >= 2 and _is_title_tail(recent[-1]) and _is_title(recent[-2]):
        title_lines = recent[-2:]
    else:
        return None
    if articles:
        previous = articles[-1].lines
        tail = [line for _, line in previous[-len(title_lines) :]]
        if tail == title_lines and len(previous) > len(title_lines):
            del previous[-len(title_lines) :]
    return join_lines(title_lines)


def _attach_footnotes(articles: list[_Article], pages: Sequence[PageText]) -> None:
    """Give each page footnote to the article whose title carries its number.

    Without a matching marker the footnote goes to the article running at that page.
    """
    for page in pages:
        candidates = [a for a in articles if a.lines[0][0] <= page.number] or articles[:1]
        for note in page.footnotes:
            leading = re.match(r"\d+", note)
            number = int(leading.group()) if leading else -1
            marked = [a for a in candidates if number in a.markers]
            target = marked[-1] if marked else candidates[-1] if candidates else None
            if target is not None:
                target.footnotes.append(note)


def _walk_articles(lines: Sequence[tuple[int, str]]) -> list[_Article]:
    """Group lines into articles while tracking the KISIM/BÖLÜM hierarchy."""
    articles: list[_Article] = []
    kisim: str | None = None
    bolum: str | None = None
    recent: list[str] = []  # plain lines since the last structural line
    i = 0
    while i < len(lines):
        page, line = lines[i]
        if hier := HIER_RE.match(line):
            name = None
            if i + 1 < len(lines) and not _is_structural(lines[i + 1][1]):
                name = lines[i + 1][1]
                i += 1
            level, label = _hierarchy_label(hier, name)
            if level == "KISIM":
                kisim, bolum = label, None
            else:
                bolum = label
            recent = []
        elif article := ARTICLE_RE.match(line):
            raw_title = _take_title(recent, articles)
            prefix = article["prefix"]
            number = f"{tr_title(prefix)} {article['no']}" if prefix else article["no"]
            hierarchy = [label for label in (kisim, bolum) if label]
            articles.append(
                _Article(
                    number,
                    strip_footnote_digit(raw_title) if raw_title else None,
                    hierarchy,
                    [(page, article["rest"])],
                    markers=footnote_markers(raw_title) if raw_title else set(),
                )
            )
            recent = []
        else:
            if articles:
                articles[-1].lines.append((page, line))
            recent = [*recent[-1:], line]
        i += 1
    return articles
