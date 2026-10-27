"""Programme page parser for tubitak.gov.tr (Drupal) HTML (FR-P7, FR-P8).

A page is a sequence of Drupal "paragraph" components inside ``<article>``. Two types carry
content: ``duz-metin`` (plain text, optional title ``field-baslik``, body ``field-icerik``)
and ``sikca-sorulan-sorular`` (FAQ, a list of ``soru-cevap`` pairs). File lists, galleries
and navigation blocks are skipped. Every FAQ pair becomes its own section.
"""

from collections.abc import Iterator
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

from tubitak_rag.domain.document import Document, Paragraph, Section
from tubitak_rag.domain.enums import SectionKind
from tubitak_rag.domain.source import SourceEntry
from tubitak_rag.ingestion.document_builder import build_document_skeleton, make_section
from tubitak_rag.ingestion.hashing import sha256_of
from tubitak_rag.ingestion.normalize import collapse_ws, normalize_text

TEXT_BLOCK = "paragraph--type--duz-metin"
FAQ_BLOCK = "paragraph--type--sikca-sorulan-sorular"
FAQ_ITEM = "paragraph--type--soru-cevap"
FAQ_HIERARCHY = "Sıkça Sorulan Sorular"
_HEADINGS = {"h2", "h3", "h4", "h5"}
_BLOCKS = {"p", "li", "ul", "ol", "table", "tbody", "thead", "tr", "div", "blockquote"}
_BLOCKS |= _HEADINGS

# A block of page text: (is_heading, text)
Block = tuple[bool, str]


def _text(node: Tag | NavigableString, separator: str = "") -> str:
    raw = node.get_text(separator) if isinstance(node, Tag) else str(node)
    return collapse_ws(normalize_text(raw))


def _field(element: Tag, name: str) -> Tag | None:
    found = element.find(class_=f"field--name-field-{name}")
    return found if isinstance(found, Tag) else None


def _has_blocks(tag: Tag) -> bool:
    return any(isinstance(child, Tag) and child.name in _BLOCKS for child in tag.descendants)


def _row(row: Tag) -> Iterator[Block]:
    cells = [c for c in row.find_all(["td", "th"], recursive=False) if isinstance(c, Tag)]
    if len(cells) == 1:
        yield from _blocks(cells[0])
    else:
        joined = " | ".join(text for cell in cells if (text := _text(cell, " ")))
        if joined:
            yield False, joined


def _blocks(container: Tag) -> Iterator[Block]:
    """Yield the text blocks of ``container`` in document order (inline runs are merged)."""
    inline: list[str] = []

    def flush() -> Iterator[Block]:
        text = collapse_ws(normalize_text("".join(inline)))
        inline.clear()
        if text:
            yield False, text

    for child in container.children:
        if isinstance(child, NavigableString):
            inline.append(str(child))
            continue
        if not isinstance(child, Tag):
            continue
        if child.name not in _BLOCKS:
            inline.append(child.get_text())
            continue
        yield from flush()
        if child.name in _HEADINGS:
            if text := _text(child):
                yield True, text
        elif child.name == "tr":
            yield from _row(child)
        elif child.name == "li":
            if text := _text(child, " "):
                yield False, f"• {text}"
        elif child.name in ("p", "blockquote") or not _has_blocks(child):
            if text := _text(child):
                yield False, text
        else:
            yield from _blocks(child)
    yield from flush()


def _section(
    doc_id: str,
    index: int,
    kind: SectionKind,
    title: str,
    hierarchy: list[str],
    paragraphs: list[Paragraph],
    text: str,
) -> Section:
    return make_section(
        doc_id, index, kind, text=text, paragraphs=paragraphs, title=title, hierarchy=hierarchy
    )


def _text_block_parts(block: Tag, page_title: str) -> Iterator[tuple[str, list[str], list[str]]]:
    """(title, hierarchy, paragraph texts) for each part of a plain-text block."""
    heading = _field(block, "baslik")
    block_title = _text(heading) if heading else ""
    body = _field(block, "icerik")
    if body is None:
        return
    title = block_title or page_title
    hierarchy: list[str] = []
    texts: list[str] = []
    for is_heading, text in _blocks(body):
        if is_heading:
            if texts:
                yield title, hierarchy, texts
            title, hierarchy, texts = text, [block_title] if block_title else [], []
        else:
            texts.append(text)
    if texts:
        yield title, hierarchy, texts


def _faq_parts(block: Tag) -> Iterator[tuple[str, str]]:
    for item in block.find_all(class_=FAQ_ITEM):
        if not isinstance(item, Tag):
            continue
        question, answer = _field(item, "soru"), _field(item, "cevap")
        if question is None or answer is None:
            continue
        answer_text = "\n\n".join(text for _, text in _blocks(answer))
        if _text(question) and answer_text:
            yield _text(question), answer_text


def extract_sections(html: str, doc_id: str, page_title: str) -> list[Section]:
    """Plain-text blocks become BOLUM sections, FAQ pairs SSS sections (in page order)."""
    soup = BeautifulSoup(html, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    root = soup.find("article") or soup
    sections: list[Section] = []
    for block in root.find_all("div", class_="paragraph"):
        if not isinstance(block, Tag):
            continue
        classes: list[str] = list(block.get("class") or [])
        if TEXT_BLOCK in classes:
            for title, hierarchy, texts in _text_block_parts(block, page_title):
                paragraphs = [Paragraph(label=None, text=t, page=None) for t in texts]
                text = "\n\n".join(texts)
                index = len(sections) + 1
                sections.append(
                    _section(doc_id, index, SectionKind.BOLUM, title, hierarchy, paragraphs, text)
                )
        elif FAQ_BLOCK in classes:
            heading = _field(block, "baslik")
            hierarchy = [_text(heading) if heading else FAQ_HIERARCHY]
            for question, answer in _faq_parts(block):
                paragraphs = [Paragraph(label=None, text=answer, page=None)]
                text = f"Soru: {question}\nCevap: {answer}"
                index = len(sections) + 1
                sections.append(
                    _section(doc_id, index, SectionKind.SSS, question, hierarchy, paragraphs, text)
                )
    return sections


class HtmlParser:
    """``DocumentParser`` for programme pages."""

    def parse(self, entry: SourceEntry, path: Path) -> Document:
        """Parse the saved ``<article>`` HTML of one programme page."""
        sections = extract_sections(path.read_text(encoding="utf-8"), entry.doc_id, entry.title)
        doc = build_document_skeleton(entry, sha256_of(path))
        return doc.model_copy(update={"sections": sections})
