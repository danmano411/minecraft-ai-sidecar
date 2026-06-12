"""
Section-based chunker for wiki HTML content.

Splits each page into chunks at h2/h3 boundaries. Tables are converted to
pipe-separated text so crafting recipe data survives as readable text.
"""

from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag

MAX_CHUNK_CHARS = 2000
MIN_CHUNK_CHARS = 80


@dataclass
class Chunk:
    page_title: str
    section_title: str
    text: str
    url: str
    chunk_id: str


def _table_to_text(table: Tag) -> str:
    rows: list[str] = []
    for row in table.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _element_to_text(el: Tag) -> str:
    if el.name == "table":
        return _table_to_text(el)
    return el.get_text(" ", strip=True)


def _extract_sections(html: str) -> list[tuple[str, str]]:
    """Return [(section_title, section_text), ...] from rendered wiki HTML."""
    soup = BeautifulSoup(html, "html.parser")
    content_div = soup.find("div", class_="mw-parser-output") or soup

    sections: list[tuple[str, str]] = []
    current_title = "Introduction"
    current_parts: list[str] = []

    def flush():
        text = "\n".join(p for p in current_parts if p.strip())
        if len(text) >= MIN_CHUNK_CHARS:
            sections.append((current_title, text))

    for el in content_div.children:
        if not isinstance(el, Tag):
            continue
        if el.name in ("h2", "h3"):
            flush()
            current_parts = []
            span = el.find("span", class_="mw-headline")
            current_title = span.get_text(strip=True) if span else el.get_text(strip=True)
        elif el.name in ("p", "ul", "ol", "dl", "table", "div"):
            text = _element_to_text(el).strip()
            if text:
                current_parts.append(text)

    flush()
    return sections


def _split_long_section(title: str, text: str) -> list[tuple[str, str]]:
    """Hard-split a section that exceeds MAX_CHUNK_CHARS into sub-chunks."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [(title, text)]
    chunks: list[tuple[str, str]] = []
    start = 0
    part = 0
    while start < len(text):
        end = start + MAX_CHUNK_CHARS
        chunk_text = text[start:end]
        label = f"{title} (part {part + 1})" if part > 0 else title
        chunks.append((label, chunk_text))
        start = end
        part += 1
    return chunks


def chunk_page(page_title: str, url: str, html: str) -> list[Chunk]:
    """Chunk a single wiki page into section-level Chunk objects."""
    sections = _extract_sections(html)
    chunks: list[Chunk] = []
    for section_title, text in sections:
        for i, (label, chunk_text) in enumerate(_split_long_section(section_title, text)):
            chunk_id = f"{page_title}::{label}".replace(" ", "_").lower()[:200]
            chunks.append(Chunk(
                page_title=page_title,
                section_title=label,
                text=chunk_text,
                url=url,
                chunk_id=chunk_id,
            ))
    return chunks
