"""Split document text into overlapping chunks that respect paragraph and sentence boundaries."""

import re
from dataclasses import dataclass, field

from app.services.parsing import Section

# Sentence ends: . ? ! and the Devanagari danda (।). Keeps the punctuation with the sentence.
_SENTENCE_END_RE = re.compile(r"(?<=[.?!।])\s+")


@dataclass
class TextChunk:
    index: int
    content: str
    meta: dict = field(default_factory=dict)


def _split_units(text: str, max_chars: int) -> list[str]:
    """Break text into units no longer than max_chars: paragraphs, then sentences, then words."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            units.append(para)
            continue
        for sentence in _SENTENCE_END_RE.split(para):
            sentence = sentence.strip()
            if len(sentence) <= max_chars:
                units.append(sentence)
                continue
            # A single huge sentence (tables, run-on legal text): hard-wrap on words.
            words, current = sentence.split(), ""
            for word in words:
                if current and len(current) + 1 + len(word) > max_chars:
                    units.append(current)
                    current = word
                else:
                    current = f"{current} {word}" if current else word
            if current:
                units.append(current)
    return units


def _tail(text: str, overlap: int) -> str:
    """Last ~overlap characters of text, starting on a word boundary."""
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    cut = text[-overlap:]
    space = cut.find(" ")
    return cut[space + 1 :] if space != -1 else cut


def chunk_sections(sections: list[Section], chunk_size: int, chunk_overlap: int) -> list[TextChunk]:
    """Greedily pack units into chunks of up to chunk_size characters. Each new chunk starts with
    the tail of the previous one so a fact split across a boundary is still retrievable."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[TextChunk] = []
    current, pages, has_new_text = "", set(), False

    def flush() -> None:
        nonlocal current, pages, has_new_text
        meta = {"pages": sorted(pages)} if pages else {}
        chunks.append(TextChunk(index=len(chunks), content=current.strip(), meta=meta))
        current = _tail(current, chunk_overlap)
        # The carried-over tail came from the last page of the chunk just written.
        pages = {max(pages)} if pages and current else set()
        has_new_text = False

    for section in sections:
        for unit in _split_units(section.text, chunk_size - chunk_overlap):
            if has_new_text and len(current) + 2 + len(unit) > chunk_size:
                flush()
            current = f"{current}\n\n{unit}" if current else unit
            has_new_text = True
            if section.page is not None:
                pages.add(section.page)

    if has_new_text:
        flush()
    return chunks
