import pytest

from app.services.chunking import chunk_sections
from app.services.parsing import Section


def test_short_text_is_one_chunk():
    chunks = chunk_sections([Section("Hello world.")], chunk_size=100, chunk_overlap=10)
    assert [c.content for c in chunks] == ["Hello world."]


def test_chunks_respect_size_and_overlap():
    paragraphs = [f"Paragraph {i} " + "word " * 30 for i in range(20)]
    chunks = chunk_sections([Section("\n\n".join(paragraphs))], chunk_size=400, chunk_overlap=60)
    assert len(chunks) > 1
    assert all(len(c.content) <= 400 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    # Each chunk after the first starts with text carried over from the previous one.
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.content.split("\n\n")[0] in prev.content


def test_long_sentence_is_hard_wrapped():
    chunks = chunk_sections([Section("x" * 10 + " " + "word " * 500)], chunk_size=200, chunk_overlap=20)
    assert all(len(c.content) <= 200 for c in chunks)


def test_devanagari_sentences_split_on_danda():
    text = "यह पहला वाक्य है। " * 40
    chunks = chunk_sections([Section(text)], chunk_size=150, chunk_overlap=0)
    assert all(c.content.endswith("।") for c in chunks)


def test_page_numbers_recorded():
    sections = [Section("Page one text.", page=1), Section("Page two text.", page=2)]
    chunks = chunk_sections(sections, chunk_size=1000, chunk_overlap=0)
    assert chunks[0].meta == {"pages": [1, 2]}


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_sections([Section("text")], chunk_size=100, chunk_overlap=100)
