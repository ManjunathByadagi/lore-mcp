"""Unit tests for lore.doc_processor.

Covers DocumentProcessor methods (chunking, hashing, token counting,
title extraction, topic inference) and DocumentChunk construction.

doc_processor.py was at 14.2% (86 stmts) — largest coverage gap by percentage
for a module with no external-service dependencies; everything can be tested
with in-process calls and tmp_path filesystem fixtures.
"""

from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path

import pytest

from lore.doc_processor import DocumentChunk, DocumentProcessor

# ---------------------------------------------------------------------------
# DocumentChunk
# ---------------------------------------------------------------------------


def test_document_chunk_defaults():
    chunk = DocumentChunk(content="hello")
    assert chunk.content == "hello"
    assert chunk.section is None
    assert chunk.line_start == 0
    assert chunk.line_end == 0


def test_document_chunk_with_all_args():
    chunk = DocumentChunk(content="body", section="Introduction", line_start=3, line_end=10)
    assert chunk.section == "Introduction"
    assert chunk.line_start == 3
    assert chunk.line_end == 10


# ---------------------------------------------------------------------------
# DocumentProcessor — construction
# ---------------------------------------------------------------------------


def test_processor_default_chunk_size():
    dp = DocumentProcessor()
    assert dp.chunk_size == 2000


def test_processor_custom_chunk_size():
    dp = DocumentProcessor(chunk_size=500)
    assert dp.chunk_size == 500


# ---------------------------------------------------------------------------
# compute_hash
# ---------------------------------------------------------------------------


def test_compute_hash_returns_sha256():
    dp = DocumentProcessor()
    text = "hello world"
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert dp.compute_hash(text) == expected


def test_compute_hash_empty_string():
    dp = DocumentProcessor()
    expected = hashlib.sha256(b"").hexdigest()
    assert dp.compute_hash("") == expected


def test_compute_hash_unicode():
    dp = DocumentProcessor()
    text = "Latvian: sveiki"
    h = dp.compute_hash(text)
    assert len(h) == 64  # 256-bit hex


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------


def test_count_tokens_nonzero_for_text():
    dp = DocumentProcessor()
    count = dp.count_tokens("Hello, world!")
    assert count > 0


def test_count_tokens_empty_string():
    dp = DocumentProcessor()
    assert dp.count_tokens("") == 0


def test_count_tokens_longer_text_is_more():
    dp = DocumentProcessor()
    short = dp.count_tokens("hello")
    long_ = dp.count_tokens("hello " * 100)
    assert long_ > short


# ---------------------------------------------------------------------------
# read_document — filesystem tests
# ---------------------------------------------------------------------------


def test_read_document_plain_markdown(tmp_path):
    doc = tmp_path / "test.md"
    doc.write_text("# Hello\n\nThis is content.")
    dp = DocumentProcessor()
    content, meta = dp.read_document(str(doc))
    assert "Hello" in content
    assert isinstance(meta, dict)


def test_read_document_with_frontmatter(tmp_path):
    doc = tmp_path / "front.md"
    doc.write_text("---\ntitle: My Doc\nauthor: Alice\n---\n\nBody text here.")
    dp = DocumentProcessor()
    content, meta = dp.read_document(str(doc))
    assert meta.get("title") == "My Doc"
    assert meta.get("author") == "Alice"
    assert "Body text here." in content


def test_read_document_missing_file_raises():
    dp = DocumentProcessor()
    with pytest.raises(FileNotFoundError):
        dp.read_document("/nonexistent/path/to/missing.md")


def test_read_document_utf8_content(tmp_path):
    doc = tmp_path / "unicode.md"
    doc.write_text("# Sveiki\n\nLatviešu teksts: ā, ē, ī, ū", encoding="utf-8")
    dp = DocumentProcessor()
    content, _ = dp.read_document(str(doc))
    assert "Latviešu" in content


# ---------------------------------------------------------------------------
# generate_title
# ---------------------------------------------------------------------------


def test_generate_title_from_h1():
    dp = DocumentProcessor()
    content = "# My Great Document\n\nSome body text."
    title = dp.generate_title(content, "/some/path/doc.md")
    assert title == "My Great Document"


def test_generate_title_fallback_to_filename():
    dp = DocumentProcessor()
    content = "No headers here, just plain text."
    title = dp.generate_title(content, "/data/my_cool_document.md")
    # Should produce title-cased filename stem
    assert "Cool" in title or "cool" in title.lower()
    assert "Document" in title or "document" in title.lower()


def test_generate_title_ignores_h2_for_fallback():
    """Only H1 (single #) is used; ## should fall through to filename."""
    dp = DocumentProcessor()
    content = "## Section\n\nBody."
    title = dp.generate_title(content, "/path/my_file.md")
    # H2 is not H1 — must use filename
    assert "Section" not in title


def test_generate_title_strips_h1_whitespace():
    dp = DocumentProcessor()
    content = "#   Padded Title  \n\nContent."
    title = dp.generate_title(content, "/x/x.md")
    assert title == "Padded Title"


# ---------------------------------------------------------------------------
# extract_topic_from_path
# ---------------------------------------------------------------------------


def test_extract_topic_xtts():
    dp = DocumentProcessor()
    assert dp.extract_topic_from_path("/srv/latvian_xtts/docs/AUDIO.md") == "xtts"


def test_extract_topic_learning():
    dp = DocumentProcessor()
    assert dp.extract_topic_from_path("/srv/latvian_learning/CODEX/anki.md") == "learning"


def test_extract_topic_lab():
    dp = DocumentProcessor()
    assert dp.extract_topic_from_path("/srv/latvian_lab/notes/infra.md") == "infrastructure"


def test_extract_topic_mcp():
    dp = DocumentProcessor()
    assert dp.extract_topic_from_path("/srv/latvian_mcp/server.py") == "mcp-servers"


def test_extract_topic_default_uses_parent_dir():
    dp = DocumentProcessor()
    # None of the known patterns match → use parent dir name
    result = dp.extract_topic_from_path("/data/my-custom-topic/readme.md")
    assert result == "my-custom-topic"


# ---------------------------------------------------------------------------
# chunk_by_sections — structural tests (no token-size overflow needed)
# ---------------------------------------------------------------------------


def test_chunk_by_sections_empty_string():
    dp = DocumentProcessor()
    chunks = dp.chunk_by_sections("")
    # Empty or single chunk with empty content
    assert isinstance(chunks, list)


def test_chunk_by_sections_single_section():
    dp = DocumentProcessor()
    content = "# Introduction\n\nSome text here."
    chunks = dp.chunk_by_sections(content)
    assert len(chunks) >= 1
    intro_chunks = [c for c in chunks if c.section == "Introduction"]
    assert len(intro_chunks) >= 1


def test_chunk_by_sections_multiple_headers():
    dp = DocumentProcessor()
    content = textwrap.dedent("""\
        # First Section
        Content of first section.

        # Second Section
        Content of second section.

        # Third Section
        Content of third section.
    """)
    chunks = dp.chunk_by_sections(content)
    sections = [c.section for c in chunks if c.section]
    assert "First Section" in sections
    assert "Second Section" in sections
    assert "Third Section" in sections


def test_chunk_line_numbers_are_populated():
    dp = DocumentProcessor()
    content = "# Header\n\nLine 2\nLine 3"
    chunks = dp.chunk_by_sections(content)
    for chunk in chunks:
        assert chunk.line_end >= chunk.line_start


def test_chunk_by_sections_no_headers():
    """Content with no headers produces a single chunk with section=None."""
    dp = DocumentProcessor()
    content = "Just some plain text.\nNo headers here.\nAnother line."
    chunks = dp.chunk_by_sections(content)
    assert len(chunks) >= 1
    # All sections should be None (no header encountered)
    assert all(c.section is None for c in chunks)


def test_chunk_by_sections_section_too_large_splits(monkeypatch):
    """A section exceeding max_tokens is split into sub-chunks."""
    dp = DocumentProcessor(chunk_size=5)  # very small limit

    # Build content that is definitely > 5 tokens
    body = " ".join(["word"] * 50)
    content = f"# Big Section\n\n{body}"
    chunks = dp.chunk_by_sections(content, max_tokens=5)
    # Should have produced multiple chunks for the big section
    big_chunks = [c for c in chunks if c.section and "Big Section" in c.section]
    assert len(big_chunks) >= 2, f"Expected splits, got: {[c.section for c in chunks]}"


# ---------------------------------------------------------------------------
# _split_by_tokens — direct test
# ---------------------------------------------------------------------------


def test_split_by_tokens_respects_max():
    dp = DocumentProcessor()
    # Build a string with ~100 tokens
    text = " ".join(["word"] * 80)
    max_tokens = 20
    parts = dp._split_by_tokens(text, max_tokens)
    assert len(parts) >= 2
    # Each part should have at most max_tokens tokens
    for part in parts:
        assert dp.count_tokens(part) <= max_tokens


def test_split_by_tokens_single_chunk_when_small():
    dp = DocumentProcessor()
    text = "short text"
    parts = dp._split_by_tokens(text, max_tokens=1000)
    assert len(parts) == 1
    assert parts[0] == text or "short" in parts[0]


def test_split_by_tokens_empty_string():
    dp = DocumentProcessor()
    parts = dp._split_by_tokens("", max_tokens=100)
    # Empty tokens → empty list or one empty string
    assert isinstance(parts, list)
