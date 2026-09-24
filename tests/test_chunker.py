"""Tests for chunking: sizes, overlap, page-boundary preservation, metadata."""

from __future__ import annotations

import logging

import pytest

from src.chunker import chunk_documents, chunk_page, chunk_statistics, chunk_text
from src.pdf_loader import DocumentPage

CHUNK_SIZE = 800
OVERLAP = 150
MIN_WORDS = 60


def text_of(n_words: int, prefix: str = "word") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n_words))


def make_page(text: str, page: int = 1, document: str = "t.pdf", empty: bool = False) -> DocumentPage:
    return DocumentPage(
        document=document,
        document_path=f"/docs/{document}",
        page=page,
        text=text,
        num_pages=3,
        is_empty=empty,
    )


def test_chunk_text_single_window_when_small() -> None:
    pieces = chunk_text(text_of(100), CHUNK_SIZE, OVERLAP, MIN_WORDS)
    assert len(pieces) == 1
    assert pieces[0].split() == [f"word{i}" for i in range(100)]


def test_chunk_text_windows_and_overlap() -> None:
    n = 2000
    pieces = chunk_text(text_of(n), CHUNK_SIZE, OVERLAP, MIN_WORDS)
    assert len(pieces) >= 2
    assert all(len(p.split()) <= CHUNK_SIZE for p in pieces)

    # Second window should start around word 650 (step = chunk_size - overlap),
    # so it re-covers the overlap region of the first window (words 650..799).
    second = pieces[1].split()
    assert "word650" in second
    assert "word799" in second  # boundary word overlapped by window 1


def test_chunk_text_merges_tiny_trailing_window() -> None:
    # words so the last raw window would be < min_chunk_words
    n = CHUNK_SIZE + (CHUNK_SIZE - OVERLAP) + 20
    pieces = chunk_text(text_of(n), CHUNK_SIZE, OVERLAP, MIN_WORDS)
    assert pieces[0].split() != pieces[1].split() or len(pieces) == 1
    assert all(len(p.split()) >= MIN_WORDS or len(pieces) == 1 for p in pieces)


def test_chunk_text_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        chunk_text("hello world", 0, 0, 1)
    with pytest.raises(ValueError):
        chunk_text("hello world", 10, 10, 1)
    with pytest.raises(ValueError):
        chunk_text("hello world", 10, -1, 1)


def test_chunk_page_skips_empty_pages() -> None:
    chunks = chunk_page(make_page("", empty=True), CHUNK_SIZE, OVERLAP, MIN_WORDS)
    assert chunks == []


def test_chunk_page_preserves_metadata() -> None:
    chunks = chunk_page(make_page(text_of(2000), page=2), CHUNK_SIZE, OVERLAP, MIN_WORDS)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.document == "t.pdf"
        assert c.document_path == "/docs/t.pdf"
        assert c.page == 2
        assert c.chunk_id
        assert c.text


def test_chunks_never_span_pages() -> None:
    page_a = make_page(text_of(2000, "alpha"), page=1, document="a.pdf")
    page_b = make_page(text_of(300, "beta"), page=5, document="a.pdf")
    chunks = chunk_documents([page_a, page_b], CHUNK_SIZE, OVERLAP, MIN_WORDS)
    pages = {c.page for c in chunks}
    assert pages == {1, 5}
    # Words from one page must never leak into chunks of the other page.
    for c in chunks:
        if c.page == 1:
            assert "alpha" in c.text and "beta" not in c.text
        else:
            assert "beta" in c.text and "alpha" not in c.text


def test_chunk_statistics() -> None:
    chunks = chunk_page(make_page(text_of(100)), CHUNK_SIZE, OVERLAP, MIN_WORDS)
    stats = chunk_statistics(chunks)
    assert stats["num_chunks"] == len(chunks)
    assert stats["avg_chunk_words"] == 100.0
    assert stats["min_chunk_words"] == 100
    assert stats["max_chunk_words"] == 100