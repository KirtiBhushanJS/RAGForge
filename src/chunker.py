"""Chunking with configurable size/overlap that preserves page boundaries.

Design notes
------------
- Text is chunked per page. Chunks NEVER span two different pages, so the
  page number stored in the chunk metadata is always exact.
- Chunk size is measured in approximate words; a sliding window with overlap
  is applied inside every page.
- Trailing windows that are very small (< `min_chunk_words`) are merged into
  the previous chunk instead of creating useless tiny chunks.
- A page whose whole text fits inside one window becomes a single chunk.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.pdf_loader import DocumentPage
from src.utils import word_count

logger = logging.getLogger("rag.chunker")

_WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    """A retrievable text unit together with its provenance metadata."""

    chunk_id: str
    document: str       # file name e.g. "manual.pdf"
    document_path: str  # full path to the source PDF
    page: int           # 1-based page number
    text: str


def _words_to_text(words: list[str]) -> str:
    return " ".join(words).strip()


def _chunk_id(relative_path: str, page: int, index: int) -> str:
    raw = f"{relative_path}|p{page}|i{index}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{digest}_p{page}_{index}"


def _relative_path(document_path: str, documents_dir: Path | None) -> str:
    """Path of the PDF relative to documents_dir (keeps duplicate names unique)."""
    path = Path(document_path)
    if documents_dir is not None:
        try:
            return path.relative_to(documents_dir).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def chunk_text(text: str, chunk_size: int, chunk_overlap: int, min_chunk_words: int) -> list[str]:
    """Split one block of text into overlapping windows of `chunk_size` words."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size")

    words = _WORD_RE.findall(text)
    if not words:
        return []

    if len(words) <= chunk_size:
        return [_words_to_text(words)]

    step = max(1, chunk_size - chunk_overlap)
    windows: list[list[str]] = []
    start = 0
    while start < len(words):
        windows.append(words[start : start + chunk_size])
        start += step
        if start >= len(words):
            break

    # Merge very small trailing chunks into the previous window.
    while len(windows) > 1 and len(windows[-1]) < min_chunk_words:
        last = windows.pop()
        windows[-1] = windows[-1] + last

    # Avoid an empty trailing window after merging.
    while windows and not windows[-1]:
        windows.pop()

    return [_words_to_text(w) for w in windows]


def chunk_page(
    page: DocumentPage,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_words: int,
    documents_dir: Path | None = None,
) -> list[Chunk]:
    """Chunk a single page. Returns [] for empty pages."""
    if page.is_empty or not page.text.strip():
        return []

    relative_path = _relative_path(page.document_path, documents_dir)
    pieces = chunk_text(page.text, chunk_size, chunk_overlap, min_chunk_words)
    chunks: list[Chunk] = []
    for index, piece in enumerate(pieces):
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(relative_path, page.page, index),
                document=page.document,
                document_path=page.document_path,
                page=page.page,
                text=piece,
            )
        )
    return chunks


def chunk_documents(
    pages: Iterable[DocumentPage],
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_words: int,
    documents_dir: Path | None = None,
) -> list[Chunk]:
    """Chunk every page of every document and flatten the results."""
    chunks: list[Chunk] = []
    for page in pages:
        chunks.extend(chunk_page(page, chunk_size, chunk_overlap, min_chunk_words, documents_dir))
    return chunks


def chunk_statistics(chunks: list[Chunk]) -> dict[str, float | int]:
    """Small summary used in the ingest report."""
    sizes = [word_count(c.text) for c in chunks]
    return {
        "num_chunks": len(chunks),
        "avg_chunk_words": round(float(sum(sizes)) / len(sizes), 1) if sizes else 0.0,
        "min_chunk_words": min(sizes) if sizes else 0,
        "max_chunk_words": max(sizes) if sizes else 0,
    }