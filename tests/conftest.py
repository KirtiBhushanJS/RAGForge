"""Shared fixtures: stub embedder, in-memory index, temp PDF factory.

Tests must not download models or call real APIs, so embeddings are produced by
a deterministic word-hash encoder and generation is stubbed.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import faiss
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunker import Chunk  # noqa: E402
from src.retriever import Retriever  # noqa: E402
from src.vector_store import _metadata_payload, build_index  # noqa: E402


class StubEmbedder:
    """Deterministic bag-of-words embedding (64 dims); text/query share words -> higher cosine."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim
        self.model_name = "stub"

    def dimension(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"\S+", text.lower()):
                bucket = int(hashlib.md5(word.encode("utf-8")).hexdigest()[:8], 16) % self._dim
                out[i, bucket] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query])


@pytest.fixture
def stub_embedder() -> StubEmbedder:
    return StubEmbedder()


@pytest.fixture
def make_pdf():
    """Return a factory that writes a real text PDF via PyMuPDF."""

    def _make(path: Path, pages: list[str]) -> Path:
        import pymupdf as fitz

        doc = fitz.open()
        for text in pages:
            page = doc.new_page(width=595, height=842)
            page.insert_textbox(fitz.Rect(50, 50, 545, 800), text, fontsize=10)
        doc.save(str(path))
        doc.close()
        return path

    return _make


def chunks_to_metadata(chunks: list[Chunk], model: str = "stub", dim: int = 64) -> dict:
    payload = _metadata_payload(chunks=chunks, model=model, dim=dim,
                                created_at="2026-01-01T00:00:00",
                                chunk_size=None, chunk_overlap=None)
    return payload


@pytest.fixture
def built_retriever(stub_embedder):
    """Return a Retriever built from ad hoc chunks."""

    def _build(chunks: list[Chunk]) -> Retriever:
        index = build_index(chunks, stub_embedder)
        payload = chunks_to_metadata(chunks)
        return Retriever(index, payload, stub_embedder)

    return _build


@pytest.fixture
def sample_chunks(stub_embedder) -> list[Chunk]:
    """Three chunks on different pages/documents with distinct topics."""

    def chunk(doc: str, page: int, text: str, idx: int) -> Chunk:
        return Chunk(
            chunk_id=f"chunk_{doc}_{page}_{idx}",
            document=doc,
            document_path=f"/docs/{doc}",
            page=page,
            text=text,
        )

    return [
        chunk("trucks.pdf", 1, "Globex operates a fleet of 320 trucks and 14 distribution centers", 0),
        chunk("trucks.pdf", 2, "The average on-time delivery rate is 96 percent", 0),
        chunk("energy.pdf", 1, "Aurora reported revenue of 2.4 billion EUR in 2025", 0),
        chunk("energy.pdf", 2, "Wind capacity is 850 MW and hydro capacity is 1100 MW", 0),
    ]


@pytest.fixture
def faiss_index_for(sample_chunks):
    return build_index(sample_chunks, StubEmbedder())


def test_faiss_importable() -> None:
    assert hasattr(faiss, "IndexFlatIP")