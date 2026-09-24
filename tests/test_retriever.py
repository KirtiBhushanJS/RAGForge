"""Tests for FAISS retrieval: top-k, ordering, metadata, deduplication."""

from __future__ import annotations

import pytest

from src.chunker import Chunk
from src.retriever import Retriever
from src.vector_store import IndexNotReadyError, load_index


def test_retrieve_top_k_sorted(sample_chunks, built_retriever) -> None:
    retriever: Retriever = built_retriever(sample_chunks)
    results = retriever.retrieve("truck fleet", top_k=2)
    assert len(results) == 2
    # sorted by descending score
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    # the truck chunk should be the best match
    assert results[0].document == "trucks.pdf"
    assert "320 trucks" in results[0].text


def test_retrieve_preserves_metadata(sample_chunks, built_retriever) -> None:
    retriever: Retriever = built_retriever(sample_chunks)
    results = retriever.retrieve("wind capacity hydro", top_k=1)
    result = results[0]
    assert result.document == "energy.pdf"
    assert result.page == 2
    assert result.document_path == "/docs/energy.pdf"
    assert result.chunk_id.startswith("chunk_")
    assert result.score >= 0.0


def test_retrieve_all_chunks_when_top_k_high(sample_chunks, built_retriever) -> None:
    retriever: Retriever = built_retriever(sample_chunks)
    results = retriever.retrieve("torque", top_k=10)
    assert len(results) == 4


def test_deduplicate_same_page(sample_chunks, built_retriever) -> None:
    """Two overlapping chunks from the same page collapse into one result."""

    def chunk(doc: str, page: int, text: str, idx: int) -> Chunk:
        return Chunk(
            chunk_id=f"c_{idx}",
            document=doc,
            document_path=f"/docs/{doc}",
            page=page,
            text=text,
        )

    overlapping_page = (
        "zebra "+ "token " * 790 + "zebra zebra " + "filler " * 300
    )
    chunks = [
        chunk("animals.pdf", 1, overlapping_page, 0),
        chunk("animals.pdf", 2, "a completely unrelated page 2 topic", 1),
        chunk("cars.pdf", 1, "cars go fast", 2),
    ]
    retriever: Retriever = built_retriever(chunks)
    results = retriever.retrieve("zebra zebra", top_k=5)
    same_page = [(r.document, r.page) for r in results if r.document == "animals.pdf" and r.page == 1]
    assert len(same_page) == 1


def test_retrieve_is_deterministic(sample_chunks, built_retriever) -> None:
    """Repeated retrieval must not reorder chunks with tied scores."""
    retriever: Retriever = built_retriever(sample_chunks)
    first = retriever.retrieve("torque", top_k=10)
    second = retriever.retrieve("torque", top_k=10)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_no_results_when_index_empty(stub_embedder) -> None:
    empty_meta = {"dim": 64, "chunks": []}
    from src.retriever import Retriever

    import faiss
    import numpy as np

    index = faiss.IndexFlatIP(64)
    retriever = Retriever(index, empty_meta, stub_embedder)
    assert retriever.retrieve("anything") == []


def test_load_index_missing_raises_helpful_error(tmp_path) -> None:
    with pytest.raises(IndexNotReadyError, match="Run `python scripts/ingest.py`"):
        load_index(tmp_path / "nope.faiss", tmp_path / "nope.json")