"""FAISS vector store with persistent index + metadata.

Layout
------
data/processed/index.faiss      -> FAISS binary index
data/processed/metadata.json    -> one dict per chunk, same order as index rows
data/processed/ingest_report.json -> summary of the last ingestion run

The metadata JSON records everything needed to produce citations later:
chunk_id, document, document_path, page and text.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.chunker import Chunk
from src.config import Settings, get_settings
from src.utils import ensure_dir, read_json, write_json

logger = logging.getLogger("rag.vector_store")


class IndexNotReadyError(Exception):
    """Raised when the persistent FAISS index is missing or unusable."""


@dataclass
class IndexMetadata:
    """Everything we need to know about a built index."""

    model: str
    dim: int
    num_chunks: int
    documents: list[str]
    created_at: str
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    chunks: list[dict[str, Any]] | None = None


def _metadata_payload(
    chunks: list[Chunk],
    model: str,
    dim: int,
    created_at: str,
    chunk_size: int | None,
    chunk_overlap: int | None,
) -> dict[str, Any]:
    """Build the JSON payload describing every chunk, in index order."""
    documents: list[str] = []
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for chunk in chunks:
        if chunk.document.lower() not in seen:
            seen.add(chunk.document.lower())
            documents.append(chunk.document)
        records.append(
            {
                "chunk_id": chunk.chunk_id,
                "document": chunk.document,
                "document_path": chunk.document_path,
                "page": chunk.page,
                "text": chunk.text,
            }
        )
    return {
        "model": model,
        "dim": dim,
        "num_chunks": len(records),
        "documents": documents,
        "created_at": created_at,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunks": records,
    }


def build_index(
    chunks: list[Chunk],
    embedder: Any,
    settings: Settings | None = None,
) -> "faiss.IndexFlatIP":  # type: ignore[name-defined]
    """Embed every chunk and build a normalized inner-product FAISS index."""
    settings = settings or get_settings()
    if not chunks:
        raise ValueError("Cannot build an index from an empty chunk list.")

    texts = [c.text for c in chunks]
    logger.info("Embedding %d chunks with '%s'...", len(texts), settings.embedding_model)
    start = time.perf_counter()
    vectors = embedder.encode(texts)
    elapsed = time.perf_counter() - start
    logger.info("Embedding %d texts took %.2fs (dim=%s).", len(vectors), elapsed, vectors.shape[1])

    import faiss

    index = faiss.IndexFlatIP(int(vectors.shape[1]))
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    return index


def save_index(
    index: Any,
    chunks: list[Chunk],
    index_path: Path,
    metadata_path: Path,
    ingest_report_path: Path,
    settings: Settings | None = None,
    ingest_stats: dict[str, Any] | None = None,
) -> None:
    """Persist the FAISS index, its metadata and an ingestion report."""
    settings = settings or get_settings()
    ensure_dir(index_path.parent)

    index_dim = int(index.d)
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    payload = _metadata_payload(
        chunks=chunks,
        model=settings.embedding_model,
        dim=index_dim,
        created_at=created_at,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    if metadata_path.exists():
        metadata_path.with_suffix(metadata_path.suffix + ".bak").write_bytes(metadata_path.read_bytes())

    write_json(metadata_path, payload)
    import faiss

    faiss.write_index(index, str(index_path))

    report: dict[str, Any] = {
        "created_at": created_at,
        "model": settings.embedding_model,
        "dim": index_dim,
        "num_chunks": len(chunks),
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
    }
    if ingest_stats:
        report["ingest_stats"] = ingest_stats
    report["embedding_model"] = settings.embedding_model
    write_json(ingest_report_path, report)
    logger.info("Saved FAISS index (%d vectors) to %s", len(chunks), index_path)


def _validate_payload(payload: Any) -> dict[str, Any]:
    """Validate the metadata JSON structure, raising IndexNotReadyError if bad."""
    if not isinstance(payload, dict):
        raise IndexNotReadyError("metadata.json is corrupted: expected a JSON object.")
    chunks = payload.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise IndexNotReadyError("metadata.json has no chunk records. Re-run scripts/ingest.py.")
    for i, record in enumerate(chunks):
        for required in ("chunk_id", "document", "document_path", "page", "text"):
            if required not in record:
                raise IndexNotReadyError(f"metadata.json record {i} is missing '{required}'.")
    return payload


def load_index(
    index_path: Path,
    metadata_path: Path,
    settings: Settings | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load a persistent index. Raises IndexNotReadyError when unavailable.

    Returns (faiss_index, metadata_payload_dict).
    """
    settings = settings or get_settings()
    if not index_path.exists() or not metadata_path.exists():
        raise IndexNotReadyError(
            f"Vector index not found at {index_path}. Run `python scripts/ingest.py` first."
        )
    try:
        import faiss

        index = faiss.read_index(str(index_path))
    except Exception as exc:
        raise IndexNotReadyError(f"Could not read FAISS index '{index_path.name}': {exc}") from exc

    payload = read_json(metadata_path, default=None)
    payload = _validate_payload(payload)

    expected_dim = int(payload["dim"])
    actual_dim = int(index.d)
    if actual_dim != expected_dim:
        raise IndexNotReadyError(
            f"FAISS index dim ({actual_dim}) does not match metadata dim ({expected_dim}). "
            "The index is stale - re-run `python scripts/ingest.py`."
        )
    if settings.embedding_model and payload.get("model") != settings.embedding_model:
        logger.warning(
            "Index was built with embedding model '%s' but config requests '%s'. "
            "Embedding dimension is what matters for FAISS, but consider re-ingesting.",
            payload.get("model"),
            settings.embedding_model,
        )
    return index, payload


def index_is_ready(index_path: Path, metadata_path: Path) -> bool:
    """Cheap existence check (no FAISS import) for the UI / scripts."""
    return index_path.exists() and metadata_path.exists()


def get_chunk_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the list of chunk records stored in a metadata payload."""
    return payload["chunks"]


def summarize_index(payload: dict[str, Any]) -> dict[str, Any]:
    """Human-readable summary of the metadata payload."""
    chunks = get_chunk_records(payload)
    unique_docs = sorted({str(c["document"]).lower() for c in chunks})
    return {
        "model": payload.get("model"),
        "dim": payload.get("dim"),
        "num_chunks": payload.get("num_chunks", len(chunks)),
        "num_documents": len(unique_docs),
        "documents": sorted({str(c["document"]) for c in chunks}),
        "created_at": payload.get("created_at"),
        "chunk_size": payload.get("chunk_size"),
        "chunk_overlap": payload.get("chunk_overlap"),
    }


def flatten_documents(metadata: dict[str, Any]) -> list[str]:
    """Return the list of distinct document names present in the metadata."""
    seen: dict[str, None] = {}
    for record in get_chunk_records(metadata):
        seen.setdefault(str(record["document"]), None)
    return list(seen.keys())