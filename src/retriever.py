"""Top-k retriever over the FAISS index.

Also implements the "confidence gate": the similarity threshold used by the
pipeline to decide whether we have enough context to answer at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.config import Settings, get_settings
from src.utils import clamp
from src.vector_store import IndexNotReadyError  # noqa: F401  (re-exported for callers)

logger = logging.getLogger("rag.retriever")

_DEDUP_FACTOR = 4  # fetch extra candidates so deduplication has slack


@dataclass
class RetrievedChunk:
    """A retrieved chunk with its similarity score (cosine similarity)."""

    chunk_id: str
    document: str
    document_path: str
    page: int
    text: str
    score: float


class Retriever:
    """Performs similarity search and returns sorted, deduplicated results."""

    def __init__(
        self,
        index: Any,
        metadata: dict[str, Any],
        embedder: Any,
        settings: Settings | None = None,
    ) -> None:
        self.index = index
        self.metadata = metadata
        self.embedder = embedder
        self.settings = settings or get_settings()
        self._records: list[dict[str, Any]] = self.metadata["chunks"]
        self._count = len(self._records)

    # ------------------------------------------------------------------
    # Introspection helpers (used by the Streamlit sidebar)
    # ------------------------------------------------------------------
    @property
    def num_chunks(self) -> int:
        return self._count

    @property
    def documents(self) -> list[str]:
        seen: list[str] = []
        seen_set: set[str] = set()
        for record in self._records:
            key = str(record["document"]).lower()
            if key not in seen_set:
                seen_set.add(key)
                seen.append(str(record["document"]))
        return sorted(seen)

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int | None = None, deduplicate: bool | None = None) -> list[RetrievedChunk]:
        """Return the top-k chunks for `query`, sorted by descending score.

        Duplicated chunks from the same (document, page) are collapsed so that
        each source appears only once in the final list.
        """
        top_k = top_k or self.settings.top_k
        deduplicate = self.settings.deduplicate_sources if deduplicate is None else deduplicate
        if not query or not query.strip():
            raise ValueError("Empty query cannot be retrieved.")

        if self._count == 0:
            return []

        query_vector = self.embedder.encode_query(query.strip())
        k = min(self._count, max(top_k, top_k * _DEDUP_FACTOR))
        scores, indices = self.index.search(np.ascontiguousarray(query_vector, dtype=np.float32), k)

        results: list[RetrievedChunk] = []
        for score, matrix_index in zip(scores[0], indices[0]):
            if matrix_index < 0 or matrix_index >= self._count:
                continue
            record = self._records[int(matrix_index)]
            results.append(
                RetrievedChunk(
                    chunk_id=str(record["chunk_id"]),
                    document=str(record["document"]),
                    document_path=str(record["document_path"]),
                    page=int(record["page"]),
                    text=str(record["text"]),
                    score=float(clamp(float(score))),
                )
            )

        # Sort by score descending with a deterministic tie-breaker BEFORE and
        # AFTER deduplication, because FAISS may return near-identical scores in
        # any order. This keeps repeated runs (and experiments) reproducible.
        results.sort(key=lambda r: (-r.score, r.chunk_id))
        if deduplicate:
            results = self._deduplicate(results)
        results.sort(key=lambda r: (-r.score, r.chunk_id))
        return results[:top_k]

    @staticmethod
    def _deduplicate(results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Keep only the highest-scoring chunk per (document, page)."""
        best_by_source: dict[tuple[str, int], RetrievedChunk] = {}
        for chunk in results:
            key = (chunk.document.lower(), chunk.page)
            current = best_by_source.get(key)
            if current is None or chunk.score > current.score:
                best_by_source[key] = chunk
        return list(best_by_source.values())

    def max_score(self, query: str, top_k: int | None = None) -> float:
        """Highest similarity across the retrieved chunks for `query`."""
        results = self.retrieve(query, top_k=top_k)
        return results[0].score if results else 0.0