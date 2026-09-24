"""Embedding model wrapper (sentence-transformers), lazy-loaded and normalized.

The heavy model is loaded on first use so importing the module (and starting
Streamlit) stays fast. Vectors are L2-normalised so FAISS inner-product search
equals cosine similarity.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import numpy as np

from src.config import get_settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger("rag.embeddings")


class EmbeddingModel:
    """Thin wrapper around a sentence-transformers model."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or get_settings().embedding_model
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        """Load the underlying model once (thread-safe)."""
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model '%s'...", self.model_name)
                self._model = SentenceTransformer(self.model_name)
        return self._model

    def dimension(self) -> int:
        """Embedding dimensionality (loads the model if needed)."""
        return int(self._load().get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return a float32 matrix of L2-normalised embeddings, shape (n, dim)."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        model = self._load()
        vectors = model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
        vectors = np.asarray(vectors, dtype=np.float32)
        # Apply the L2 norm again defensively (cheap, guarantees unit vectors).
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query into a (1, dim) normalized vector."""
        if not query.strip():
            raise ValueError("Empty query cannot be embedded.")
        return self.encode([query])


_embedder: EmbeddingModel | None = None
_embedder_lock = threading.Lock()


def get_embedding_model(model_name: str | None = None) -> EmbeddingModel:
    """Return a process-wide shared EmbeddingModel instance."""
    global _embedder
    key = model_name or get_settings().embedding_model
    if _embedder is not None and _embedder.model_name == key:
        return _embedder
    with _embedder_lock:
        if _embedder is None or _embedder.model_name != key:
            _embedder = EmbeddingModel(key)
    return _embedder