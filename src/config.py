"""Central configuration for the RAG pipeline.

Every tunable parameter lives here (or can be overridden via environment
variables), so the pipeline can be configured without touching core code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Project root = parents[1] of this file (src/config.py -> rag-document-qa/).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    """Runtime configuration for the whole project."""

    # ------------------------------------------------------------------
    # Paths (relative to the project root so the code is not cwd-sensitive)
    # ------------------------------------------------------------------
    project_root: Path = PROJECT_ROOT
    documents_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "documents")
    processed_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
    index_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "index.faiss")
    metadata_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "metadata.json")
    ingest_report_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "ingest_report.json")
    evaluation_questions_path: Path = field(default_factory=lambda: PROJECT_ROOT / "evaluation" / "questions.json")
    evaluation_results_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "evaluation" / "results")
    experiments_path: Path = field(default_factory=lambda: PROJECT_ROOT / "experiments" / "experiment_results.csv")

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    chunk_size: int = 800          # approximate words per chunk
    chunk_overlap: int = 150       # overlapping words between neighbouring chunks
    min_chunk_words: int = 60      # trailing windows smaller than this are merged back

    # ------------------------------------------------------------------
    # Embeddings / retrieval
    # ------------------------------------------------------------------
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 5                 # default number of chunks retrieved
    similarity_threshold: float = 0.40  # below this the pipeline answers "I don't know"
    deduplicate_sources: bool = True    # keep only best chunk per (document, page)

    # ------------------------------------------------------------------
    # Gemini generation
    # ------------------------------------------------------------------
    gemini_model: str = "gemini-1.5-flash"
    gemini_api_key: str = ""
    gemini_max_retries: int = 3
    gemini_temperature: float = 0.2

    # Optional cost tracking (USD per 1M tokens). 0 => cost is not calculated.
    gemini_input_cost_per_1m: float = 0.0
    gemini_output_cost_per_1m: float = 0.0

    def __post_init__(self) -> None:
        override_map = {
            "chunk_size": ("RAG_CHUNK_SIZE", _get_int),
            "chunk_overlap": ("RAG_CHUNK_OVERLAP", _get_int),
            "min_chunk_words": ("RAG_MIN_CHUNK_WORDS", _get_int),
            "embedding_model": ("RAG_EMBEDDING_MODEL", lambda k, d: os.getenv(k, d)),
            "top_k": ("RAG_TOP_K", _get_int),
            "similarity_threshold": ("RAG_SIMILARITY_THRESHOLD", _get_float),
            "deduplicate_sources": ("RAG_DEDUPLICATE_SOURCES", _get_bool),
            "gemini_model": ("GEMINI_MODEL", lambda k, d: os.getenv(k, d)),
            "gemini_max_retries": ("GEMINI_MAX_RETRIES", _get_int),
            "gemini_temperature": ("GEMINI_TEMPERATURE", _get_float),
            "gemini_input_cost_per_1m": ("GEMINI_INPUT_COST_PER_1M", _get_float),
            "gemini_output_cost_per_1m": ("GEMINI_OUTPUT_COST_PER_1M", _get_float),
        }
        for attr, (env_name, parser) in override_map.items():
            object.__setattr__(self, attr, parser(env_name, getattr(self, attr)))
        # API key read last so .env is already loaded above.
        object.__setattr__(self, "gemini_api_key", os.getenv("GEMINI_API_KEY", ""))


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a lazily created, shared Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings