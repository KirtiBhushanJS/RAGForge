"""Evaluation framework: retrieval metrics + answer-quality records.

What is honest to automate
--------------------------
- Retrieval metrics (Hit Rate@K, MRR) are fully automated and reproducible.
- Answer *quality* (correctness / groundedness) cannot be honestly automated
  with a rubric alone, so that layer produces CSV rows for MANUAL review and
  only automatically fills in citation correctness and metadata bookkeeping.

The scripts/run_experiments.py and scripts/evaluate.py use these functions, so
experiment numbers always come from real execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from src.retriever import RetrievedChunk, Retriever
from src.utils import ensure_dir, normalize_document_name, read_json, write_json

logger = logging.getLogger("rag.evaluation")

ANSWER_QUALITY_COLUMNS = [
    "id",
    "question",
    "expected_answer",
    "generated_answer",
    "retrieved_sources",
    "citation_correct",     # automated: all cited [n] exist in retrieval
    "correctness_score",    # manual rubric 1-5 (blank until reviewed)
    "groundedness_score",   # manual rubric 1-5 (blank until reviewed)
    "max_similarity",
    "is_unknown",
    "retrieval_latency_ms",
    "generation_latency_ms",
    "note",
]


# ---------------------------------------------------------------------------
# Question loading
# ---------------------------------------------------------------------------
def load_questions(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Load evaluation questions, excluding entries marked as examples.

    Returns (questions, skipped_examples).
    """
    raw = read_json(path, default=None)
    if raw is None:
        raise FileNotFoundError(f"Evaluation questions file not found: {path}")
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON array of question objects.")

    real: list[dict[str, Any]] = []
    examples = 0
    for item in raw:
        if not isinstance(item, dict):
            continue
        if item.get("example") is True:
            examples += 1
            continue
        missing = [k for k in ("id", "question", "expected_answer") if k not in item]
        if missing:
            logger.warning("Skipping evaluation item missing %s: %s", missing, item.get("question", "?"))
            continue
        real.append(item)
    real = sorted(real, key=lambda q: int(q.get("id", 0)))
    return real, examples


def question_text(q: dict[str, Any]) -> str:
    return str(q.get("question", "")).strip()


def has_expected_sources(q: dict[str, Any]) -> bool:
    sources = q.get("expected_sources")
    return bool(sources and isinstance(sources, list) and len(sources) > 0)


def expected_source_keys(q: dict[str, Any]) -> set[tuple[str, int]]:
    """Normalised {(document_name_lower, page)} for a question."""
    keys: set[tuple[str, int]] = set()
    for source in q.get("expected_sources", []) or []:
        doc = normalize_document_name(str(source.get("document", "")))
        page = int(source.get("page", 0))
        if doc:
            keys.add((doc, page))
    return keys


def chunk_is_relevant(chunk: RetrievedChunk, expected: set[tuple[str, int]]) -> bool:
    """True if a retrieved chunk matches any expected (document, page)."""
    key = (normalize_document_name(chunk.document), int(chunk.page))
    return key in expected


# ---------------------------------------------------------------------------
# Retrieval metrics (fully automated)
# ---------------------------------------------------------------------------
@dataclass
class RetrievalMetrics:
    n_questions: int
    n_scored: int                 # questions that have expected sources
    hit_rate_at: dict[int, float]  # {k: hit rate @ k}
    mrr: float
    any_hits: list[bool]
    ranks: list[int | None]       # rank of first expected hit (1-based) or None


def measure_retrieval(
    retriever: Retriever,
    questions: list[dict[str, Any]],
    ks: Sequence[int] = (3, 5),
    top_k: int | None = None,
    deduplicate: bool = True,
) -> RetrievalMetrics:
    """Run retrieval for every question and compute Hit Rate@K + MRR.

    Questions without expected sources are recorded but excluded from the
    metric denominators (they cannot be scored for retrieval).
    """
    n_scored = sum(1 for q in questions if has_expected_sources(q))
    ranks: list[int | None] = []
    any_hits: list[bool] = []
    hits: dict[int, int] = {k: 0 for k in ks}

    for q in questions:
        if not has_expected_sources(q):
            ranks.append(None)
            any_hits.append(False)
            continue
        expected = expected_source_keys(q)
        chunks = retriever.retrieve(question_text(q), top_k=top_k, deduplicate=deduplicate)
        rank = None
        for i, chunk in enumerate(chunks, start=1):
            if chunk_is_relevant(chunk, expected):
                rank = i
                break
        ranks.append(rank)
        hit = rank is not None
        any_hits.append(hit)
        if hit:
            for k in ks:
                if rank <= k:
                    hits[k] += 1

    hit_rates = {k: (hits[k] / n_scored) if n_scored else 0.0 for k in ks}
    reciprocal_ranks = [1.0 / r for r in ranks if r is not None]
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    return RetrievalMetrics(
        n_questions=len(questions),
        n_scored=n_scored,
        hit_rate_at=hit_rates,
        mrr=mrr,
        any_hits=any_hits,
        ranks=ranks,
    )


def retrieval_metrics_table(metrics: RetrievalMetrics) -> pd.DataFrame:
    rows = [{"metric": "Hit Rate @ 3", "value": metrics.hit_rate_at.get(3, 0.0),
             "definition": "fraction of scored questions with an expected source in top-3"},
            {"metric": "Hit Rate @ 5", "value": metrics.hit_rate_at.get(5, 0.0),
             "definition": "fraction of scored questions with an expected source in top-5"},
            {"metric": "Hit Rate @ 8", "value": metrics.hit_rate_at.get(8, 0.0),
             "definition": "fraction of scored questions with an expected source in top-8"},
            {"metric": "MRR", "value": metrics.mrr,
             "definition": "mean reciprocal rank of the first expected source"}]
    table = pd.DataFrame(rows)
    table["value"] = table["value"].round(4)
    return table


# ---------------------------------------------------------------------------
# Answer-quality records (automated scaffolding + manual rubric columns)
# ---------------------------------------------------------------------------
def run_answer_evaluation(
    pipeline: Any,
    questions: list[dict[str, Any]],
    results_dir: Path,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate an answer per question and write an analysis-ready CSV.

    The CSV includes both automated fields (retrieved sources, max similarity,
    citation correctness, latencies) and manual rubric columns
    (correctness_score, groundedness_score) that stay blank until reviewed.

    Correctness rubric (1-5):
      1 incorrect · 2 mostly incorrect · 3 partially correct · 4 mostly
      correct · 5 fully correct. Groundedness rubric (1-5): 1 = fully
      hallucinated, 5 = fully grounded in retrieved context.
    """
    ensure_dir(results_dir)
    records: list[dict[str, Any]] = []
    generated_count = 0

    for q in questions:
        result = pipeline.answer(question_text(q), top_k=top_k, similarity_threshold=similarity_threshold)
        retrieved_sources = ", ".join(
            f"{s.document} (page {s.page})" for s in result.sources
        )
        citation_correct = None
        if result.answer and not result.is_unknown:
            try:
                from src.citations import verify_citations

                check = verify_citations(result.answer, result.retrieved_chunks)
                citation_correct = bool(check.ok)
            except Exception:  # pragma: no cover - defensive
                citation_correct = None
        if not result.is_unknown:
            generated_count += 1
        records.append(
            {
                "id": q.get("id"),
                "question": question_text(q),
                "expected_answer": str(q.get("expected_answer", "")),
                "generated_answer": result.answer,
                "retrieved_sources": retrieved_sources,
                "citation_correct": citation_correct,
                "correctness_score": "",   # manual review
                "groundedness_score": "",  # manual review
                "max_similarity": round(result.max_similarity, 4),
                "is_unknown": result.is_unknown,
                "retrieval_latency_ms": round(result.retrieval_latency_s * 1000, 2),
                "generation_latency_ms": round(result.generation_latency_s * 1000, 2),
                "note": result.note,
            }
        )

    table = pd.DataFrame(records, columns=ANSWER_QUALITY_COLUMNS)
    csv_path = results_dir / "answer_quality.csv"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary = {
        "total_questions": len(records),
        "metrics": "Automated retrieval metrics are computed in evaluate_retrieval().",
        "correctness/groundedness": "Manual 1-5 scores go into answer_quality.csv columns.",
        "csv_path": str(csv_path),
    }
    return table, summary


def summarize_failure_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Rows that need manual attention (unknown answers or missing citations)."""
    if table.empty:
        return table
    mask = table["is_unknown"] | table["citation_correct"].isna()
    return table[mask]