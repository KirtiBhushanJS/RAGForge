"""Controlled experiments: change ONE variable at a time.

Experiment 1 - chunk size   : {400, 800, 1200}  (index rebuilt per size)
Experiment 2 - top-k        : {3, 5, 8}        (default index, retrieval depth varies)
Experiment 3 - (optional)   : embedding model if --embedding-models is given

All numbers come from actually running retrieval against the built indexes and
are appended to experiments/experiment_results.csv. Nothing is fabricated.

Usage:
    python scripts/run_experiments.py
    python scripts/run_experiments.py --chunk-sizes 400 800 1200
    python scripts/run_experiments.py --embedding-models all-MiniLM-L6-v2 all-mpnet-base-v2
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.chunker import chunk_documents  # noqa: E402
from src.config import Settings, get_settings  # noqa: E402
from src.embeddings import get_embedding_model  # noqa: E402
from src.evaluation import expected_source_keys, load_questions, measure_retrieval  # noqa: E402
from src.pdf_loader import flatten_pages, load_documents  # noqa: E402
from src.retriever import Retriever  # noqa: E402
from src.utils import ensure_dir, setup_logging  # noqa: E402
from src.vector_store import build_index, save_index, load_index  # noqa: E402

logger = logging.getLogger("rag.experiments")

RESULTS_COLUMNS = [
    "experiment",
    "parameter",
    "chunk_size",
    "top_k",
    "embedding_model",
    "num_questions",
    "hit_rate_at_3",
    "hit_rate_at_5",
    "hit_rate_at_8",
    "mrr",
    "avg_retrieval_latency_ms",
    "index_build_seconds",
    "notes",
]


def _load_pages(settings: Settings):
    documents, report = load_documents(settings.documents_dir)
    if not documents:
        print(
            "\nERROR: No readable PDFs found in data/documents/. "
            "Add PDFs and run `python scripts/ingest.py` at least once.\n"
        )
        sys.exit(1)
    return flatten_pages(documents), report


def _build_retriever(pages, settings: Settings, tmp_dir: Path):
    """Chunk, embed, persist (to temp), load and return (retriever, build_s)."""
    chunks = chunk_documents(
        pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        min_chunk_words=settings.min_chunk_words,
        documents_dir=settings.documents_dir,
    )
    embedder = get_embedding_model(settings.embedding_model)
    t0 = time.perf_counter()
    index = build_index(chunks, embedder, settings)
    build_s = round(time.perf_counter() - t0, 2)

    index_path = tmp_dir / "index.faiss"
    metadata_path = tmp_dir / "metadata.json"
    report_path = tmp_dir / "ingest_report.json"
    save_index(index, chunks, index_path, metadata_path, report_path, settings)
    loaded_index, payload = load_index(index_path, metadata_path, settings)
    return Retriever(loaded_index, payload, embedder, settings=settings), build_s


def _measure(retriever: Retriever, questions, top_k, notes: str) -> dict:
    t0 = time.perf_counter()
    metrics = measure_retrieval(retriever, questions, ks=(3, 5, 8), top_k=top_k)
    avg_latency_ms = round(
        (time.perf_counter() - t0) / max(1, len(questions)) * 1000, 2
    )

    def hr(k):
        if k > (top_k or retriever.settings.top_k):
            return float("nan")
        return round(metrics.hit_rate_at.get(k, 0.0), 4)

    return {
        "hit_rate_at_3": hr(3),
        "hit_rate_at_5": hr(5),
        "hit_rate_at_8": hr(8),
        "mrr": round(metrics.mrr, 4),
        "avg_retrieval_latency_ms": avg_latency_ms,
        "num_questions": metrics.n_questions,
        "notes": notes,
    }


def run_experiments(settings: Settings, args) -> pd.DataFrame:
    ensure_dir(settings.experiments_path.parent)
    questions_file = args.questions or settings.evaluation_questions_path
    questions, examples = load_questions(questions_file)
    if not questions:
        print(
            f"\nNo real (non-example) questions found in {questions_file}.\n"
            "Write your real evaluation questions first, then re-run experiments.\n"
        )
        sys.exit(1)
    print(f"Using {len(questions)} questions from {questions_file}.")

    pages, _ = _load_pages(settings)
    rows: list[dict] = []

    # ---- Experiment 1: chunk size ----------------------------------------
    for chunk_size in args.chunk_sizes:
        exp_settings = Settings(**{**settings.__dict__, "chunk_size": chunk_size})
        with tempfile.TemporaryDirectory(prefix="rag-exp-") as tmp:
            retriever, build_s = _build_retriever(pages, exp_settings, Path(tmp))
            measure = _measure(retriever, questions, args.base_top_k,
                               f"index embedded with chunk_size={chunk_size}")
            rows.append({
                "experiment": "chunk_size",
                "parameter": chunk_size,
                "chunk_size": chunk_size,
                "top_k": args.base_top_k,
                "embedding_model": settings.embedding_model,
                "index_build_seconds": build_s,
                **measure,
            })
        print(f"  chunk_size={chunk_size}: "
              f"hit@3={measure['hit_rate_at_3']}, hit@5={measure['hit_rate_at_5']}, "
              f"mrr={measure['mrr']} (index build {build_s}s)")

    # ---- Experiment 2: top-k ----------------------------------------------
    # Base index is built once at the configured (default) chunk size.
    with tempfile.TemporaryDirectory(prefix="rag-exp-") as tmp:
        retriever, build_s = _build_retriever(pages, settings, Path(tmp))
        for top_k in args.top_k_values:
            measure = _measure(retriever, questions, top_k,
                               "hit@k reported only where k <= top_k")
            rows.append({
                "experiment": "top_k",
                "parameter": top_k,
                "chunk_size": settings.chunk_size,
                "top_k": top_k,
                "embedding_model": settings.embedding_model,
                "index_build_seconds": build_s,
                **measure,
            })
            print(f"  top_k={top_k}: hit@3={measure['hit_rate_at_3']}, "
                  f"hit@5={measure['hit_rate_at_5']}, mrr={measure['mrr']}")

    # ---- Experiment 3 (optional): embedding model -------------------------
    for model in args.embedding_models:
        exp_settings = Settings(**{**settings.__dict__, "embedding_model": model})
        with tempfile.TemporaryDirectory(prefix="rag-exp-") as tmp:
            retriever, build_s = _build_retriever(pages, exp_settings, Path(tmp))
            measure = _measure(retriever, questions, args.base_top_k,
                               f"embedding model {model}")
            rows.append({
                "experiment": "embedding_model",
                "parameter": model,
                "chunk_size": settings.chunk_size,
                "top_k": args.base_top_k,
                "embedding_model": model,
                "index_build_seconds": build_s,
                **measure,
            })
            print(f"  embedding_model={model}: hit@3={measure['hit_rate_at_3']}, "
                  f"hit@5={measure['hit_rate_at_5']}, mrr={measure['mrr']}")

    table = pd.DataFrame(rows, columns=RESULTS_COLUMNS)
    if args.append and settings.experiments_path.exists():
        existing = pd.read_csv(settings.experiments_path)
        table = pd.concat([existing, table], ignore_index=True)
    table.to_csv(settings.experiments_path, index=False, encoding="utf-8-sig")
    print(f"\nResults written to {settings.experiments_path}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled RAG experiments.")
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[400, 800, 1200])
    parser.add_argument("--top-k-values", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--embedding-models", type=str, nargs="+", default=[])
    parser.add_argument("--base-top-k", type=int, default=None,
                        help="retrieval depth used for chunk-size and model experiments")
    parser.add_argument("--append", action="store_true", help="append to existing CSV")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    run_experiments(get_settings(), args)


if __name__ == "__main__":
    main()