"""One-shot ingestion pipeline.

Usage:
    python scripts/ingest.py                  # use config defaults
    python scripts/ingest.py --chunk-size 600 --chunk-overlap 120
    python scripts/ingest.py --model all-MiniLM-L6-v2

The script builds data/processed/index.faiss + metadata.json from every PDF in
data/documents and prints ingestion statistics.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunker import chunk_documents, chunk_statistics  # noqa: E402
from src.config import Settings, get_settings  # noqa: E402
from src.embeddings import get_embedding_model  # noqa: E402
from src.pdf_loader import load_documents, flatten_pages  # noqa: E402
from src.utils import setup_logging  # noqa: E402
from src.vector_store import build_index, save_index  # noqa: E402

logger = logging.getLogger("rag.ingest")


def _build_settings(args) -> Settings:
    settings = get_settings()
    overrides = {}
    if args.chunk_size is not None:
        overrides["chunk_size"] = args.chunk_size
    if args.chunk_overlap is not None:
        overrides["chunk_overlap"] = args.chunk_overlap
    if args.min_chunk_words is not None:
        overrides["min_chunk_words"] = args.min_chunk_words
    if args.model:
        overrides["embedding_model"] = args.model
    if overrides:
        settings = Settings(**{**settings.__dict__, **overrides})
    return settings


def run_ingestion(settings: Settings) -> dict:
    """Execute the ingest pipeline and return a stats dictionary."""
    started = time.perf_counter()

    # 1. Load PDFs ----------------------------------------------------------
    documents, loader_report = load_documents(settings.documents_dir)
    if not documents:
        print(
            "\nERROR: No readable PDFs found in:\n"
            f"  {settings.documents_dir}\n"
            "Place your .pdf files there and run this script again.\n"
        )
        sys.exit(1)

    pages = flatten_pages(documents)

    # 2. Chunk --------------------------------------------------------------
    chunks = chunk_documents(
        pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        min_chunk_words=settings.min_chunk_words,
        documents_dir=settings.documents_dir,
    )
    logger.info("Created %d chunks.", len(chunks))

    # 3. Embed + build FAISS index ------------------------------------------
    embedder = get_embedding_model(settings.embedding_model)
    index = build_index(chunks, embedder, settings)
    dim = int(index.d)

    # 4. Persist + report ----------------------------------------------------
    ingest_stats = {
        "num_pdfs_attempted": loader_report.files_attempted,
        "num_pdfs_loaded": loader_report.files_loaded,
        "num_pdfs_failed": loader_report.files_failed,
        "num_pages": loader_report.pages_extracted,
        "empty_pages_skipped": loader_report.empty_pages_skipped,
        **chunk_statistics(chunks),
    }
    save_index(
        index=index,
        chunks=chunks,
        index_path=settings.index_path,
        metadata_path=settings.metadata_path,
        ingest_report_path=settings.ingest_report_path,
        settings=settings,
        ingest_stats=ingest_stats,
    )

    elapsed = time.perf_counter() - started
    print_stats(loader_report, ingest_stats, dim, elapsed)
    return {"stats": ingest_stats, "dim": dim, "elapsed_s": elapsed}


def print_stats(loader_report, stats: dict, dim: int, elapsed: float) -> None:
    print("\n" + "=" * 56)
    print("INGESTION COMPLETE")
    print("=" * 56)
    print(f"Number of PDFs attempted   : {stats['num_pdfs_attempted']}")
    print(f"Number of PDFs loaded      : {stats['num_pdfs_loaded']}")
    print(f"Number of PDFs failed      : {stats['num_pdfs_failed']}")
    print(f"Number of pages extracted  : {stats['num_pages']}")
    print(f"Empty pages skipped        : {stats['empty_pages_skipped']}")
    print(f"Number of chunks           : {stats['num_chunks']}")
    print(f"Average chunk size (words) : {stats['avg_chunk_words']}")
    print(f"Embedding dimension        : {dim}")
    print(f"Processing time            : {elapsed:.2f}s")
    if loader_report.failed_files:
        print(f"Failed files               : {', '.join(loader_report.failed_files)}")
    print("=" * 56)
    print(f"Index     -> {Path('data/processed/index.faiss')}")
    print(f"Metadata  -> {Path('data/processed/metadata.json')}")
    print(f"Report    -> {Path('data/processed/ingest_report.json')}")
    print("=" * 56 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs and build the FAISS index.")
    parser.add_argument("--chunk-size", type=int, default=None, help="words per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="overlap words")
    parser.add_argument("--min-chunk-words", type=int, default=None, help="merge trailing chunks below this")
    parser.add_argument("--model", type=str, default=None, help="sentence-transformers model name")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    run_ingestion(_build_settings(args))


if __name__ == "__main__":
    main()