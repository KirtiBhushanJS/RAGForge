"""Run the evaluation suite: retrieval metrics + answer-quality records.

Usage:
    python scripts/evaluate.py                         # default questions file
    python scripts/evaluate.py --questions evaluation/questions_sample.json
    python scripts/evaluate.py --top-k 5

Retrieval metrics (Hit Rate@K, MRR) run fully automatically against the
persistent FAISS index. Answer-quality evaluation calls the LLM for each
question (requires GEMINI_API_KEY) and writes a CSV for manual scoring.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.citations import format_citation_list  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.evaluation import (  # noqa: E402
    load_questions,
    measure_retrieval,
    retrieval_metrics_table,
    run_answer_evaluation,
)
from src.rag_pipeline import RAGPipeline  # noqa: E402
from src.utils import setup_logging  # noqa: E402

logger = logging.getLogger("rag.evaluate")


def _try_load_index(pipeline: RAGPipeline):
    if not pipeline.index_ready():
        print(
            "\nERROR: No FAISS index found. Run `python scripts/ingest.py` first,\n"
            "then re-run this script.\n"
        )
        sys.exit(1)


def print_retrieval_report(metrics) -> None:
    print("\n" + "=" * 64)
    print("RETRIEVAL EVALUATION (automated)")
    print("=" * 64)
    print(f"Questions evaluated  : {metrics.n_questions}")
    print(f"Questions scored     : {metrics.n_scored} "
          f"({(metrics.n_questions - metrics.n_scored)} without expected sources are excluded)")
    print(retrieval_metrics_table(metrics).to_string(index=False))
    print("=" * 64 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument("--questions", type=Path, default=None, help="path to questions.json")
    parser.add_argument("--top-k", type=int, default=None, help="retrieval depth")
    parser.add_argument("--no-answers", action="store_true", help="skip LLM answer generation")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    settings = get_settings()
    questions_path = args.questions or settings.evaluation_questions_path

    questions, examples = load_questions(questions_path)
    if not questions:
        print(
            f"\nNo real (non-example) questions found in {questions_path}.\n\n"
            "The evaluation file ships with example entries. To use this framework:\n"
            "  1. Add your final 10-20 PDFs to data/documents/\n"
            "  2. Run `python scripts/ingest.py`\n"
            "  3. Write 30+ real question/answer pairs in evaluation/questions.json\n"
            "     (keep expected_sources pointing at real document/page pairs)\n"
            "  4. Re-run `python scripts/evaluate.py`\n"
        )
        sys.exit(0)

    print(f"Loaded {len(questions)} questions (skipped {examples} example template entries).")

    pipeline = RAGPipeline(settings=settings)
    _try_load_index(pipeline)

    summary = pipeline.index_summary()
    print(f"Index: {summary['num_chunks']} chunks / {summary['num_documents']} documents "
          f"/ model {summary['model']}.")

    retriever = pipeline._load_index()  # noqa: SLF001 - script-level convenience

    # ---- Retrieval metrics (no LLM needed) --------------------------------
    metrics = measure_retrieval(retriever, questions, ks=(3, 5, 8), top_k=args.top_k)
    print_retrieval_report(metrics)

    # ---- per-question retrieval failures for failure analysis -------------
    from src.evaluation import expected_source_keys, has_expected_sources, question_text
    from src.utils import write_json

    failure_rows = []
    for q, hit in zip(questions, metrics.any_hits):
        if not hit and has_expected_sources(q):
            keys = sorted(expected_source_keys(q))
            failure_rows.append({
                "id": q.get("id"),
                "question": question_text(q),
                "expected_sources": [{"document": d, "page": p} for d, p in keys],
                "top_retrieved": [format_citation_list(retriever.retrieve(
                    question_text(q), top_k=args.top_k or settings.top_k))],
            })
    if failure_rows:
        failure_path = settings.evaluation_results_dir / "retrieval_failures.json"
        write_json(failure_path, failure_rows)
        print(f"\n{len(failure_rows)} retrieval miss(es) written to {failure_path} "
              "(use them for the failure analysis template).\n")
    else:
        print("\nNo retrieval misses to record.\n")

    # ---- Answer-quality evaluation (needs LLM) ----------------------------
    if args.no_answers or not pipeline.generator_configured():
        if not pipeline.generator_configured():
            print(
                "SKIPPING answer-quality evaluation: GEMINI_API_KEY is not set.\n"
                "Add it to .env and re-run to generate answers and the CSV."
            )
        raise SystemExit(0)

    print("Generating answers for each question (LLM)...")
    table, summary = run_answer_evaluation(
        pipeline,
        questions,
        settings.evaluation_results_dir,
        top_k=args.top_k,
    )
    print(f"Answer-quality CSV written to {summary['csv_path']}")
    print(f"Questions with generated answers: {len(table)}")
    print("\nAutomated metrics = retrieval Hit Rate@K / MRR (see above).\n"
          "Answer correctness/groundedness are manual 1-5 scores - review answer_quality.csv.\n")


if __name__ == "__main__":
    main()