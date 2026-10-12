import csv
import os
import sys
import time
import random
from pathlib import Path

# Make the project root importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.retriever import retrieve_documents
from src.retrieval.answer_generator import generate_answer


EVALUATION_FILE = PROJECT_ROOT / "evaluation" / "evaluation_questions.csv"
RESULTS_FILE = PROJECT_ROOT / "evaluation" / "evaluation_results.csv"

TOP_K = 5

# Start conservatively. This spaces out generation calls.
REQUEST_INTERVAL_SECONDS = 5

# Retry temporary failures a limited number of times.
MAX_RETRIES = 3
INITIAL_RETRY_DELAY_SECONDS = 15

EXPECTED_COLUMNS = [
    "question_id",
    "category",
    "question",
    "expected_answer",
    "expected_document",
    "expected_page",
    "test_type",
    "pass_condition",
]

RESULT_COLUMNS = EXPECTED_COLUMNS + [
    "top_k",
    "retrieved_count",
    "retrieved_documents",
    "retrieved_pages",
    "top_similarity_score",
    "generated_answer",
    "error",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file, skipinitialspace=True))


def write_results(results):
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_FILE, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()

        for row in results:
            writer.writerow({
                column: row.get(column, "")
                for column in RESULT_COLUMNS
            })


def is_retryable_error(error_text):
    text = (error_text or "").lower()

    return (
        "429" in text
        or "resource_exhausted" in text
        or "503" in text
        or "unavailable" in text
        or "timeout" in text
        or "timed out" in text
    )


def retry_delay(error_text, attempt):
    """
    Use a conservative exponential delay.
    If the API error contains a suggested retry delay, use that
    information manually in a future refinement.
    """
    delay = INITIAL_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
    return delay + random.uniform(0, 3)


def evaluate_question(question_row):
    last_error = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            retrieved_documents = retrieve_documents(
                question_row["question"],
                top_k=TOP_K,
            )

            answer = generate_answer(
                question_row["question"],
                retrieved_documents,
            )

            retrieved_names = sorted({
                str(doc.get("document") or "")
                for doc in retrieved_documents
                if doc.get("document")
            })

            retrieved_pages = [
                str(doc.get("page") or "")
                for doc in retrieved_documents
            ]

            scores = [
                doc["score"]
                for doc in retrieved_documents
                if isinstance(doc.get("score"), (int, float))
            ]

            return {
                **question_row,
                "top_k": TOP_K,
                "retrieved_count": len(retrieved_documents),
                "retrieved_documents": "; ".join(retrieved_names),
                "retrieved_pages": "; ".join(retrieved_pages),
                "top_similarity_score": max(scores) if scores else "",
                "generated_answer": answer or "",
                "error": "",
            }

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

            if attempt >= MAX_RETRIES or not is_retryable_error(last_error):
                break

            delay = retry_delay(last_error, attempt)

            print(
                f"  Attempt {attempt}/{MAX_RETRIES} failed: "
                f"{last_error[:250]}"
            )
            print(f"  Waiting {delay:.1f} seconds before retrying...")
            time.sleep(delay)

    return {
        **question_row,
        "top_k": TOP_K,
        "retrieved_count": "",
        "retrieved_documents": "",
        "retrieved_pages": "",
        "top_similarity_score": "",
        "generated_answer": "",
        "error": last_error,
    }


def main():
    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation questions file not found: {EVALUATION_FILE}"
        )

    questions = read_csv(EVALUATION_FILE)

    if not questions:
        raise ValueError("The evaluation questions CSV is empty.")

    missing_columns = [
        column for column in EXPECTED_COLUMNS
        if column not in questions[0]
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required CSV columns: {missing_columns}"
        )

    # Load previous results so successful questions are not repeated.
    previous_results = {}
    if RESULTS_FILE.exists():
        for row in read_csv(RESULTS_FILE):
            question_id = row.get("question_id", "").strip()
            if question_id:
                previous_results[question_id] = row

    results_by_id = dict(previous_results)

    pending_questions = []
    for question in questions:
        question_id = question["question_id"].strip()
        previous = previous_results.get(question_id)

        # Skip successful results; retry only missing/failed questions.
        if previous and not previous.get("error", "").strip():
            continue

        pending_questions.append(question)

    print(f"Loaded {len(questions)} evaluation questions.")
    print(f"Questions needing evaluation/retry: {len(pending_questions)}")
    print(f"Top-k: {TOP_K}")
    print(f"Generation request spacing: {REQUEST_INTERVAL_SECONDS} seconds")
    print(f"Maximum attempts per question: {MAX_RETRIES}")
    print("Successful existing results will be preserved.")
    print()

    for index, question in enumerate(pending_questions, start=1):
        question_id = question["question_id"].strip()

        print(
            f"[{index}/{len(pending_questions)}] "
            f"{question_id}: {question['question']}"
        )

        result = evaluate_question(question)
        results_by_id[question_id] = result

        # Save immediately so progress survives interruptions.
        ordered_results = [
            results_by_id[q["question_id"].strip()]
            for q in questions
            if q["question_id"].strip() in results_by_id
        ]
        write_results(ordered_results)

        if result.get("error"):
            print(f"  FAILED: {result['error'][:250]}")
        else:
            print("  Completed successfully.")

        # Space out requests between questions.
        if index < len(pending_questions):
            time.sleep(REQUEST_INTERVAL_SECONDS)

    final_results = [
        results_by_id[q["question_id"].strip()]
        for q in questions
        if q["question_id"].strip() in results_by_id
    ]
    write_results(final_results)

    successes = sum(
        1 for row in final_results
        if not row.get("error", "").strip()
    )
    failures = len(final_results) - successes

    print()
    print("Evaluation run finished.")
    print(f"Results available: {len(final_results)} / {len(questions)}")
    print(f"Rows without execution errors: {successes}")
    print(f"Rows with execution errors: {failures}")
    print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
