import csv
import sys
import time
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.retriever import retrieve_documents
from src.retrieval.answer_generator import generate_answer

EVALUATION_FILE = PROJECT_ROOT / "evaluation" / "evaluation_questions.csv"
RESULTS_FILE = PROJECT_ROOT / "evaluation" / "evaluation_results_updated.csv"

TOP_K = 5
REQUEST_INTERVAL_SECONDS = 5
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
    with open(RESULTS_FILE, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(
            {column: row.get(column, "") for column in RESULT_COLUMNS}
            for row in results
        )


def retry_delay(error_text, attempt):
    return INITIAL_RETRY_DELAY_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 3)


def is_retryable_error(error_text):
    text = (error_text or "").lower()
    return any(
        marker in text
        for marker in (
            "429", "resource_exhausted", "503",
            "unavailable", "timeout", "timed out",
        )
    )


def evaluate_question(question):
    last_error = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            documents = retrieve_documents(question["question"], top_k=TOP_K)
            answer = generate_answer(question["question"], documents)

            names = sorted({
                str(doc.get("document") or "")
                for doc in documents
                if doc.get("document")
            })
            pages = [str(doc.get("page") or "") for doc in documents]
            scores = [
                doc["score"] for doc in documents
                if isinstance(doc.get("score"), (int, float))
            ]

            return {
                **question,
                "top_k": TOP_K,
                "retrieved_count": len(documents),
                "retrieved_documents": "; ".join(names),
                "retrieved_pages": "; ".join(pages),
                "top_similarity_score": max(scores) if scores else "",
                "generated_answer": answer or "",
                "error": "",
            }

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

            if attempt == MAX_RETRIES or not is_retryable_error(last_error):
                break

            delay = retry_delay(last_error, attempt)
            print(f"  Attempt {attempt} failed: {last_error[:200]}")
            print(f"  Retrying in {delay:.1f} seconds...")
            time.sleep(delay)

    return {
        **question,
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
        raise FileNotFoundError(f"Missing evaluation file: {EVALUATION_FILE}")

    questions = read_csv(EVALUATION_FILE)
    if not questions:
        raise ValueError("Evaluation questions file is empty.")

    missing = [column for column in EXPECTED_COLUMNS if column not in questions[0]]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if RESULTS_FILE.exists():
        raise FileExistsError(
            f"{RESULTS_FILE.name} already exists. Rename or back it up before rerunning."
        )

    print(f"Loaded {len(questions)} questions.")
    print(f"Output: {RESULTS_FILE}")
    print("All questions will be evaluated; no previous results will be skipped.")
    print()

    results = []

    for index, question in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] {question['question_id']}: {question['question']}")
        result = evaluate_question(question)
        results.append(result)
        write_results(results)

        if result["error"]:
            print(f"  FAILED: {result['error'][:200]}")
        else:
            print("  Completed.")

        if index < len(questions):
            time.sleep(REQUEST_INTERVAL_SECONDS)

    successes = sum(not row["error"] for row in results)
    print("\nEvaluation finished.")
    print(f"Results available: {len(results)} / {len(questions)}")
    print(f"Rows without execution errors: {successes}")
    print(f"Rows with execution errors: {len(results) - successes}")
    print(f"Saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
