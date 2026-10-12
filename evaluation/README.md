# RAGForge — Evaluation Guide

## 1. Purpose

This folder contains the evaluation dataset, evaluation runner, Qdrant index setup utility, recorded evaluation results, and final evaluation report for RAGForge.

## 2. Evaluation Dataset

`evaluation_questions.csv` contains 40 manually prepared questions:

- 32 answerable questions
- 8 unanswerable questions

Each question includes its expected answer, expected document, expected page where applicable, test type, and pass condition.

## 3. System Configuration

- Embedding model: `gemini-embedding-2`
- Language model: `gemini-3.5-flash-lite`
- Vector database: Qdrant Cloud
- Collection: `ragforge_documents`
- Retrieval setting used for the recorded run: Top-k = 5
- Ingestion chunk size: 800
- Chunk overlap: 120
- Indexed collection: 13 smartphone manuals and 5,359 chunks

## 4. Running the Evaluation

Run commands from the repository root with the project virtual environment activated.

```powershell
python evaluation\run_evaluation.py
```

The runner retrieves passages, generates answers, records retrieved document/page information, and saves results to a CSV file. It retries selected temporary API failures and saves progress during execution.

The default runner preserves successful existing results. Therefore, rerunning it does not automatically regenerate every answer after a prompt or code change.

For a fresh run, use the runner's explicit fresh-run option and select a new output filename. Never overwrite the recorded results unintentionally.

## 5. Files

- `evaluation_questions.csv` — evaluation questions and reference answers.
- `evaluation_results_updated.csv` — recorded output from the updated-prompt run.
- `run_evaluation.py` — evaluation runner.
- `create_document_index.py` — creates a keyword index on the Qdrant `document` payload field.
- `RAGForge_Evaluation_Report.md` — findings, observed failure cases, and limitations.

## 6. Metrics and Scoring

### Retrieval Hit Rate

A question is a hit when its expected source document appears in the retrieved results. Where page-level ground truth is defined, the expected page should also be checked.

`Hit Rate = hits / total questions × 100`

### Answer Quality

Grade each answer against its reference answer and pass condition:

- **2 — Correct:** accurate, sufficiently complete, and supported by the cited source.
- **1 — Partially correct:** useful but incomplete or containing a meaningful omission.
- **0 — Incorrect:** incorrect, unsupported, or fails to answer an answerable question.

`Answer Quality = mean(answer_score) / 2 × 100`

### Citation Accuracy

Check whether the cited document and page support the answer.

`Citation Accuracy = correct citations / evaluated citations × 100`

These metrics require actual grading. The recorded output CSV does not contain completed answer-quality or citation scores.

## 7. Controlled Experiments

The planned Top-k experiment compares values of 1, 3, and 5 while keeping the question set and other RAG settings unchanged.

This controlled comparison has not been completed. No comparative results should be reported until those runs have been performed and graded.

Optional future experiments include chunk-size comparisons and embedding-model comparisons. Change only one variable at a time.

## 8. Failure Categories

Use these categories when reviewing failures:

- `retrieval_miss`
- `wrong_document`
- `wrong_page`
- `insufficient_context`
- `answer_generation_error`
- `citation_error`
- `out_of_scope`
- `other`

Record representative failures rather than hiding unsuccessful cases.

## 9. Current Findings

The updated-prompt run completed all 40 questions without recorded execution errors.

Preliminary phrase-based abstention analysis found:

- 8 of 32 answerable questions triggered abstention wording.
- 7 of 8 unanswerable questions triggered abstention wording.

These figures are abstention measurements, not overall accuracy scores. Retrieval hit rate, answer quality, and citation accuracy have not been fully graded across all questions.

## 10. Limitations and Reproducibility

- Successful execution does not guarantee a correct answer.
- Retrieval may return passages from other device manuals.
- Some citations may refer to a table of contents or a related passage instead of the strongest supporting page.
- The independent baseline history is not verified.
- Preserve previous results before rerunning evaluations.
- Record the prompt, code version, dataset, and configuration used for each run.
- Do not invent unmeasured metrics.
- Keep API keys in `.env` and out of version control.