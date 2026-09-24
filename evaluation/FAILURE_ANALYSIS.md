# Failure Analysis Template

**Status:** template - fill in during/after a real evaluation run.

This document records actual failures observed when running
`python scripts/evaluate.py` and the manual answer-quality review
(`evaluation/results/answer_quality.csv`). Do **not** invent failures - only
copy entries that really happened during a run.

## 1. Retrieval failures (from `results/retrieval_failures.json`)

| Question ID | Question | Expected source (doc, page) | Why it likely failed | Fix |
|-------------|----------|-----------------------------|----------------------|-----|
| *(e.g. 7)* | *(paste question)* | *(doc.pdf, p. 4)* | *(chunking splits the answer across two chunks / question phrasing too specific / page uses a table image with no text)* | *(increase chunk overlap / rephrase question / OCR the PDF)* |

## 2. Incorrect answers (manual review)

| Question ID | Expected answer | Generated answer | Correctness (1–5) | Root cause | Fix |
|-------------|-----------------|------------------|-------------------|------------|-----|
|             |                 |                  |                   |            |     |

## 3. Insufficient context / "I don't know" false negatives

| Question ID | Question | Max similarity | Threshold used | Was the answer actually in the docs? |
|-------------|----------|----------------|----------------|--------------------------------------|
|             |          |                |                |                                      |

## 4. Ambiguous questions

| Question ID | Question | Why ambiguous | Suggested rephrase |
|-------------|----------|---------------|--------------------|
|             |          |               |                    |

## 5. Poor chunking

| Chunk size tried | Symptom | Evidence (snippet of a bad chunk) | Proposed change |
|------------------|---------|-----------------------------------|-----------------|
|                  |         |                                   |                 |

## 6. Wrong page citation

| Question | Cited page | Page where info actually is | Cause |
|----------|------------|-----------------------------|-------|
|          |            |                             |       |

## 7. Hallucinations

| Question | Hallucinated claim | Where in the context should it have come from | Action |
|----------|--------------------|-----------------------------------------------|--------|
|          |                    |                                               |        |

## 8. Questions outside document scope

| Question | Expected: "I don't know"? | Did the system answer? | Note |
|----------|---------------------------|------------------------|------|
|          |                           |                        |      |