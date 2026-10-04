# RAGForge — Complete Evaluation Execution Plan

## What the task requires

The assignment has two evaluation layers:

1. **Baseline evaluation (Days 7–8 / Scope)**
   - At least 30 manually prepared question-answer pairs.
   - Retrieval Hit Rate.
   - Answer Quality.

2. **Controlled experiments (Days 9–11)**
   - Compare one retrieval variable at a time.
   - Recommended first experiment: Top-k = 1, 3, 5.
   - Keep the question set and all other RAG settings unchanged.
   - Record Retrieval Hit Rate and Answer Quality for every run.
   - Review representative failures.

The source task explicitly requires both layers and an evaluation report containing experiments, a results table, and failure analysis.

## Recommended execution order

### Run 0 — Baseline
Use the current production configuration.
Run all 30+ questions.
Record:
- retrieved document/page
- generated answer
- retrieval_hit (0/1)
- answer_score (0/1/2)
- citation_correct (0/1)
- failure_type and failure_notes when applicable

### Run 1 — Top-k = 1
Change only top-k to 1.
Run exactly the same questions.

### Run 2 — Top-k = 3
Change only top-k to 3.
Run exactly the same questions.

### Run 3 — Top-k = 5
Change only top-k to 5.
Run exactly the same questions.

### Optional second experiment — Chunk size
After the Top-k experiment, choose a controlled chunk-size comparison, for example:
- 300
- 500
- 800

Keep top-k, embedding model, LLM, overlap, documents, and questions fixed.

### Optional third experiment — Embedding model
Only if the existing project supports multiple embedding models cleanly.
Keep all other variables fixed.

## Scoring

### Retrieval Hit Rate
A question is a hit when the expected source document is present in the retrieved results (and the expected page is present when page-level ground truth is defined).

`Hit Rate = hits / total questions × 100`

### Answer Quality
Use:
- 2 = correct
- 1 = partially correct
- 0 = incorrect

`Answer Quality = mean(answer_score) / 2 × 100`

### Citation Accuracy
When citation checking is available:

`Citation Accuracy = correct citations / evaluated citations × 100`

## Failure taxonomy

Use one of:
- retrieval_miss
- wrong_document
- wrong_page
- insufficient_context
- answer_generation_error
- citation_error
- out_of_scope
- other

Do not hide unsuccessful cases. Include representative examples in the report.

## Important honesty rule

Do not invent experiment results. The final percentages must come from actual runs of the RAG system on the fixed evaluation set.
