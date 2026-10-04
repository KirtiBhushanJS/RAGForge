# RAGForge — Evaluation Report
### Document Q&A Assistant (RAG) | AI Engineer Intern Task 2

## 1. Objective
RAGForge is evaluated as a retrieval-augmented generation system for answering questions from a collection of smartphone manuals. The evaluation covers both the baseline requirement and the controlled Days 9–11 experiment.

## 2. Evaluation Setup
- Document collection: 13 smartphone manuals
- Indexed chunks: 5,359
- Vector database: Qdrant
- Generation: Gemini-powered RAG
- Evaluation set: 30+ manually prepared questions
- Primary retrieval metric: Retrieval Hit Rate
- Primary answer metric: Answer Quality

## 3. Baseline Evaluation
The same manually prepared question set is used for the baseline and every later experiment.

| Metric | Baseline |
|---|---:|
| Questions evaluated | PENDING ACTUAL RUN |
| Retrieval Hit Rate | PENDING ACTUAL RUN |
| Answer Quality | PENDING ACTUAL RUN |
| Citation Accuracy | PENDING ACTUAL RUN |

Retrieval Hit Rate measures whether the expected source was retrieved. Answer Quality uses a 0–2 reference-answer coverage scale: 0 incorrect, 1 partially correct, 2 correct.

## 4. Days 9–11 Controlled Experiment — Top-k
Only Top-k is changed. Documents, questions, embedding model, chunking, LLM, prompts, and scoring remain fixed.

| Top-k | Retrieval Hit Rate | Answer Quality |
|---:|---:|---:|
| 1 | PENDING ACTUAL RUN | PENDING ACTUAL RUN |
| 3 | PENDING ACTUAL RUN | PENDING ACTUAL RUN |
| 5 | PENDING ACTUAL RUN | PENDING ACTUAL RUN |

## 5. Failure Analysis

| Question ID | Failure Type | Observed Issue | Likely Cause |
|---|---|---|---|
| PENDING | PENDING | PENDING | PENDING |
| PENDING | PENDING | PENDING | PENDING |
| PENDING | PENDING | PENDING | PENDING |

Typical failure categories are retrieval miss, wrong document/page, insufficient context, answer-generation error, citation error, and out-of-scope question. The final version should replace the rows above with actual observed cases.

## 6. Conclusion
The evaluation is complete only after both layers are executed: the 30+ question baseline and the controlled Days 9–11 experiment. The final conclusion must be based on measured results and observed failure cases rather than assumptions.

## 7. Reproducibility
All experiments should use the same evaluation questions. Changes should be isolated to one variable at a time, with configuration and results recorded for every run.
