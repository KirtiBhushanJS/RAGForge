# RAGForge — Evaluation Report

**Document Q&A Assistant (RAG) | AI Engineer Intern Task 2**

## 1. Objective

RAGForge is a retrieval-augmented generation system for answering questions about smartphone manuals. This report documents the evaluation setup, observed answer-generation behavior, known failure cases, and limitations.

## 2. Evaluation Setup

- Document collection: 13 smartphone manuals
- Indexed chunks: 5,359
- Vector database: Qdrant Cloud
- Embedding model: `gemini-embedding-2`
- Language model: `gemini-3.5-flash-lite`
- Evaluation dataset: 40 manually prepared questions
- Answerable questions: 32
- Unanswerable questions: 8
- Retrieval setting: Top-k = 5

## 3. Evaluation Execution

The updated-prompt evaluation completed all 40 questions without recorded execution errors.

Execution success does not establish answer correctness. Retrieval hit rate, answer quality, and citation accuracy have not been fully graded across all questions.

## 4. Abstention Results

The updated evaluation output was analyzed for configured abstention wording.

| Measure | Result |
|---|---:|
| Total questions | 40 |
| Answerable questions | 32 |
| Unanswerable questions | 8 |
| Answerable questions triggering abstention | 8/32 (25%) |
| Unanswerable questions triggering abstention | 7/8 (87.5%) |

These are preliminary phrase-based measurements, not overall accuracy metrics. Phrase matching may miss semantically equivalent refusals or fail to distinguish partial answers from complete answers.

One unanswerable question about the phone's current battery health received a limitation statement and instructions for checking battery information. This avoided inventing a live reading.

## 5. Observed Failure Cases

| Question ID | Failure type | Observed issue |
|---|---|---|
| Q02 | Retrieval / answer generation | The assistant abstained on the Galaxy A57 5G SIM question despite relevant material being retrieved. |
| Q06 | Retrieval / answer generation | The assistant abstained on the Galaxy A57 5G Wi-Fi question. |
| Q07 | Retrieval / answer generation | The assistant abstained on the Galaxy A57 5G battery-settings question. |
| Q09 | Retrieval / answer generation | The assistant abstained on the Galaxy A57 5G software-update question. |
| Q10 | Retrieval / answer generation | The assistant abstained on the Galaxy A57 5G screen-recording question. |
| Q11 | Retrieval | The assistant abstained on the Galaxy Z Fold8 Ultra screenshot question. |
| Q19 | Retrieval | The top-five results did not include the expected Motorola Edge 50 Wi-Fi passage. |
| Q28 | Retrieval | The OnePlus 8 screenshot passage was absent from the unrestricted top-five results. |
| Q39 | Abstention / response usefulness | The assistant acknowledged that live battery health was unavailable and offered a way to check battery information. |

These observations come from the recorded evaluation outputs and retrieval inspections. They are diagnostic observations, not fully adjudicated answer-quality scores.

## 6. Controlled Top-k Experiment

The planned Top-k = 1, 3, and 5 comparison has not been completed as a controlled experiment.

| Top-k | Retrieval Hit Rate | Answer Quality |
|---:|---|---|
| 1 | Not measured | Not measured |
| 3 | Not measured | Not measured |
| 5 | Not fully graded | Not fully graded |

No comparative conclusion should be drawn until the same questions are run at each setting, the outputs are graded, and all other RAG settings are held constant.

## 7. Limitations

- Successful execution does not guarantee correct answers.
- Retrieval may return passages from manuals for other devices.
- Some answers may cite a table of contents or related passage rather than the strongest supporting page.
- Overall answer quality and citation accuracy require grading against reference answers and pass conditions.
- The independent baseline history is not verified; existing files should not be presented as independent baseline runs without evidence.
- The current results do not establish overall answer accuracy.

## 8. Conclusion

RAGForge has a functioning ingestion, retrieval, and answer-generation workflow. The updated-prompt evaluation completed all 40 questions without recorded execution errors.

Preliminary abstention measurements show that the assistant frequently abstains on unsupported questions, but it also abstains unnecessarily on some answerable questions. Retrieval misses and citation-quality issues remain.

The project is suitable for handoff as a functional prototype with documented limitations. It should not be described as having measured overall answer accuracy or a completed Top-k comparison.

## 9. Reproducibility

The evaluation dataset is `evaluation_questions.csv`. The updated-prompt output is `evaluation_results_updated.csv`.

Use the same question set and configuration for comparisons. Preserve existing outputs and identify the prompt and code version used for each run.

Keep credentials out of the repository and report measured results without inventing missing metrics.