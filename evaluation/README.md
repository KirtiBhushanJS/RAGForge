# RAGForge Evaluation Kit

This package prepares the complete evaluation structure for the Faucone AI Engineer Intern Task 2.

## Included
- `evaluation_questions.csv` — 30-question evaluation-set structure
- `evaluation_results_template.csv` — per-question result schema
- `evaluate_metrics.py` — calculates baseline/experiment metrics from actual run results
- `EVALUATION_PLAN.md` — exact execution procedure
- `RAGForge_Evaluation_Report.md` — final report structure

## What is intentionally not fabricated
The task document and the existing RAGForge report establish that the project uses 13 smartphone manuals, 5,359 chunks, Qdrant, and Gemini, but they do not contain the actual 30 questions, retrieved chunks, generated answers, or measured experiment outputs.

Therefore this kit does **not** invent percentages or claim that experiments were run. Those values must come from the real RAG project.

## To finish the actual evaluation
Place the real 30+ questions and actual RAG outputs into `evaluation_results.csv`, then run:

```bash
python evaluate_metrics.py
```

The script will generate:
- `experiment_comparison.csv`
- `failure_analysis.csv`

The final report can then be filled with the measured values.
