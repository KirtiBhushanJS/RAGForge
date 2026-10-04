#!/usr/bin/env python3
"""
RAGForge evaluation metric runner.

Input:
  evaluation_results.csv

Required columns:
  experiment, question_id, retrieval_hit, answer_score

Optional:
  citation_correct, failure_type, failure_notes, top_k, chunk_size, embedding_model

answer_score convention:
  0 = incorrect
  1 = partially correct
  2 = correct

The script calculates:
  - Retrieval Hit Rate
  - Average Answer Quality (%)
  - Citation Accuracy (%), when available
  - Failure counts
and produces experiment_comparison.csv and failure_analysis.csv.
"""

from pathlib import Path
import pandas as pd

INPUT = Path("evaluation_results.csv")

df = pd.read_csv(INPUT)
required = {"experiment","question_id","retrieval_hit","answer_score"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

df["retrieval_hit"] = pd.to_numeric(df["retrieval_hit"], errors="coerce").fillna(0)
df["answer_score"] = pd.to_numeric(df["answer_score"], errors="coerce")

summary_rows = []
for experiment, g in df.groupby("experiment", dropna=False):
    hit_rate = g["retrieval_hit"].mean() * 100
    answer_quality = g["answer_score"].mean() / 2 * 100
    row = {
        "experiment": experiment,
        "questions": len(g),
        "retrieval_hit_rate_pct": round(hit_rate, 2),
        "answer_quality_pct": round(answer_quality, 2),
    }
    if "citation_correct" in g.columns:
        c = pd.to_numeric(g["citation_correct"], errors="coerce").dropna()
        if len(c):
            row["citation_accuracy_pct"] = round(c.mean() * 100, 2)
    for col in ["top_k","chunk_size","embedding_model"]:
        if col in g.columns:
            vals = g[col].dropna().astype(str).unique()
            row[col] = vals[0] if len(vals) == 1 else "; ".join(vals)
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
summary.to_csv("experiment_comparison.csv", index=False)

if "failure_type" in df.columns:
    failures = df[df["failure_type"].fillna("").astype(str).str.strip() != ""].copy()
    cols = [c for c in [
        "question_id","experiment","failure_type","failure_notes",
        "question","expected_document","expected_page",
        "retrieved_documents","retrieved_pages","generated_answer"
    ] if c in failures.columns]
    failures[cols].to_csv("failure_analysis.csv", index=False)

print("\n=== RAGForge Evaluation Summary ===")
print(summary.to_string(index=False))
print("\nFiles written:")
print("  experiment_comparison.csv")
if "failure_type" in df.columns:
    print("  failure_analysis.csv")
