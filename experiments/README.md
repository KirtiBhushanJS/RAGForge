# Experiments

Controlled experiments changing **one variable at a time**. Results are written
to `experiments/experiment_results.csv` by `python scripts/run_experiments.py`.
Every number in that CSV comes from actually running retrieval against rebuilt
indexes - nothing is fabricated.

## Experiment 1 - Chunk size

Changes the chunking config and rebuilds the index for each size:
- 400 words
- 800 words (default)
- 1200 words

Expected insight: larger chunks give more context per hit but can blur a page
boundary; smaller chunks are more precise but may split an answer.

## Experiment 2 - Top-k

Keeps the default index and changes only retrieval depth:
- 3
- 5 (default)
- 8

Note: Hit Rate @ K is only reported where K <= top_k for that run.

## Experiment 3 (optional) - Embedding model

Pass `--embedding-models all-MiniLM-L6-v2 all-mpnet-base-v2` to compare
embedding models (requires downloading the models).

## Columns recorded

| Column | Meaning |
|--------|---------|
| `experiment` | which variable was changed |
| `parameter` | the value tested |
| `chunk_size` / `top_k` / `embedding_model` | full config snapshot |
| `hit_rate_at_3/5/8` | Hit Rate @ K |
| `mrr` | Mean Reciprocal Rank |
| `avg_retrieval_latency_ms` | average retrieval time per question |
| `index_build_seconds` | time to embed + index the corpus |
| `notes` | caveats for that row |

## How to interpret

Compare rows within one experiment only (everything else is held fixed). For
example a higher `hit_rate_at_3` for chunk_size=1200 means larger chunks helped
retrieval on your corpus; lower MRR at top_k=3 means the right source often
lands outside the top-3.

## Reproducibility note

Retrieval ranking is made deterministic within a process (sorted by score with
a chunk-id tie-breaker). On very small corpora (like the 3 sample PDFs), many
chunks receive near-identical scores, so the *exact* top-5 can occasionally
flip across processes due to floating-point noise in CPU embedding. On a real
10-20 document corpus with hundreds of chunks this effect is negligible.