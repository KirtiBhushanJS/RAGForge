# Document Q&A Assistant (RAG)

A production-quality **Retrieval-Augmented Generation (RAG)** system that answers
questions from a collection of PDF documents and **cites the exact source
document and page number** — with a real, runnable evaluation harness (retrieval
Hit Rate @ K, MRR, answer-quality CSV, controlled experiments).

> Everything the evaluation reports is produced by actually running the code on
> your documents. No numbers are fabricated anywhere.

---

## 1. Project overview

```
PDFs
 ↓
PyMuPDF text extraction (page numbers preserved)
 ↓
Text cleaning + chunking (configurable size/overlap, page-safe)
 ↓
Sentence-Transformers embeddings (all-MiniLM-L6-v2, L2-normalized)
 ↓
FAISS vector index + metadata.json (persistent)
 ↓
Top-K retrieval (cosine similarity, deduplicated sources)
 ↓
Confidence gate  ── below threshold ──▶ "I don't know based on the provided documents."
 ↓
Gemini answer generation (system prompt: answer only from context, cite [n])
 ↓
Answer + numbered citations (document + page)
 ↓
Streamlit UI (conversation history, expandable sources)
```

## 2. Problem statement

Question answering over private/technical documents is often done by copy-pasting
into a chatbot — unreliable, uncited, and impossible to audit. This project
builds an assistant that:

- retrieves the most relevant fragments from a local corpus of PDFs,
- only answers when retrieval confidence is high enough,
- grounds every answer in retrieved context with a strict system prompt,
- cites `document — page` for every claim,
- verifies its own answers with retrieval metrics and a manual quality rubric,
- measures the effect of chunk size and top-k via controlled experiments.

## 3. Architecture

| Layer | Component | Files |
|-------|-----------|-------|
| Data | PDFs → extracted pages | `src/pdf_loader.py` |
| Chunking | page-safe sliding windows | `src/chunker.py` |
| Embeddings | sentence-transformers wrapper | `src/embeddings.py` |
| Storage | FAISS + metadata.json | `src/vector_store.py` |
| Retrieval | top-k + dedup + threshold | `src/retriever.py` |
| Generation | Gemini with strict system prompt | `src/generator.py` |
| Citations | `[1] doc.pdf — Page 4` formatting & verification | `src/citations.py` |
| Orchestration | retrieve → gate → generate → cite | `src/rag_pipeline.py` |
| Evaluation | hit rate, MRR, quality CSV | `src/evaluation.py` |
| UI | Streamlit | `app.py` |
| Scripts | ingest / evaluate / experiments | `scripts/` |
| Tests | pytest (no API key required) | `tests/` |

## 4. Tech stack

- Python 3.11+ (developed on 3.13)
- Streamlit — frontend
- PyMuPDF (fitz) — PDF text extraction
- sentence-transformers — embeddings
- FAISS — vector search (inner-product over normalized vectors = cosine)
- Google Gemini API (`google-generativeai`) — answer generation
- python-dotenv — environment variables
- pandas / numpy — evaluation and experiments
- pytest — tests

No LangChain / LlamaIndex by design — the pipeline is plain, readable Python.

## 5. Folder structure

```
rag-document-qa/
├── app.py                     # Streamlit UI
├── requirements.txt
├── .env.example               # copy → .env
├── .gitignore                 # .env, indexes, results are ignored
├── README.md
├── data/
│   ├── documents/             # ← put your 10–20 PDFs here
│   └── processed/             # index.faiss, metadata.json, ingest_report.json
├── src/                       # modular pipeline code
├── scripts/
│   ├── ingest.py              # build/rebuild the vector index
│   ├── evaluate.py            # retrieval metrics + answer-quality CSV
│   ├── run_experiments.py     # controlled experiments → CSV
│   └── create_sample_pdfs.py  # OPTIONAL sample docs for local testing
├── evaluation/
│   ├── questions.json         # template for your 30+ real questions
│   ├── questions_sample.json  # 30+ questions generated for the sample PDFs
│   ├── results/               # answer_quality.csv, retrieval_failures.json
│   ├── FAILURE_ANALYSIS.md    # template (fill in after evaluation)
│   └── README.md
├── experiments/
│   ├── experiment_results.csv # real output of run_experiments.py
│   └── README.md
└── tests/                     # pytest suite (no API key needed)
```

## 6. Installation

```bash
cd rag-document-qa
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## 7. Environment setup

```bash
copy .env.example .env        # Windows
cp .env.example .env          # macOS / Linux
```

Open `.env` and set your Gemini key:

```
GEMINI_API_KEY=your_real_key_here
```

Get a free key at <https://aistudio.google.com/apikey>. Optional: set
`GEMINI_MODEL`, or pricing values to enable cost tracking. The `.env` file is
git-ignored — API keys are never hardcoded or committed.

> Note: `.env` must live at the **project root** (`rag-document-qa/.env`).

## 8. Adding PDFs

Drop your 10–20 `.pdf` files into:

```
data/documents/
```

- File names are read from the directory — nothing is hardcoded.
- Every chunk stores `chunk_id, document, document_path, page, text`, so page
  numbers survive ingestion and citation generation.
- Empty pages are skipped; malformed PDFs are logged and skipped with a summary
  in the ingest report.

> Optional (local testing only): `python scripts/create_sample_pdfs.py`
> generates 3 small fictional PDFs plus `evaluation/questions_sample.json`
> (30+ questions grounded in those exact pages). Delete them before real work.

## 9. Running ingestion

```bash
python scripts/ingest.py
# options: --chunk-size 600 --chunk-overlap 120 --model all-MiniLM-L6-v2
```

The first run downloads the embedding model (one-time). Output lands in
`data/processed/`:

```
index.faiss      # FAISS binary index
metadata.json    # chunk_id, document, document_path, page, text (index order)
ingest_report.json  # stats: pdfs, pages, chunks, avg chunk size, dim, time
```

The app loads this **persistent index** — PDFs are not re-embedded on every
Streamlit start. Rebuild any time your documents change.

## 10. Running Streamlit

```bash
streamlit run app.py
```

- **Sidebar:** top-k slider, similarity threshold, chunking config, number of
  indexed documents/chunks, embedding model, Gemini model + key status.
- **Main:** ask a question → answer with inline `[n]` citations, a numbered
  Sources list (`[1] doc.pdf — Page 4`), and expandable retrieved chunks
  (document, page, similarity score, chunk text, chunk ID).
- Conversation history is kept in `st.session_state`.
- If the API key is missing, the app still returns retrieved sources with a
  clear warning instead of crashing.

## 11. Running evaluation

```bash
python scripts/evaluate.py                              # default questions file
python scripts/evaluate.py --questions evaluation/questions_sample.json
python scripts/evaluate.py --top-k 5
```

- **Retrieval metrics (fully automated):** Hit Rate @ 3 / @ 5 / @ 8 and MRR,
  computed by embedding each question and searching the real index.
- **Answer quality:** generates answers with Gemini and writes
  `evaluation/results/answer_quality.csv` with `correctness_score` /
  `groundedness_score` columns left **blank for manual 1–5 review** — automated
  evaluation is clearly separated from human judgement.
- Retrieval misses are exported to `results/retrieval_failures.json` to drive
  the failure analysis.

### Writing the 30+ question set

`evaluation/questions.json` ships with two `example` entries that the scripts
skip. Replace them with your real questions **tied to actual pages of your
PDFs** (see `evaluation/README.md`). Keep a couple of out-of-scope questions
(empty `expected_sources`) to test the "I don't know" path.

## 12. Running experiments

```bash
python scripts/run_experiments.py
# options: --chunk-sizes 400 800 1200  --top-k-values 3 5 8
#          --embedding-models all-MiniLM-L6-v2 all-mpnet-base-v2
```

Changes **one variable at a time** and writes real results to
`experiments/experiment_results.csv`:

| experiment | parameter | hit_rate_at_3 | hit_rate_at_5 | hit_rate_at_8 | mrr | avg_retrieval_latency_ms | index_build_seconds | notes |
|---|---|---|---|---|---|---|---|---|

Every row comes from an actual retrieval run against freshly built indexes.

## 13. Evaluation metrics

| Metric | Definition | Automated? |
|--------|-----------|-----------|
| Hit Rate @ K | share of scored questions with an expected source in the top-K retrieved chunks | yes |
| MRR | mean over questions of `1 / rank_of_first_expected_source` (0 if missed) | yes |
| Citation correctness | all `[n]` numbers used by the model exist in the retrieved context | automated part |
| Correctness 1–5 | manual rubric against `expected_answer` | manual |
| Groundedness 1–5 | manual judgement: is every claim supported by retrieved text | manual |

Latency (ingestion, embedding, retrieval, generation, total) is measured with
`time.perf_counter` and reported in the UI and result CSVs.

## 14. Results table (placeholder)

Fill in after running the evaluation and experiments on your final PDF set:

| Setting | Hit Rate @ 3 | Hit Rate @ 5 | MRR | Notes |
|---------|-------------|-------------|-----|-------|
| chunk_size=800, top_k=5 (baseline) | | | | |
| chunk_size=400 | | | | |
| chunk_size=1200 | | | | |
| top_k=3 | | | | |
| top_k=8 | | | | |

## 15. Failure analysis

`evaluation/FAILURE_ANALYSIS.md` is a template covering: retrieval failures,
incorrect answers, insufficient context, ambiguous questions, poor chunking,
wrong page citations, hallucinations, and out-of-scope questions. Populate it
from real `retrieval_failures.json` output and genuine CSV reviews.

## 16. Limitations

- **Embedding quality vs. semantic nuance:** `all-MiniLM-L6-v2` is fast but not
  the strongest model; swap via `RAG_EMBEDDING_MODEL` if needed.
- **Page numbers** come from PyMuPDF's *physical* pages. If your PDFs have
  different printed numbering (e.g. "Page iii"), citations reflect the physical
  page index.
- **Scanned/image PDFs** yield no text; OCR is out of scope.
- **Similarity threshold** is a blunt gate — tune it for your corpus via the
  sidebar or `.env` (`RAG_SIMILARITY_THRESHOLD`).
- **Answer quality scoring is manual** by design; automated LLM-as-judge is a
  future improvement.
- The Gemini model can still hallucinate despite the strict prompt; hence
  citation verification + manual review.

## 17. Future improvements

- Hybrid retrieval (BM25 + dense) with reranking.
- Automatic LLM-as-judge groundedness scoring with human sampling.
- OCR fallback for scanned PDFs (Tesseract/PaddleOCR).
- Streaming answers, chat memory, and per-user document sets.
- Cost-aware evaluation using recorded token usage.

---

## Development / QA commands

```bash
# tests (no API key, no model downloads)
pytest -q

# rebuild index after changing PDFs
python scripts/ingest.py

# start the app
streamlit run app.py

# evaluate retrieval + answers
python scripts/evaluate.py

# run controlled experiments
python scripts/run_experiments.py
```