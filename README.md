# RAGForge — Document Q&A Assistant

RAGForge is a Retrieval-Augmented Generation (RAG) application that answers questions about smartphone user manuals. It retrieves relevant passages from a Qdrant vector database and uses Google Gemini to generate answers with document and page references.

This project was developed as an AI engineering task and is presented as a functional prototype. Answer correctness and citation quality still require further evaluation.

## Features

- PDF text extraction and chunking
- Gemini-based text embeddings
- Semantic retrieval from Qdrant Cloud
- Gemini-generated answers with source references
- Streamlit user interface
- Evaluation dataset covering answerable and unanswerable questions

## Document Collection

The indexed collection contains **13 smartphone manuals** from Samsung, Motorola, and OnePlus, represented by **5,359 chunks** in Qdrant.

The original manufacturer PDFs are not included in this repository. Obtain them from official manufacturer sources and place them in `data/documents/`. The local filenames must be consistent with the document-loading and ingestion code.

### Samsung

| Model | Official manual |
|---|---|
| Galaxy A57 5G | [Samsung PDF](https://downloadcenter.samsung.com/content/UM/202603/20260326125812922/SM-A576B_A376_UG_EU_16_Eng_Rev.1.0_260126.pdf) |
| Galaxy S25 Ultra | [Samsung PDF](https://downloadcenter.samsung.com/content/UM/202509/20250929135046516/SM-S93X_S92X_UG_EU_16_Eng_Rev.1.0_250929.pdf) |
| Galaxy S26 Ultra | [Samsung PDF](https://downloadcenter.samsung.com/content/UM/202602/20260227095338477/SM-S94X_UG_EU_16_Eng_Rev.1.0_260220.pdf) |
| Galaxy Z Flip7 | [Samsung PDF](https://downloadcenter.samsung.com/content/UM/202507/20250711090617100/SM-F766B_F761B_UG_EU_16_Eng_Rev.1.0_250709.pdf) |
| Galaxy Z Fold8 Ultra | [Samsung PDF](https://downloadcenter.samsung.com/content/UM/202607/20260723101548322/SM-F976B_F971B_F776B_UG_EU_17_Eng_Rev.1.0_260708.pdf) |

### Motorola

| Model | Official manual |
|---|---|
| Motorola Edge 50 | [Motorola PDF](https://help.motorola.com/hc/7156/16/pdf/help-motorola-edge-50-16-global-en-gb.pdf) |
| Motorola Edge 60 | [Motorola PDF](https://help.motorola.com/hc/7162/16/pdf/help-motorola-edge-60-16-global-en-gb.pdf) |
| Motorola Edge 70 Pro+ | [Motorola PDF](https://help.motorola.com/hc/7175/16/pdf/help-motorola-edge-70-pro-plus-16-global-en-gb.pdf) |
| Motorola Signature | [Motorola PDF](https://help.motorola.com/hc/6200/16/pdf/help-motorola-signature-16-global-en-gb.pdf) |

### OnePlus

| Model | Official manual |
|---|---|
| OnePlus 8 | [OnePlus manual](https://service.oneplus.com/in/user-manual#/detail?equipmentModelId=9249212a-a0ce-11ef-9811-fa168f20b2a1) |
| OnePlus 9 | [OnePlus manual](https://service.oneplus.com/in/user-manual#/detail?equipmentModelId=924926b0-a0ce-11ef-9811-fa168f20b2a1) |
| OnePlus 13 | [OnePlus manual](https://service.oneplus.com/in/user-manual#/detail?equipmentModelId=1875066204395962370) |
| OnePlus 15 | [OnePlus manual](https://service.oneplus.com/in/user-manual#/detail?equipmentModelId=1979231416971198465) |

The PDFs remain local for educational and evaluation purposes. Do not commit them unless redistribution permission has been confirmed.

## Technology

- **Language:** Python
- **Interface:** Streamlit
- **LLM:** Google Gemini
- **Embeddings:** `gemini-embedding-2`
- **Vector database:** Qdrant Cloud
- **Document processing:** PDF text extraction and chunking

## Setup

Run commands from the repository root in PowerShell.

### 1. Install dependencies

Activate your virtual environment, then run:

```powershell
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a local `.env` file in the repository root:

```env
GEMINI_API_KEY=your_gemini_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
```

Use your own credentials. Never commit `.env` or publish API keys.

### 3. Prepare the document collection

Place the required PDF manuals in `data/documents/`. Follow the existing ingestion code and configuration to extract, chunk, embed, and upload the documents to Qdrant.

The existing Qdrant collection is named `ragforge_documents`. Re-ingestion is not required when the collection already contains the intended indexed documents.

### 4. Run the application

From the repository root:

```powershell
streamlit run src/app/app.py
```

## Evaluation

The evaluation dataset is `evaluation/evaluation_questions.csv` and contains **40 questions**:

- 32 answerable questions
- 8 unanswerable questions

The updated-prompt run completed all 40 questions without recorded execution errors. A successful run does not mean every answer is correct.

The current preliminary analysis found that 8 of 32 answerable questions triggered abstention wording, while 7 of 8 unanswerable questions did so. These are phrase-based results, not a full semantic accuracy assessment.

See the following files for details:

- `evaluation/README.md` — evaluation workflow and metrics
- `evaluation/RAGForge_Evaluation_Report.md` — results and limitations
- `evaluation/evaluation_results_updated.csv` — updated-prompt run output

A controlled comparison of Top-k values 1, 3, and 5 has **not** been completed. Retrieval quality, answer correctness, and citation accuracy need further grading.

## Repository Structure

```text
src/
  app/          Streamlit interface
  ingestion/    PDF extraction, chunking, embeddings, indexing
  retrieval/    Retrieval and answer generation
evaluation/     Questions, evaluation scripts, results, reports
data/           Local PDF manuals (not committed)
```

## Team Workflow

Pull the latest changes before starting work:

```powershell
git pull origin main
```

After reviewing your changes, stage only the intended files, commit, and push:

```powershell
git add README.md
git commit -m "Update project documentation"
git push origin main
```

Adjust the `git add` command when committing other reviewed project files.