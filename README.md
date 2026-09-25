# RAGForge — Document Q&A Assistant

## Project Overview

RAGForge is a Retrieval-Augmented Generation (RAG) based Document Q&A Assistant.

The project uses a collection of smartphone user manuals from Samsung, Motorola, and OnePlus. Users can ask questions about the manuals, and the system retrieves relevant document sections before generating an answer with source/page citations.

## Document Collection

The project uses **14 smartphone manuals**, which is within the assignment requirement of 10–20 PDFs.

> **Note:** The original manufacturer PDFs are not stored in this public repository. Download the manuals from the official manufacturer sources below and place them locally in `data/documents/`.

### Samsung

| # | Model | Official Manual |
|---|---|---|
| 1 | Galaxy A57 5G | [Official Samsung PDF](https://downloadcenter.samsung.com/content/UM/202603/20260326125812922/SM-A576B_A376_UG_EU_16_Eng_Rev.1.0_260126.pdf) |
| 2 | Galaxy S25 Ultra | [Official Samsung PDF](https://downloadcenter.samsung.com/content/UM/202509/20250929135046516/SM-S93X_S92X_UG_EU_16_Eng_Rev.1.0_250929.pdf) |
| 3 | Galaxy S26 Ultra | [Official Samsung PDF](https://downloadcenter.samsung.com/content/UM/202602/20260227095338477/SM-S94X_UG_EU_16_Eng_Rev.1.0_260220.pdf) |
| 4 | Galaxy Z Flip7 | [Official Samsung PDF](https://downloadcenter.samsung.com/content/UM/202507/20250711090617100/SM-F766B_F761B_UG_EU_16_Eng_Rev.1.0_250709.pdf) | 
| 5 | Galaxy Z Fold8 Ultra | [Official Samsung PDF](https://downloadcenter.samsung.com/content/UM/202607/20260723101548322/SM-F976B_F971B_F776B_UG_EU_17_Eng_Rev.1.0_260708.pdf) |

### Motorola

| # | Model | Official Manual |
|---|---|---|
| 1 | Motorola Edge 50 | [Official Motorola PDF](https://help.motorola.com/hc/7156/16/pdf/help-motorola-edge-50-16-global-en-gb.pdf) |
| 2 | Motorola Edge 60 | [Official Motorola PDF](https://help.motorola.com/hc/7162/16/pdf/help-motorola-edge-60-16-global-en-gb.pdf) |
| 3 | Motorola Edge 70 Pro+ | [Official Motorola PDF](https://help.motorola.com/hc/7175/16/pdf/help-motorola-edge-70-pro-plus-16-global-en-gb.pdf) |
| 4 | Motorola Signature | [Official Motorola PDF](https://help.motorola.com/hc/6200/16/pdf/help-motorola-signature-16-global-en-gb.pdf) |

### OnePlus

| # | Model | Official Manual |
|---|---|---|
| 1 | OnePlus 8 | [Official OnePlus Manual](https://service.oneplus.com/in/user-manual#/detail?equipmentModelId=9249212a-a0ce-11ef-9811-fa168f20b2a1) |
| 2 | OnePlus 9 | [Official OnePlus Manual](https://service.oneplus.com/in/user-manual#/detail?equipmentModelId=924926b0-a0ce-11ef-9811-fa168f20b2a1) |
| 3 | OnePlus 13 | [Official OnePlus Manual](https://service.oneplus.com/in/user-manual#/detail?equipmentModelId=1875066204395962370) |
| 4 | OnePlus 15 | [Official OnePlus Manual](https://service.oneplus.com/in/user-manual#/detail?equipmentModelId=1979231416971198465) |

> **Source note:** The links above are reproduced from the provided document containing the official manual URLs. The OnePlus 9 and OnePlus 13 entries were supplied with the same URL in that document; the project should verify the correct model-specific page before final ingestion.

## Local Document Setup

After downloading the required manuals, place the PDF files inside:

```text
data/
└── documents/
    ├── Galaxy_A57_5G.pdf
    ├── Galaxy_S25_Ultra.pdf
    ├── Galaxy_S26_Ultra.pdf
    ├── Galaxy_Z_Flip7.pdf
    ├── Galaxy_Z_Fold8_Ultra.pdf
    ├── Motorola_Edge_50.pdf
    ├── Motorola_Edge_60.pdf
    ├── Motorola_Edge_70_Pro_Plus.pdf
    ├── Motorola_Signature.pdf
    ├── OnePlus_8.pdf
    ├── OnePlus_9.pdf
    ├── OnePlus_13.pdf
    └── OnePlus_15.pdf
```

Keep the downloaded PDFs local and do not commit the original manufacturer documents to the public repository unless redistribution permission has been confirmed.

## RAG Pipeline

```text
PDF Documents
      ↓
Text Extraction
      ↓
Text Chunking
      ↓
Gemini Embeddings
      ↓
Qdrant Vector Database
      ↓
User Question
      ↓
Question Embedding
      ↓
Top-K Similarity Retrieval
      ↓
Gemini LLM
      ↓
Answer + Document/Page Citation
```

## Planned Technology Stack

- **LLM:** Google Gemini
- **Embeddings:** Gemini Embeddings
- **Vector Database:** Qdrant Cloud
- **Interface:** Streamlit
- **Document Processing:** PDF text extraction + chunking
- **Language:** Python

## Environment Variables

Create a local `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
```

Never commit API keys or other secrets to GitHub.

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Evaluation

The project will evaluate:

- Retrieval hit rate
- Answer quality
- Citation correctness
- Different chunk sizes
- Different Top-K values
- Embedding/retrieval configurations
- Failure cases and unanswerable questions

## Team Workflow

The team uses the `main` branch as the shared development branch.

Before starting work:

```bash
git pull origin main
```

After completing work:

```bash
git add .
git commit -m "Describe your change"
git push origin main
```
