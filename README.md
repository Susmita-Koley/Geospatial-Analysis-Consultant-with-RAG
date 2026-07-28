# 🛰️ Project 1 — Geospatial Document Intelligence (RAG + RAGAS Evaluation)

> **Portfolio project showcasing: RAG pipeline · Vector search · LLM evaluation**

A production-grade Retrieval-Augmented Generation (RAG) system for querying official ESA, NASA, and USGS geospatial documentation in natural language — with source-page citations and automated quality evaluation using RAGAS.

---

## Why RAG for this use case?

| Requirement | Why RAG solves it |
|---|---|
| Grounding is mandatory | Hallucinated sensor specs / band values cause costly errors; RAG keeps answers tied to verified PDFs |
| Static corpus, no fine-tuning | Knowledge base rarely changes; vector retrieval is far more efficient than retraining |
| Source traceability | Analysts need to know *which document + page* answered a query; vanilla LLMs cannot do this |
| Measurable quality | RAGAS evaluation answers *"how do you know it works?"* — the standard production follow-up |

---

## Document Corpus

| Document | Agency | Size |
|---|---|---|
| Sentinel-2 User Handbook | ESA | ~1.7 MB |
| Sentinel-2 Product Specification v15.0 | ESA | ~8.9 MB |
| MODIS LST User Guide (MOD11, C6.1) | LP DAAC/USGS | ~1.1 MB |
| MODIS Land Cover User Guide (MCD12) | LP DAAC/USGS | ~0.4 MB |
| MODIS Vegetation Index User Guide (MOD13) | LP DAAC/USGS | ~2.5 MB |
| MODIS NPP/GPP User Guide (MOD17) | NASA/USGS | ~24 MB |
| Landsat Collection 2 Data Format Guide | USGS | ~0.6 MB |

All 7 PDFs are in the `docs/` folder.

---

## Architecture

```
PDFs (docs/)
  → PyMuPDF parser (page-by-page, preserves metadata)
  → RecursiveCharacterTextSplitter (512 tokens, 64 overlap)
  → SentenceTransformers (all-MiniLM-L6-v2, local CPU)
  → ChromaDB vector store (persistent, data/chroma_db/)

Query
  → ChromaDB top-k retrieval
  → Groq API (llama3-8b-8192) → streamed answer + source citations

RAGAS Evaluation (batch):
  testset.json (60 auto-generated QA pairs)
  → RAG pipeline → RAGAS metrics:
    faithfulness | answer_relevancy | context_precision | context_recall

Streamlit UI:
  Tab 1: Query interface (answer + source PDF page citations)
  Tab 2: RAGAS dashboard (radar chart + per-question scores)
  Tab 3: Setup / ingestion management
```

---

## Quick Start

### Prerequisites
- Python 3.12
- Groq API key (free at [console.groq.com](https://console.groq.com))
- **Windows note**: No Microsoft C++ Build Tools needed — we use pre-built wheels.

### 1. Clone & navigate
```bash
cd d:\Project\project1_rag
```

### 2. Activate virtual environment
```powershell
.\venv\Scripts\Activate.ps1    # Windows PowerShell
# source venv/bin/activate     # macOS / Linux
```

### 3. Install dependencies
```powershell
pip install -r requirements.txt
```

### 4. Download the document corpus (~36 MB, one-time)
```powershell
python scripts/download_corpus.py
```
This downloads all 7 official PDFs from ESA, NASA, and USGS into `docs/`.
> **Note**: If you already have the PDFs, just place them in the `docs/` folder — the script will skip existing files.

### 5. Configure environment
```powershell
copy .env.example .env
```
Edit `.env` and paste your Groq API key:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
GROQ_MODEL=llama-3.3-70b-versatile
```

### 4. Install dependencies (Windows-safe)

> **Windows note**: `--prefer-binary` is required for all packages to avoid
> `chroma-hnswlib` trying to compile from C++ source (which needs MSVC).

```powershell
.\venv\Scripts\pip.exe install -r requirements.txt --prefer-binary
```

### 5. Build the vector store (one-time, ~2–4 min)
```powershell
.\venv\Scripts\python.exe -m src.rag.ingest
```
This parses all 7 PDFs, embeds them locally, and persists to ChromaDB.

### 6. Launch the app
```powershell
.\venv\Scripts\streamlit.exe run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Evaluation Workflow

### Step 1 — Generate testset (run once)
```bash
python scripts/generate_testset.py
```
Uses RAGAS `TestsetGenerator` to auto-create **60 diverse QA pairs** from the corpus — no manual annotation needed.

### Step 2 — Run evaluation
```bash
python -m src.rag.evaluator
```
Scores the RAG pipeline on all 60 questions. Results saved to `data/eval_results.csv`.

### Step 3 — View dashboard
The 📊 Evaluation tab in the app shows:
- **Radar chart** of 4 RAGAS metrics
- **Per-question breakdown** with color-coded scores

---

## Project Structure

```
project1_rag/
├── app.py                       # Streamlit UI (3 tabs)
├── requirements.txt             # All pinned dependencies
├── .env.example                 # Environment template
├── .gitignore
├── docs/                        # 7 PDF documents (corpus)
├── data/
│   ├── chroma_db/               # ChromaDB vector store (generated)
│   ├── testset.json             # RAGAS QA pairs (generated)
│   └── eval_results.csv         # Evaluation scores (generated)
├── scripts/
│   └── generate_testset.py      # RAGAS testset generation
└── src/
    ├── config.py                # Centralised settings from .env
    └── rag/
        ├── ingest.py            # PDF parse → chunk → embed → ChromaDB
        ├── retriever.py         # Query → retrieve → Groq → cited answer
        └── evaluator.py         # RAGAS evaluation pipeline
```

---

## Tech Stack

| Component | Technology | Reason |
|---|---|---|
| PDF parsing | PyMuPDF (fitz) | Fast, no Poppler dependency, preserves page numbers |
| Chunking | LangChain RecursiveCharacterTextSplitter | Respects sentence boundaries, configurable |
| Embeddings | SentenceTransformers all-MiniLM-L6-v2 | Local CPU inference, 384-dim, excellent for English |
| Vector store | ChromaDB (persistent) | Lightweight, file-based, production-ready |
| LLM | Groq API llama-3.3-70b-versatile | Free tier, fast inference, no GPU needed |
| Evaluation | RAGAS 0.2.x | Industry standard RAG evaluation library |
| UI | Streamlit | Rapid iteration, no frontend boilerplate |

---

## Interview Talking Points

**Q: Why did you use RAG instead of just prompting the LLM?**
> The corpus is ~40 MB of technical PDF documentation with precise numerical values (band wavelengths, radiometric resolutions, calibration coefficients). LLMs hallucinate these kinds of facts. RAG grounds every answer in verified source text and provides the exact document + page number so the analyst can verify.

**Q: How do you know the RAG system works?**
> We use RAGAS to evaluate four orthogonal failure modes: faithfulness (is the answer grounded?), answer relevancy (does it address the question?), context precision (signal-to-noise in retrieval?), and context recall (is the right info being retrieved?). The QA pairs are auto-generated by RAGAS TestsetGenerator using simple, reasoning, and multi-context evolution strategies — so the benchmark is realistic and unbiased.

**Q: Why all-MiniLM-L6-v2 for embeddings?**
> It's a well-benchmarked general-purpose embedding model, runs entirely on CPU at ~50ms per chunk, and produces 384-dimensional vectors that give excellent retrieval quality for English technical text. On this hardware profile (i3, 8GB RAM, no GPU), it's the right trade-off between quality and latency.
