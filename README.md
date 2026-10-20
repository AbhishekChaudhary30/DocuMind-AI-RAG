# DocuMind AI

> **"Understand Your Documents. Find Answers with Evidence."**

DocuMind AI is an AI-powered document intelligence and Question Answering application built on a genuine Retrieval-Augmented Generation (RAG) architecture. It transforms complex PDF files into searchable knowledge, synthesizing factual answers backed by page-level source references and deterministic citation validation.

Created by **Abhishek Chaudhary**.

---

## Project Status: Complete (All 3 Phases Implemented & Verified)

| Phase | Milestone Focus | Status |
|---|---|---|
| **Phase 1** | Application architecture, Streamlit modern multi-page routing, Graphite/Slate/Violet design system, centralized Pydantic settings, credential masking & regex redaction, PDF metadata extraction (PyMuPDF), health service, curated sample documents, automated test suite. | **Completed & Verified** |
| **Phase 2** | Text chunking, token boundary overlap, multi-provider LLM adapters (OpenAI, Anthropic, Gemini, Groq, Ollama, Offline Fallback), dense embeddings (FAISS), sparse keyword retrieval (BM25), hybrid search (RRF), automated document analysis, and grounded answer synthesis with page citations. | **Completed & Verified** |
| **Phase 3** | Automated quantitative RAG benchmark suite (8 test scenarios), deterministic retrieval evaluation (Hit Rate@3, MRR, Precision, Recall, Citation Grounding Precision), structured report export (Markdown, JSON, CSV), private GitHub repo strategy, and Streamlit Community Cloud readiness. | **Completed & Verified** |

---

## Core Capabilities

- **Reliable PDF Ingestion & Validation**: Comprehensive inspection of file size, PDF magic byte header (`%PDF-`), page count, password encryption, and scanned text detection using PyMuPDF.
- **Cryptographic Duplicate Detection**: SHA-256 content hashing to avoid redundant parsing and embedding of identical files.
- **Semantic Page-Aware Chunking**: Tokenizer-aware sliding window chunking with configurable size and overlap that tracks 1-based start and end page spans.
- **Dense Vector Search (FAISS)**: In-memory `IndexFlatIP` cosine similarity search with exact vector normalization and document-level deletion.
- **Sparse Keyword Search (BM25)**: Fast lexical retrieval using `BM25Okapi` with tokenization and term frequency saturation.
- **Hybrid Retrieval (Reciprocal Rank Fusion)**: Combines dense semantic and sparse keyword rankings using RRF ($k=60$) without raw score mismatch.
- **Multi-Provider LLM Integration**: Working adapters for OpenAI, Anthropic Claude, Google Gemini, Groq, local Ollama, and an offline extractive engine.
- **100% Offline Privacy Mode**: Built-in `FastLocalEmbedder` (deterministic 384-dimensional sublinear projection) and `OfflineExtractiveProvider` for zero-network, zero-API-key execution.
- **Deterministic Citation Validation**: Every answer citation is validated against the active source registry, verifying document name, 1-based page numbers, and supporting text excerpts.
- **Automated Document Intelligence**: Generates executive summaries (hierarchical map-reduce for long texts), thematic topic badges, entity extraction, key facts, and exploratory questions.
- **Quantitative Benchmark & Telemetry**: Integrated 8-scenario benchmark suite measuring Hit Rate@3, MRR, Precision@K, Recall@K, Citation Precision, and Grounding Accuracy with live latency auditing.
- **Structured Report Exports**: Instant downloads of Q&A evidence reports, corpus intelligence findings, and benchmark audits in Markdown, JSON, and CSV formats.
- **Prompt Injection Defense**: Contextual XML delimiters (`<context>...</context>`) and regex-based adversarial instruction neutralization.

---

## System Architecture Data Flow

```
[ PDF Upload / Sample Selection ]
              │
              ▼
    [ Structural Validation ] ──► (Magic Byte %PDF-, Size Limits, Encryption Check)
              │
              ▼
   [ Text Extraction (PyMuPDF) ] ──► (Page-by-page text, 1-based numbering, OCR check)
              │
              ▼
 [ Text Normalization & Chunking ] ──► (Sliding window, token overlap, page span tracking)
              │
              ▼
[ Embeddings & Dual Indexing ] ──► 1. Dense Vector Store (FAISS IndexFlatIP)
                                  2. Sparse Keyword Index (BM25Okapi)
              │
              ▼
    [ User Question Input ]
              │
              ▼
     [ Hybrid Search (RRF) ] ──► (Dense search + BM25 search -> Reciprocal Rank Fusion)
              │
              ▼
[ Prompt Construction & Delimitation ] ──► (<context>[Source N] (doc - Page P)...</context>)
              │
              ▼
   [ LLM Answer Synthesis ] ──► (OpenAI / Anthropic / Gemini / Groq / Ollama / Offline)
              │
              ▼
 [ Citation & Grounding Audit ] ──► (Validate [Source N] against ground context chunks)
              │
              ▼
 [ Evidence Display with Citations ]
```

---

## Technology Stack

- **Application Framework**: Streamlit (>= 1.36.0, verified on 1.49.1)
- **Programming Language**: Python 3.11 - 3.13 (tested on Python 3.13.7)
- **PDF Extraction**: PyMuPDF (`pymupdf` 2.0.0)
- **Vector Search**: FAISS-CPU (`faiss-cpu` 1.15.1)
- **Keyword Search**: rank-bm25 (`rank-bm25` 0.2.2)
- **Embeddings**: Sentence-Transformers & FastLocalEmbedder
- **Data Validation & Settings**: Pydantic v2 & Pydantic-Settings
- **Testing**: pytest (72 automated tests, 100% passing)

---

## Directory Structure

```
documind-ai/
├── streamlit_app.py           # Application entry point and router
├── requirements.txt           # Pinned production dependencies
├── pyproject.toml             # Build system and tool configuration
├── pytest.ini                 # Test suite runner settings
├── LICENSE                    # MIT License (Abhishek Chaudhary)
├── README.md                  # System manual and documentation
│
├── .streamlit/
│   └── config.toml            # Dark theme tokens and server settings
│
├── app/
│   ├── core/                  # Configuration, constants, exceptions, redaction logging
│   ├── document_processing/   # PDF validation, cleaning, PyMuPDF parsing, chunking, OCR
│   ├── embeddings/            # FastLocalEmbedder, SentenceTransformers, provider embeddings
│   ├── retrieval/             # FAISS vector store, BM25 keyword search, hybrid RRF, reranker
│   ├── llm/                   # Multi-provider LLM adapters, model registry, defensive prompts
│   ├── rag/                   # Context builder, citation validator, answer generator, pipeline
│   ├── analysis/              # Document summary, key topics, entities, facts, questions
│   ├── security/              # Secrets policy, credential purge, prompt injection defense
│   ├── services/              # Ingestion, indexing, query, cleanup, and health services
│   ├── ui/                    # UI layout, branding, design system, navigation components
│   └── models/                # Pydantic domain schemas
│
├── pages/                     # Streamlit views
│   ├── overview.py            # Product mission and interactive workflow guide
│   ├── workspace.py           # End-to-end PDF Q&A workbench with citations drawer
│   ├── documents.py           # Document catalog, chunk metrics, and sample loader
│   ├── analysis.py            # Automated corpus intelligence and topic extraction
│   ├── model_settings.py      # Provider credentials, connection tests, and purge controls
│   ├── evaluation.py          # Real-time pipeline diagnostics, latency, and audit trail
│   └── about.py               # System architecture, capabilities, and attribution
│
├── assets/                    # Static branding
│   └── logo/                  # Developer and product logos
│
├── data/                      # Data assets
│   ├── samples/               # 3 curated sample PDFs (CC0 / Public Domain)
│   └── exports/               # Destination for exported reports
│
├── tests/                     # 72 automated unit, integration, and security tests
│   ├── unit/                  # Tests for document processing, chunker, retrieval, citations, analysis
│   ├── integration/           # End-to-end RAG pipeline tests
│   ├── security/              # Prompt injection, credential purge, and session isolation tests
│   └── conftest.py            # Pytest synthetic PDF fixtures and helpers
│
└── docs/                      # Technical documentation
    ├── ARCHITECTURE.md        # Architecture specification and data flow diagrams
    ├── SECURITY.md            # Threat model, credential handling, and safety controls
    └── DEVELOPMENT.md         # Local development, testing, and debugging guide
```

---

## Local Setup & Quickstart

### Prerequisites
- Python 3.11+ installed.
- PowerShell (Windows) or Terminal (macOS/Linux).

### Quickstart Commands

```powershell
# 1. Navigate to project root
cd "c:\Users\ABHISHEK CHAUDHARY\Desktop\project_dl\documind_ai"

# 2. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the application
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## Running the Automated Test Suite

Execute the full automated test suite (72 unit, integration, and security tests):

```powershell
pytest -v
```

---

## Curated Sample Documents

DocuMind AI includes 3 pre-configured sample PDFs in `data/samples/`:
1. **Technical Architecture Guide** (`technical_architecture_guide.pdf`): Tests microservices topology retrieval and code specifications.
2. **Business Financial Report** (`business_financial_report.pdf`): Tests multi-column financial tables and quarterly KPI synthesis.
3. **Enterprise AI Governance Policy** (`ai_governance_policy.pdf`): Tests numbered clause citation and compliance verification.

---

## License & Attribution

This project is open-source under the **MIT License**.

Created by **Abhishek Chaudhary** · Follow & Connect:
- [GitHub](https://github.com/AbhishekChaudhary-dev)
- [LinkedIn](https://www.linkedin.com/in/abhishek-chaudhary-ai)
