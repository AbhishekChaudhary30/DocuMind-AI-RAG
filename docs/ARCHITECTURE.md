# DocuMind AI — System Architecture Specification

## 1. Architectural Philosophy

DocuMind AI is engineered as a production-minded document intelligence and Retrieval-Augmented Generation (RAG) platform. The primary architectural objective is to deliver verifiable, hallucination-resistant answers with authentic page-level citations while strictly enforcing user credential privacy.

### Key Architectural Tenets
1. **Separation of Concerns**: UI rendering is strictly decoupled from business logic, data processing, security controls, and model adapters.
2. **Ephemeral Credential Security**: API credentials exist solely in server-side session memory and are scrubbed from logs, errors, and disk persistence.
3. **Truth and Evidence First**: Answers require factual grounding against extracted document passages with exact 1-based page citations.
4. **Provider Neutrality**: Extensible adapter interfaces accommodate commercial cloud models (OpenAI, Anthropic, Gemini, Groq), private open-weights execution (Ollama), and local offline extractive fallback.

---

## 2. Directory Structure and Module Boundaries

```
documind-ai/
├── streamlit_app.py           # Application entry point and router
├── requirements.txt           # Pinned production dependencies
├── pyproject.toml             # Build system and tool configuration
├── pytest.ini                 # Test suite runner settings
├── LICENSE                    # MIT License (Abhishek Chaudhary)
├── README.md                  # System overview and setup manual
│
├── .streamlit/
│   └── config.toml            # Custom dark theme and server settings
│
├── app/
│   ├── core/                  # System core
│   │   ├── config.py          # Centralized Pydantic settings
│   │   ├── constants.py       # Constants, provider registry, sample catalog
│   │   ├── exceptions.py      # Sanitized exception hierarchy
│   │   └── logging_config.py  # Credential scrubbing and logging setup
│   ├── document_processing/   # Document processing subsystem
│   │   ├── validators.py      # PDF format, magic byte, and integrity checks
│   │   ├── text_cleaner.py    # Unicode normalization, cleaning, token estimation
│   │   ├── pdf_loader.py      # PyMuPDF 1-based page text extraction
│   │   ├── chunker.py         # Semantic chunking with page span tracking
│   │   ├── metadata.py        # SHA-256 content hashing and metadata extraction
│   │   └── ocr.py             # OCR detection and pytesseract execution
│   ├── embeddings/            # Dense embedding subsystem
│   │   ├── base.py            # BaseEmbeddingProvider interface and L2 normalization
│   │   ├── local_embeddings.py# FastLocalEmbedder and SentenceTransformers
│   │   └── provider_embeddings.py # OpenAI, Gemini, and Ollama embeddings
│   ├── retrieval/             # Retrieval subsystem
│   │   ├── vector_store.py    # In-memory FAISS-CPU IndexFlatIP (cosine similarity)
│   │   ├── keyword_search.py  # BM25Okapi sparse lexical retrieval
│   │   ├── hybrid_search.py   # Reciprocal Rank Fusion (RRF) search
│   │   └── reranker.py        # Candidate deduplication and query term boosting
│   ├── llm/                   # Large Language Model integration
│   │   ├── base.py            # BaseLLMProvider interface
│   │   ├── model_registry.py  # Provider capabilities and model validation
│   │   ├── prompts.py         # Grounded prompts with XML context delimiters
│   │   └── providers.py       # OpenAI, Anthropic, Gemini, Groq, Ollama, Offline fallback
│   ├── rag/                   # End-to-end RAG pipeline
│   │   ├── context_builder.py # Context budget assembly and source registry
│   │   ├── citations.py       # Deterministic citation parser and validation
│   │   ├── answer_generator.py# Coordinated answer synthesis and citation binding
│   │   └── pipeline.py        # Unified RAGPipeline orchestrator
│   ├── analysis/              # Automatic document analysis
│   │   ├── document_summary.py# Hierarchical map-reduce executive summarization
│   │   ├── key_topics.py      # Thematic concepts and named entity extraction
│   │   └── document_insights.py # Facts, question generation, and corpus stats
│   ├── security/              # Security and privacy controls
│   │   ├── secrets_policy.py  # Key masking, format checks, session purging
│   │   └── document_safety.py # Prompt injection defense and XML sanitization
│   ├── evaluation/            # Quantitative RAG evaluation & benchmarking
│   │   ├── benchmark_dataset.py # Curated 8-scenario benchmark test cases
│   │   ├── metrics.py         # Hit Rate@K, MRR, Precision, Recall, Citation Precision
│   │   └── runner.py          # Benchmark execution engine & aggregate telemetry
│   ├── services/              # High-level services
│   │   ├── document_service.py# Ingestion, validation, and duplicate detection
│   │   ├── indexing_service.py# Embedding, FAISS indexing, and BM25 building
│   │   ├── query_service.py   # Query execution and history coordination
│   │   ├── cleanup_service.py # Session isolation and index reset
│   │   ├── export_service.py  # Markdown, JSON, and CSV report serialization
│   │   └── health_service.py  # Runtime dependency diagnostics
│   └── ui/                    # Presentation layer
│       ├── layout.py          # Session state init and page setup
│       ├── branding.py        # Brand header, footer, and logo rendering
│       ├── navigation.py      # Multi-page router configuration
│       ├── components.py      # Reusable UI widgets
│       └── styles.py          # Custom CSS design system tokens
│
├── pages/                     # Application views (Streamlit modern pages)
│   ├── overview.py            # Product mission and RAG concept overview
│   ├── workspace.py           # Document Q&A workbench and evidence layout
│   ├── documents.py           # Document catalog, index status, and sample loader
│   ├── analysis.py            # Automated corpus diagnostics and executive intelligence
│   ├── model_settings.py      # Provider configuration, testing, and ephemeral keys
│   ├── evaluation.py          # RAG operational diagnostics and verification audit
│   └── about.py               # System diagnostics, roadmap, and attribution
│
├── data/                      # Data assets
│   ├── samples/               # Curated sample PDFs (CC0 / Public Domain)
│   └── exports/               # Destination for exported reports
│
└── tests/                     # Automated test suite
    ├── conftest.py
    ├── test_*.py
    ├── unit/
    ├── integration/
    └── security/
```

---

## 3. End-to-End RAG Pipeline Data Flow

```
[ User PDF Upload / Sample Document ]
       │
       ▼
[ Validators: Extension, File Size, Magic Bytes (%PDF-), Encryption Check ]
       │
       ▼
[ PDF Loader: PyMuPDF Page-by-Page 1-based Text Extraction ]
       │
       ▼
[ Text Cleaner: Unicode Normalization (NFKC), Control Char Stripping ]
       │
       ▼
[ Semantic Chunker: Sentence-Boundary Splitting + Overlap Window + Page Range Map ]
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
[ Dense Embeddings (FAISS IndexFlatIP) ] [ Sparse BM25 Keyword Index ]
       │                                 │
       └────────────────┬────────────────┘
                        ▼
       [ Hybrid Retrieval via Reciprocal Rank Fusion (RRF) ]
                        │
                        ▼ Top-K Relevant Passages
       [ Candidate Re-ranking & Deduplication ]
                        │
                        ▼
       [ Context Builder: Token Budget + Deterministic [Source N] Binding ]
                        │
                        ▼
       [ LLM Inference Engine (OpenAI / Anthropic / Gemini / Groq / Ollama / Local Extractive) ]
                        │
                        ▼
       [ Citation Validator: Intercepts & Validates Source IDs against Context ]
                        │
                        ▼
       [ Verified Answer with Interactive Page-Numbered Evidence Passages ]
```

---

## 4. State Management and Session Isolation

Streamlit maintains state via `st.session_state`. All vector indices (`FaissVectorStore`) and keyword models (`BM25KeywordSearch`) are instantiated per-session, ensuring complete multi-tenant memory isolation.

| State Key | Type | Description |
|---|---|---|
| `uploaded_documents` | `List[DocumentMetadata]` | Ingested PDF metadata in current session |
| `processed_documents` | `Dict[str, ProcessedDocument]` | In-memory parsed pages, chunks, and hashes |
| `vector_store` | `FaissVectorStore` | Session-isolated FAISS cosine similarity index |
| `keyword_search` | `BM25KeywordSearch` | Session-isolated BM25Okapi lexical index |
| `active_document_id` | `Optional[str]` | Target document filter (`None` = all documents) |
| `retrieval_mode` | `str` | Active strategy (`hybrid`, `dense`, or `keyword`) |
| `retrieval_top_k` | `int` | Number of candidate chunks retrieved (default: 5) |
| `selected_provider` | `str` | Active provider key (`openai`, `anthropic`, `ollama`) |
| `model_config` | `Dict[str, Any]` | Model identifier, temperature, custom endpoint |
| `api_key` | `Optional[str]` | Ephemeral session API key (never written to disk) |
| `chat_history` | `List[Dict]` | User queries and grounded responses with citations |
| `active_analysis` | `Optional[AnalysisResult]` | Cached corpus intelligence and summary |
