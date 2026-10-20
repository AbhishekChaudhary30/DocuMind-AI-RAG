# DocuMind AI — Local Development Guide

This guide provides step-by-step instructions for running, testing, and developing DocuMind AI locally.

---

## 1. Prerequisites

- **Python**: Version 3.11, 3.12, or 3.13.
- **Git**: Version 2.30+.
- **Operating System**: Windows 10/11, macOS, or Linux.
- **Optional OCR Engine**: Tesseract OCR (if processing scanned or image-only PDFs).

---

## 2. Environment Setup

### 2.1 Clone Repository and Enter Workspace
```powershell
# Windows PowerShell
cd "c:\Users\ABHISHEK CHAUDHARY\Desktop\project_dl\documind_ai"
```

### 2.2 Create and Activate a Python Virtual Environment
```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux (bash/zsh)
python3 -m venv .venv
source .venv/bin/activate
```

### 2.3 Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Running the Application

Launch the Streamlit web application:

```powershell
streamlit run streamlit_app.py
```

Once running, access the application in your browser at:
`http://localhost:8501`

---

## 4. Running the Automated Test Suite

DocuMind AI uses `pytest` for unit, integration, and security testing:

```powershell
# Run all tests with standard verbosity
pytest

# Run full test suite with detailed output
pytest -v

# Run specific test suites
pytest tests/unit/
pytest tests/integration/
pytest tests/security/
```

---

## 5. Offline Operation vs. Cloud Model Configuration

- **Zero API Key Requirement**: By default, DocuMind AI operates completely offline using `FastLocalEmbedder` (deterministic 384-dimensional sublinear projection) and `OfflineExtractiveProvider` (verbatim evidence extractor).
- **Optional Cloud Providers**: Users can connect OpenAI, Anthropic, Google Gemini, Groq, or local Ollama on the **Model Settings** page by providing their respective API keys.

---

## 6. Coding Standards & Conventions

- **Type Hints**: Mandatory for all public functions, schemas, and service methods.
- **Docstrings**: Clear docstrings explaining inputs, outputs, and edge cases.
- **Sensitive Data**: Never log raw API keys, session tokens, or private secrets. Use `app.core.logging_config.logger` which includes automatic regex redaction filters.
- **Session Isolation**: Maintain all mutable index and corpus state in Streamlit `st.session_state` to prevent cross-session leakage.
