# DocuMind AI — Deployment Guide & Infrastructure Readiness

This document outlines the deployment strategy, private GitHub repository workflow, and Streamlit Community Cloud hosting configuration for **DocuMind AI**.

---

## 1. Deployment Philosophy & Security Posture

- **Private Repository Model**: To protect proprietary architecture, codebase structure, and portfolio assets, DocuMind AI is configured for a **private GitHub repository**.
- **Ephemeral User Secrets**: Real user API keys must **never** be committed to Git or stored in Streamlit Cloud secrets. Users enter their keys in the browser session, where they remain in ephemeral server RAM only for the connection lifetime.
- **Zero Paid Cloud Requirement**: The application deploys and operates 100% free-tier compatible. Local embeddings (`FastLocalEmbedder`) and the local extractive engine (`OfflineExtractiveProvider`) execute on standard CPU containers without requiring paid cloud API keys or GPU accelerators.

---

## 2. Platform Requirements

| Target Environment | Operating System | Python Runtime | Memory (RAM) | System Dependencies |
|---|---|---|---|---|
| **Local Development** | Windows 10/11, macOS, Linux | Python 3.11 - 3.13 | 4 GB+ | Optional: Tesseract OCR |
| **Streamlit Community Cloud** | Debian GNU/Linux 12 | Python 3.11 / 3.12 | 1 GB - 3 GB (Standard Tier) | Defined in `packages.txt` |

---

## 3. Step-by-Step Private GitHub Repository Workflow

Follow these exact steps from your local terminal. Do **not** commit credentials, temporary uploads, or virtual environments.

### Step 3.1: Verify Working Directory and Pre-Flight Checks
```powershell
# Navigate to workspace
cd "c:\Users\ABHISHEK CHAUDHARY\Desktop\project_dl\documind_ai"

# Run tests to guarantee a green baseline before committing
pytest -v
```

### Step 3.2: Inspect Repository Status & Verify Git Exclusions
```powershell
# Inspect untracked and modified files
git status
```
Verify that `.gitignore` correctly prevents staging of:
- `.env` and `.env.*`
- `.streamlit/secrets.toml`
- `.venv/` and `venv/`
- `data/uploads/*`
- `data/exports/*`
- `__pycache__/` and `*.pyc`

### Step 3.3: Stage and Commit Workspace Source
```powershell
git add .
git commit -m "feat(phase-3): complete RAG evaluation suite, benchmark telemetry, and export reporting"
```

### Step 3.4: Create Private Remote Repository on GitHub
1. Log into your GitHub account at `https://github.com/new`.
2. Enter Repository name: `documind-ai`.
3. Set Visibility to **Private** (recommended).
4. Leave "Add a README file", ".gitignore", and "License" unchecked (these exist locally).
5. Click **Create repository**.

### Step 3.5: Push Code to Private GitHub Repository
```powershell
# Rename default branch to main if needed
git branch -M main

# Add your private GitHub remote (replace with your personal GitHub URL)
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/documind-ai.git

# Push code to GitHub
git push -u origin main
```

---

## 4. Deploying to Streamlit Community Cloud

### Step 4.1: Connect Account
1. Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Ensure Streamlit Cloud has authorization to access your private repositories.

### Step 4.2: Configure New App
Click **Create app** (or **New app**) and specify:
- **Repository**: `<YOUR_GITHUB_USERNAME>/documind-ai`
- **Branch**: `main`
- **Main file path**: `streamlit_app.py`
- **App URL**: `documind-ai` (or your preferred subdomain)

### Step 4.3: Advanced Settings (Python Version)
In **Advanced settings**:
- **Python version**: Select `3.11` or `3.12`.
- **Secrets**: Do **not** enter API keys here if you want users to provide their own credentials. If providing a shared demonstrator key, format as:
  ```toml
  # Optional shared demo credentials (omit if using client-side inputs)
  # OPENAI_API_KEY = "sk-..."
  ```

### Step 4.4: Deploy & Monitor Build Logs
Click **Deploy!**. Streamlit Cloud will:
1. Provision a Debian Linux container.
2. Install system packages from `packages.txt` (`tesseract-ocr`, `tesseract-ocr-eng`).
3. Install Python dependencies from `requirements.txt` via `pip`.
4. Launch `streamlit run streamlit_app.py`.

---

## 5. Linux System Dependencies (`packages.txt`)

Streamlit Community Cloud reads `packages.txt` to install native OS libraries before pip runs:
```
tesseract-ocr
tesseract-ocr-eng
```
These packages enable optical character recognition for scanned PDFs without manual server configuration.

---

## 6. Operational Resource Constraints & Safety Safeguards

Streamlit Community Cloud enforces container constraints:
- **RAM Threshold**: Standard containers have approximately 1 to 2.7 GB of available RAM.
- **CPU Limits**: Shared multi-tenant vCPUs.

### Built-in Safeguards in DocuMind AI:
1. **Lightweight Embedding Engine**: `FastLocalEmbedder` generates 384-dimensional vectors instantly in NumPy with near-zero RAM overhead, preventing Out-Of-Memory (OOM) crashes during container startup.
2. **Batch Ingestion Limits**: File size limit defaults to 20 MB; page counts default to 100 pages per document.
3. **Lazy Model Loading**: Heavy neural models (e.g. `SentenceTransformers`) are cached lazily and never downloaded during application initialization.
4. **Controlled Context Budget**: Prompts are capped at 3,000 tokens to prevent context overflow and external API timeouts.

---

## 7. Troubleshooting Deployment Issues

| Symptom | Probable Cause | Remediation |
|---|---|---|
| `ModuleNotFoundError: No module named 'app'` | Incorrect working directory or root path | Ensure `streamlit_app.py` is the entry point at repo root; `PROJECT_ROOT` is resolved in `sys.path`. |
| Container OOM (Out Of Memory) during pip install | Heavy wheel compilation | Dependencies are pinned to pre-compiled binary wheels (`pymupdf`, `faiss-cpu`, `torch`). |
| Tesseract not found in PATH | Missing OS binary on Linux | Verify `packages.txt` exists in root with `tesseract-ocr`. |
| Timeout on large PDF upload | Network or Streamlit upload cap | Adjust `maxUploadSize` in `.streamlit/config.toml` (default is 25 MB). |
