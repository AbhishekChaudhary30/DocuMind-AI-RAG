# DocuMind AI — Security Architecture & Threat Model

## 1. Security Architecture Principles

DocuMind AI implements defensive application security engineering to protect user credentials, proprietary documents, and computational infrastructure across all phases of processing.

---

## 2. API Key Handling and Credential Privacy

### 2.1 Ephemeral Server Memory Model
- **Password-Masked Input**: All API credentials entered in the UI utilize masked inputs (`type="password"`).
- **Zero Persistent Disk Storage**: API keys are never written to disk, SQLite databases, JSON files, local cache files, or browser `localStorage`.
- **Session-Scoped Lifetime**: Keys reside strictly in ephemeral server-side memory for the duration of the active browser connection.
- **Explicit Credential Purge**: Users can terminate and purge all session credentials at any time via the "Purge Credentials" action in Model Settings.
- **Redacting Log Formatter**: Standard application loggers utilize `RedactingFormatter` to scrub recognized key patterns (`sk-...`, `sk-ant-...`, `AIza...`, `gsk_...`) before any message is emitted to stdout or stderr.

### 2.2 Server-Side Execution Disclosure
> [!IMPORTANT]
> Streamlit executes Python code on the host application server. When a user inputs an API key into DocuMind AI, that key is transmitted across the network connection to the server runtime. It does not remain exclusively client-side in the browser. 

- **Egress to Third-Party Providers**: When executing embeddings or queries with commercial cloud providers, credentials and prompt payloads are transmitted via HTTPS directly to the user-selected provider (OpenAI, Anthropic, Google, Groq).
- **Third-Party Data Policies**: Each provider's published data-retention, compliance, privacy, and training opt-out terms apply to transmitted payloads.
- **Offline Privacy Mode**: DocuMind AI includes a built-in `OfflineExtractiveProvider` and `FastLocalEmbedder` that execute 100% locally without network egress or external credentials. Users operating with strictly confidential documents can also use self-hosted **Ollama**.

---

## 3. PDF Ingestion Security Controls

To prevent denial-of-service (DoS), memory exhaustion, and arbitrary script execution through malicious documents, DocuMind AI enforces multiple layers of document validation:

1. **Extension Verification**: Only `.pdf` filenames are permitted.
2. **File Size Enforcement**: Strict maximum threshold (default: 20 MB). Files exceeding this bound are rejected prior to parsing.
3. **Magic Byte Signature Inspection**: Uploaded bytes are inspected for the `%PDF-` binary magic header within the first 1,024 bytes.
4. **No Executable Code Execution**: Embedded JavaScript, dynamic ActionScript, and automated URI launching mechanisms within PDF streams are ignored and never executed.
5. **No External Network Fetching**: URL hyperlinks found inside PDF pages are never followed or fetched automatically by the ingestion pipeline.
6. **Encrypted Document Rejection**: Password-protected or encrypted PDFs are safely rejected with actionable guidance.

---

## 4. Prompt Injection Defense & Content Isolation

1. **Boundary Isolation via XML Delimiters**: Extracted document passages are enclosed in strict contextual XML-style delimiters (`<context>...</context>`).
2. **Adversarial Pattern Inspection**: Untrusted document passages are scanned for known prompt-injection signatures (e.g., `ignore previous instructions`, `disregard system prompt`, `system override:`, `reveal api key`).
3. **Instruction Neutralization**: Detected injection attempts are neutralized (`[UNTRUSTED_DOC_INSTRUCTION_NEUTRALIZED: ...]`) before being presented to the language model.
4. **Untrusted Data Framing**: System instructions explicitly instruct the model to treat all document text as untrusted reference data, not system instructions.

---

## 5. Session Isolation and Storage Boundaries

1. **Session-Isolated Indexes**: Each user session maintains its own distinct `FaissVectorStore` and `BM25KeywordSearch` instances in session state. Chunks belonging to one session are never visible to or searchable by another session.
2. **Safe Serialization**: DocuMind AI does not use Python `pickle` for storing or loading vector stores or uploaded documents, preventing arbitrary code execution through untrusted deserialization.
3. **Document Removal**: When a document is deleted, its chunks are purged from both FAISS and BM25 indexes immediately, leaving zero orphaned vectors in the active searchable corpus.

---

## 6. Repository Hygiene

The repository enforces strict `.gitignore` rules preventing inadvertent commits of:
- `.env` and `.env.*` configuration files.
- `.streamlit/secrets.toml`.
- Uploaded user documents in `data/uploads/`.
- Exported reports in `data/exports/`.
- Vector index databases (`chroma_db/`, `faiss_index/`, `*.sqlite3`).
- Python virtual environment directories (`.venv/`, `venv/`).

---

## 7. Export Sanitization & Benchmark Isolation

1. **Zero Secret Leakage in Reports**: Exported Markdown, JSON, and CSV reports contain strictly user queries, synthesized findings, sanitized metadata, and citation excerpts. API credentials, authorization headers, and server-side environment variables are strictly excluded.
2. **Path & Filename Neutralization**: Filenames generated for export downloads are scrubbed of path traversal characters (`..`, `/`, `\`) and illegal filesystem symbols.
3. **Benchmark Corpus Isolation**: Automated benchmark evaluation queries run against session-managed indexes and do not persist query records or telemetry outside the active session.

---

## 8. Known Risks & Disclosures

- **Forensic Memory Erasure**: While Python references to keys and documents are cleared upon disconnect or deletion, garbage collection does not guarantee immediate overwrite of raw process RAM.
- **Adversarial Prompt Evasion**: No prompt-level guard can guarantee 100% immunity against novel jailbreak or prompt-injection attacks. Application-level sanitization provides a critical defense-in-depth layer.
