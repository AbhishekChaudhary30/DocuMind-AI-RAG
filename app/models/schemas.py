"""Data schemas, domain entities, and type validation models for DocuMind AI."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class DocumentPage(BaseModel):
    """Represents a single extracted page from a PDF document."""

    doc_id: str = Field(default="", description="Parent document ID")
    page_number: int = Field(default=1, ge=1, description="1-based page index")
    raw_text: str = Field(default="", description="Unprocessed text extracted directly from PDF")
    cleaned_text: str = Field(default="", description="Normalized and sanitized text")
    char_count: int = Field(default=0, ge=0, description="Character count of cleaned text")
    estimated_tokens: int = Field(default=0, ge=0, description="Estimated token count (approx. 4 chars per token)")
    is_scanned: bool = Field(default=False, description="Flag indicating little or no native extractable text")


class DocumentChunk(BaseModel):
    """Represents an atomic semantic chunk produced for embedding and retrieval."""

    chunk_id: str = Field(description="Unique chunk identifier, e.g. docid_c001")
    doc_id: str = Field(description="Parent document ID")
    filename: str = Field(description="Original document filename")
    chunk_index: int = Field(ge=0, description="Sequential index within document")
    text: str = Field(description="Cleaned text content of the chunk")
    char_count: int = Field(default=0, ge=0, description="Length in characters")
    estimated_tokens: int = Field(default=0, ge=0, description="Estimated tokens count")
    start_page: int = Field(ge=1, description="First page number spanned by chunk (1-based)")
    end_page: int = Field(ge=1, description="Last page number spanned by chunk (1-based)")
    page_numbers: List[int] = Field(default_factory=list, description="All pages spanned by this chunk")
    extraction_method: str = Field(default="pymupdf", description="Extraction backend used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary chunk metadata")


class ProcessedDocument(BaseModel):
    """Encapsulates a fully parsed and chunked document ready for vector indexing."""

    doc_id: str = Field(description="Deterministic cryptographic document ID")
    filename: str = Field(description="Original uploaded filename")
    content_hash: str = Field(description="SHA-256 hash of original file bytes")
    size_bytes: int = Field(ge=0, description="File size in bytes")
    page_count: int = Field(ge=0, description="Total number of pages")
    pages: List[DocumentPage] = Field(default_factory=list, description="Extracted page objects")
    chunks: List[DocumentChunk] = Field(default_factory=list, description="Generated semantic chunks")
    is_scanned: bool = Field(default=False, description="True if document appears image-based/scanned")
    has_ocr: bool = Field(default=False, description="True if OCR was applied")
    warnings: List[str] = Field(default_factory=list, description="Processing warnings")
    processed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 timestamp",
    )

    @property
    def size_mb(self) -> float:
        return round(self.size_bytes / (1024 * 1024), 2)

    @property
    def total_tokens(self) -> int:
        return sum(c.estimated_tokens for c in self.chunks)


class DocumentMetadata(BaseModel):
    """Represents verified metadata for an ingested PDF document."""

    doc_id: str = Field(description="Unique document identifier (UUID or hash-based)")
    filename: str = Field(description="Original uploaded or sample filename")
    size_bytes: int = Field(ge=0, description="File size in bytes")
    page_count: Optional[int] = Field(default=None, ge=1, description="Extracted PDF page count")
    sha256_hash: str = Field(description="Cryptographic SHA-256 integrity hash")
    upload_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 upload timestamp",
    )
    processing_status: str = Field(
        default="uploaded",
        description="Current state: uploaded, parsed, indexed, error",
    )
    error_message: Optional[str] = Field(default=None, description="Detailed error description if failed")
    is_sample: bool = Field(default=False, description="True if loaded from curated samples")
    chunk_count: int = Field(default=0, ge=0, description="Number of indexed chunks")
    is_indexed: bool = Field(default=False, description="True if chunks exist in vector/keyword index")
    warnings: List[str] = Field(default_factory=list, description="Ingestion/extraction warnings")

    @property
    def size_mb(self) -> float:
        """Returns file size formatted in megabytes."""
        return round(self.size_bytes / (1024 * 1024), 2)


class SampleDocument(BaseModel):
    """Metadata schema for curated sample documents."""

    id: str
    title: str
    filename: str
    rel_path: str
    category: str
    source: str
    license: str
    intended_purpose: str
    description: str


class ProviderCapabilities(BaseModel):
    """Defines capabilities, limits, and authentication requirements for an LLM provider."""

    provider: str
    display_name: str
    description: str
    requires_api_key: bool
    key_env_var: str
    key_prefix_hint: str
    allows_custom_endpoint: bool
    default_endpoint: str
    default_model: str
    available_models: List[str]
    embedding_models: List[str]
    supports_streaming: bool = True


class ModelConfig(BaseModel):
    """Active configuration for model inference and embeddings."""

    provider: str = Field(default="openai", description="Active provider identifier")
    model_id: str = Field(default="gpt-4o-mini", description="Primary generation model")
    api_key: Optional[str] = Field(default=None, description="Ephemeral user-provided API key")
    custom_endpoint: Optional[str] = Field(default=None, description="Custom base URL for compatible APIs")
    temperature: float = Field(default=0.2, ge=0.0, le=1.0, description="Sampling temperature")
    max_tokens: int = Field(default=1024, ge=128, le=8192, description="Max generated tokens")
    embedding_model: Optional[str] = Field(default=None, description="Selected embedding model")
    is_connected: bool = Field(default=False, description="True if key format / health check passed")

    @field_validator("temperature")
    @classmethod
    def clamp_temperature(cls, v: float) -> float:
        return round(v, 2)


class Citation(BaseModel):
    """Page-level source reference backing an answer."""

    citation_id: str = Field(description="Unique identifier for citation record")
    chunk_id: str = Field(description="ID of referenced chunk")
    doc_id: str = Field(description="ID of source document")
    filename: str = Field(description="Source document filename")
    page_number: int = Field(ge=1, description="Primary 1-based page number")
    page_range: Optional[str] = Field(default=None, description="Formatted page range, e.g. 'p. 1-2'")
    snippet: str = Field(description="Verbatim text excerpt supporting claim")
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Retrieval similarity score")
    source_label: Optional[str] = Field(default=None, description="Deterministic label e.g. [Source 1]")
    verification_status: str = Field(
        default="verified",
        description="Verification state: verified, unverified, invalid_source",
    )
    is_valid: bool = Field(default=True, description="True if citation matched active context chunk")


class RetrievalResult(BaseModel):
    """Encapsulates a retrieved chunk candidate with similarity metrics."""

    chunk: DocumentChunk
    score: float = Field(description="Final ranking score (higher is more relevant)")
    rank: int = Field(default=1, ge=1, description="1-based candidate rank")
    retrieval_mode: str = Field(description="Retrieval engine: dense, keyword, or hybrid")
    dense_score: Optional[float] = Field(default=None, description="Raw cosine similarity score")
    keyword_score: Optional[float] = Field(default=None, description="Raw BM25 score")
    source_label: Optional[str] = Field(default=None, description="Context label e.g. [Source 1]")


class LLMResponse(BaseModel):
    """Encapsulates structured generation output from an LLM provider."""

    content: str = Field(description="Generated response text")
    citations: List[Citation] = Field(default_factory=list, description="Extracted and verified citations")
    prompt_tokens: Optional[int] = Field(default=None, description="Prompt tokens used")
    completion_tokens: Optional[int] = Field(default=None, description="Completion tokens generated")
    model_name: str = Field(description="Model identifier executed")
    provider: str = Field(description="Provider executed")
    latency_seconds: float = Field(ge=0.0, description="Wall clock roundtrip latency")
    is_evidence_sufficient: bool = Field(
        default=True,
        description="False if model indicated context lacked sufficient evidence",
    )


class AnalysisResult(BaseModel):
    """Structured document or corpus intelligence analysis output."""

    doc_id: Optional[str] = Field(default=None, description="Document ID or None if corpus-wide")
    document_titles: List[str] = Field(default_factory=list, description="Documents covered")
    summary: str = Field(description="Executive summary")
    key_topics: List[str] = Field(default_factory=list, description="Extracted key topics / domains")
    important_facts: List[str] = Field(default_factory=list, description="Critical factual statements")
    key_entities: Dict[str, List[str]] = Field(default_factory=dict, description="Categorized entities")
    suggested_questions: List[str] = Field(default_factory=list, description="Recommended exploratory questions")
    structural_stats: Dict[str, Any] = Field(default_factory=dict, description="Metrics: pages, chunks, words")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp",
    )


class EvaluationMetric(BaseModel):
    """Reliability and performance metric definition for the RAG pipeline."""

    name: str
    category: str
    target_threshold: float
    unit: str
    description: str
    status: str = Field(default="planned_phase_3")


class SystemHealth(BaseModel):
    """Snapshot of application health and runtime dependencies."""

    status: str = Field(description="'healthy', 'degraded', or 'warning'")
    python_version: str
    streamlit_version: str
    pymupdf_available: bool
    pydantic_version: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    checks: Dict[str, bool]
