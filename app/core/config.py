"""Centralized configuration module for DocuMind AI using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.constants import (
    APP_NAME,
    APP_VERSION,
    DEFAULT_MAX_FILE_SIZE_MB,
    DEFAULT_MAX_PAGE_COUNT,
    DEFAULT_SOCIAL_LINKS,
    LOGO_DIR,
    ROOT_DIR,
    SUPPORTED_EXTENSIONS,
)


class Settings(BaseSettings):
    """Application settings and runtime parameters.

    Reads from environment variables and `.env` files with secure defaults.
    Secrets are never stored or given default values in this class.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="DOCUMIND_",
        extra="ignore",
    )

    # Application Metadata
    app_name: str = Field(default=APP_NAME, description="Public application name")
    app_version: str = Field(default=APP_VERSION, description="Semver release string")
    environment: str = Field(default="development", description="Runtime environment (development, test, production)")
    debug: bool = Field(default=False, description="Enable verbose developer diagnostics")
    log_level: str = Field(default="INFO", description="Logging verbosity level")

    # Document Processing Limits
    max_file_size_mb: int = Field(default=DEFAULT_MAX_FILE_SIZE_MB, description="Maximum upload size in megabytes")
    max_page_count: int = Field(default=DEFAULT_MAX_PAGE_COUNT, description="Maximum PDF page limit per document")
    allowed_extensions: List[str] = Field(default_factory=lambda: list(SUPPORTED_EXTENSIONS))

    # Chunking & Tokenizer Controls
    chunk_size_tokens: int = Field(default=800, ge=100, le=4000, description="Target chunk size in tokens")
    chunk_overlap_tokens: int = Field(default=120, ge=0, le=1000, description="Overlap between consecutive chunks")

    # Retrieval Controls
    retrieval_top_k: int = Field(default=5, ge=1, le=20, description="Number of candidate chunks retrieved")
    default_retrieval_mode: str = Field(
        default="hybrid",
        description="Retrieval engine: 'dense', 'keyword', or 'hybrid'",
    )
    context_budget_tokens: int = Field(default=3000, ge=500, le=16000, description="Max prompt context budget")

    # Embedding Architecture
    local_embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Default local SentenceTransformers model identifier",
    )
    embedding_batch_size: int = Field(default=32, ge=1, le=256, description="Embedding encoding batch size")

    # Feature Toggles
    enable_sample_docs: bool = Field(default=True, description="Enable curated sample documents")
    enable_ocr: bool = Field(default=False, description="Enable OCR for scanned/image-based PDFs")
    enable_telemetry: bool = Field(default=False, description="Strictly opt-in telemetry")

    # UI Branding Configuration
    social_links: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_SOCIAL_LINKS))
    custom_logo_path: Optional[str] = Field(default=None, description="Explicit path to brand logo")

    @field_validator("max_file_size_mb")
    @classmethod
    def validate_file_size(cls, v: int) -> int:
        if v <= 0 or v > 200:
            raise ValueError("max_file_size_mb must be between 1 and 200 MB")
        return v

    @field_validator("max_page_count")
    @classmethod
    def validate_page_count(cls, v: int) -> int:
        if v <= 0 or v > 1000:
            raise ValueError("max_page_count must be between 1 and 1000 pages")
        return v

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "Settings":
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be strictly smaller than chunk_size_tokens")
        return self

    @property
    def max_file_size_bytes(self) -> int:
        """Returns the maximum allowed file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024

    def resolve_logo_path(self) -> Optional[Path]:
        """Resolves the logo file path by inspecting assets and workspace root."""
        # 1. Custom path override if configured and exists
        if self.custom_logo_path:
            custom_p = Path(self.custom_logo_path)
            if custom_p.is_file():
                return custom_p

        # 2. Check standard assets/logo/ directory
        candidates = [
            LOGO_DIR / "logo.png",
            LOGO_DIR / "logo.svg",
            LOGO_DIR / "logo.jpg",
            ROOT_DIR / "logo.png",
        ]
        for candidate in candidates:
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate

        return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns the cached singleton Settings instance."""
    return Settings()
