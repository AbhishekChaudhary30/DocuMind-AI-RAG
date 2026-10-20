"""Document metadata extraction, cryptographic content hashing, and ID generation."""

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pymupdf

from app.models.schemas import DocumentMetadata


def calculate_content_hash(file_bytes: bytes) -> str:
    """Calculates SHA-256 cryptographic digest of raw file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def generate_doc_id(content_hash: str) -> str:
    """Generates a stable 16-character document identifier derived from SHA-256."""
    return content_hash[:16]


def extract_pdf_document_properties(doc: pymupdf.Document) -> Dict[str, Any]:
    """Extracts standard PDF metadata fields from document catalog."""
    meta = doc.metadata or {}
    return {
        "title": meta.get("title") or None,
        "author": meta.get("author") or None,
        "subject": meta.get("subject") or None,
        "keywords": meta.get("keywords") or None,
        "creator": meta.get("creator") or None,
        "producer": meta.get("producer") or None,
        "creation_date": meta.get("creationDate") or None,
        "format": doc.name or "PDF",
    }


def create_document_metadata(
    filename: str,
    file_bytes: bytes,
    page_count: int,
    chunk_count: int = 0,
    is_indexed: bool = False,
    is_sample: bool = False,
    warnings: Optional[List[str]] = None,
) -> DocumentMetadata:
    """Creates a validated DocumentMetadata model from file characteristics."""
    content_hash = calculate_content_hash(file_bytes)
    doc_id = generate_doc_id(content_hash)

    return DocumentMetadata(
        doc_id=doc_id,
        filename=filename,
        size_bytes=len(file_bytes),
        page_count=page_count,
        sha256_hash=content_hash,
        upload_timestamp=datetime.now(timezone.utc).isoformat(),
        processing_status="indexed" if is_indexed else "parsed",
        is_sample=is_sample,
        chunk_count=chunk_count,
        is_indexed=is_indexed,
        warnings=warnings or [],
    )
