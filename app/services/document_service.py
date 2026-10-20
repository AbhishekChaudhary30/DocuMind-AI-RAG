"""High-level document processing service, duplicate detection, and validation."""

from typing import Dict, List, Optional, Tuple

from app.core.config import Settings, get_settings
from app.core.exceptions import DocumentProcessingError
from app.core.logging_config import logger
from app.document_processing.chunker import chunk_document_pages
from app.document_processing.metadata import calculate_content_hash, generate_doc_id
from app.document_processing.pdf_loader import load_pdf_pages
from app.document_processing.validators import validate_pdf_content
from app.models.schemas import ProcessedDocument


def process_uploaded_pdf(
    filename: str,
    file_bytes: bytes,
    settings: Optional[Settings] = None,
    enable_ocr: bool = False,
    existing_documents: Optional[Dict[str, ProcessedDocument]] = None,
) -> Tuple[Optional[ProcessedDocument], Optional[str], List[str]]:
    """Validates, extracts, and chunks a PDF document into a ProcessedDocument object.

    Handles duplicate content detection to prevent redundant processing.
    Returns:
        (processed_doc_or_none, error_message_or_none, warnings_list)
    """
    cfg = settings or get_settings()
    warnings: List[str] = []

    # 1. Structural Validation
    is_valid, err, val_warnings = validate_pdf_content(
        filename=filename,
        file_bytes=file_bytes,
        max_size_bytes=cfg.max_file_size_bytes,
        max_page_count=cfg.max_page_count,
    )
    warnings.extend(val_warnings)

    if not is_valid:
        return None, err, warnings

    # 2. Duplicate Detection via Content Hash
    content_hash = calculate_content_hash(file_bytes)
    doc_id = generate_doc_id(content_hash)

    if existing_documents and doc_id in existing_documents:
        existing_doc = existing_documents[doc_id]
        warnings.append(
            f"Duplicate content detected: '{filename}' has identical SHA-256 hash to '{existing_doc.filename}'. "
            "Reusing existing processed chunks."
        )
        return existing_doc, None, warnings

    # 3. Text Extraction Page-by-Page
    try:
        pages, is_scanned, loader_warnings = load_pdf_pages(
            file_bytes=file_bytes,
            doc_id=doc_id,
            enable_ocr=enable_ocr or cfg.enable_ocr,
        )
        warnings.extend(loader_warnings)
    except Exception as exc:
        logger.error(f"Text extraction failed for '{filename}': {exc}")
        return None, f"Failed to extract text from '{filename}': {str(exc)}", warnings

    if not pages:
        return None, f"Could not extract any readable pages from '{filename}'.", warnings

    # 4. Semantic Token-Aware Chunking
    try:
        chunks = chunk_document_pages(
            pages=pages,
            filename=filename,
            doc_id=doc_id,
            chunk_size_tokens=cfg.chunk_size_tokens,
            chunk_overlap_tokens=cfg.chunk_overlap_tokens,
            extraction_method="pymupdf",
        )
    except Exception as exc:
        logger.error(f"Chunking failed for '{filename}': {exc}")
        return None, f"Failed to partition text chunks: {str(exc)}", warnings

    processed_doc = ProcessedDocument(
        doc_id=doc_id,
        filename=filename,
        content_hash=content_hash,
        size_bytes=len(file_bytes),
        page_count=len(pages),
        pages=pages,
        chunks=chunks,
        is_scanned=is_scanned,
        has_ocr=enable_ocr,
        warnings=warnings,
    )

    return processed_doc, None, warnings
