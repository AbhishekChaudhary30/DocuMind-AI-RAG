"""Document processing, text normalization, PDF extraction, and chunking pipeline."""

from app.document_processing.validators import validate_pdf_content
from app.document_processing.text_cleaner import (
    clean_text,
    estimate_tokens,
    detect_scanned_text,
    remove_repetitive_headers_footers,
)
from app.document_processing.ocr import is_ocr_available, extract_ocr_from_image_bytes, get_ocr_prerequisites
from app.document_processing.metadata import (
    calculate_content_hash,
    generate_doc_id,
    extract_pdf_document_properties,
    create_document_metadata,
)
from app.document_processing.pdf_loader import load_pdf_pages
from app.document_processing.chunker import chunk_document_pages, split_into_sentences

__all__ = [
    "validate_pdf_content",
    "clean_text",
    "estimate_tokens",
    "detect_scanned_text",
    "remove_repetitive_headers_footers",
    "is_ocr_available",
    "extract_ocr_from_image_bytes",
    "get_ocr_prerequisites",
    "calculate_content_hash",
    "generate_doc_id",
    "extract_pdf_document_properties",
    "create_document_metadata",
    "load_pdf_pages",
    "chunk_document_pages",
    "split_into_sentences",
]
