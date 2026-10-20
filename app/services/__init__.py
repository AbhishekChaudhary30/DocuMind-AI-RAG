"""System, document, indexing, and query services for DocuMind AI."""

from app.services.health_service import (
    get_system_health,
    get_sample_documents,
    calculate_file_hash,
    get_pdf_metadata,
)
from app.services.document_service import process_uploaded_pdf
from app.services.indexing_service import (
    index_processed_document,
    remove_document_from_indices,
    rebuild_indices_from_documents,
)
from app.services.query_service import execute_document_query
from app.services.cleanup_service import reset_user_workspace

__all__ = [
    "get_system_health",
    "get_sample_documents",
    "calculate_file_hash",
    "get_pdf_metadata",
    "process_uploaded_pdf",
    "index_processed_document",
    "remove_document_from_indices",
    "rebuild_indices_from_documents",
    "execute_document_query",
    "reset_user_workspace",
]
