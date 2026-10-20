"""Automatic document and corpus intelligence analysis subsystem."""

from app.analysis.document_summary import generate_document_summary
from app.analysis.key_topics import extract_key_topics, extract_key_entities
from app.analysis.document_insights import (
    extract_important_facts,
    generate_suggested_questions,
    analyze_processed_documents,
)

__all__ = [
    "generate_document_summary",
    "extract_key_topics",
    "extract_key_entities",
    "extract_important_facts",
    "generate_suggested_questions",
    "analyze_processed_documents",
]
