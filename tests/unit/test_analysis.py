"""Unit tests for automatic document analysis subsystem."""

import pytest
from app.models.schemas import DocumentChunk, ProcessedDocument, DocumentMetadata, DocumentPage
from app.analysis.document_summary import generate_document_summary
from app.analysis.key_topics import extract_key_topics, extract_key_entities
from app.analysis.document_insights import extract_important_facts, generate_suggested_questions, analyze_processed_documents
from app.llm.providers import OfflineExtractiveProvider


@pytest.fixture
def mock_processed_docs():
    meta = DocumentMetadata(
        doc_id="doc-corp-1",
        filename="ai_report.pdf",
        size_bytes=1024,
        sha256_hash="abcdef1234567890",
        page_count=2,
    )
    p1 = DocumentPage(
        page_number=1,
        raw_text="Executive Overview: Deep learning models have achieved 94% accuracy in document intelligence tasks.",
        cleaned_text="Executive Overview: Deep learning models have achieved 94% accuracy in document intelligence tasks.",
        token_count=15,
    )
    p2 = DocumentPage(
        page_number=2,
        raw_text="Compliance Architecture: Data privacy regulations mandate session-scoped storage and complete isolation.",
        cleaned_text="Compliance Architecture: Data privacy regulations mandate session-scoped storage and complete isolation.",
        token_count=14,
    )
    c1 = DocumentChunk(
        chunk_id="chk-1",
        doc_id="doc-corp-1",
        filename="ai_report.pdf",
        chunk_index=0,
        start_page=1,
        end_page=1,
        page_numbers=[1],
        text=p1.cleaned_text,
        token_count=15,
    )
    c2 = DocumentChunk(
        chunk_id="chk-2",
        doc_id="doc-corp-1",
        filename="ai_report.pdf",
        chunk_index=1,
        start_page=2,
        end_page=2,
        page_numbers=[2],
        text=p2.cleaned_text,
        token_count=14,
    )
    return [
        ProcessedDocument(
            doc_id="doc-corp-1",
            filename="ai_report.pdf",
            content_hash="abcdef1234567890",
            size_bytes=1024,
            page_count=2,
            pages=[p1, p2],
            chunks=[c1, c2],
        )
    ]


def test_generate_document_summary(mock_processed_docs):
    llm = OfflineExtractiveProvider()
    summary = generate_document_summary(mock_processed_docs[0].pages, llm)
    assert len(summary) > 10
    assert "accuracy" in summary.lower() or "deep learning" in summary.lower() or "document" in summary.lower()


def test_extract_key_topics():
    text = "Machine learning algorithms optimize vector embeddings for semantic document retrieval and artificial intelligence."
    topics = extract_key_topics(text, top_n=4)
    assert len(topics) >= 1
    for t in topics:
        assert isinstance(t, str) and len(t) > 0


def test_extract_key_entities():
    text = "The Google Systems team achieved FY2025 revenue of $142.5M utilizing PyMuPDF and FAISS vector databases."
    entities = extract_key_entities(text)
    assert len(entities) > 0
    assert "Monetary Values" in entities or "Dates & Periods" in entities or "Technical Terms" in entities


def test_analyze_processed_documents_full_suite(mock_processed_docs):
    llm = OfflineExtractiveProvider()
    analysis = analyze_processed_documents(mock_processed_docs, llm)
    
    assert analysis.summary is not None
    assert len(analysis.key_topics) >= 1
    assert len(analysis.important_facts) >= 1
    assert len(analysis.suggested_questions) >= 1
    assert analysis.structural_stats["Total Documents"] == 1
    assert analysis.structural_stats["Total Pages"] == 2
    assert analysis.structural_stats["Total Chunks"] == 2
