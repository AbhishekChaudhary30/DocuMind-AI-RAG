"""Unit tests for the 8-stage document processing pipeline tracker."""

import pytest
from app.core.config import get_settings
from app.embeddings.local_embeddings import FastLocalEmbedder
from app.llm.providers import OfflineExtractiveProvider
from app.rag.pipeline import RAGPipeline
from app.retrieval.keyword_search import BM25KeywordSearch
from app.retrieval.vector_store import FaissVectorStore
from app.services.pipeline_tracker import (
    IngestionPipelineReport,
    PipelineStage,
    create_initial_pipeline_stages,
    execute_document_ingestion_pipeline,
)


def make_test_rag_pipeline():
    """Helper to build a standalone RAGPipeline for testing."""
    vector_store = FaissVectorStore(dimension=384)
    keyword_search = BM25KeywordSearch()
    embedder = FastLocalEmbedder(dimension=384)
    llm = OfflineExtractiveProvider()
    return RAGPipeline(
        vector_store=vector_store,
        keyword_search=keyword_search,
        embedding_provider=embedder,
        llm_provider=llm,
    )


def test_pipeline_stages_structure():
    """Verify all 8 pipeline stages are properly declared."""
    stages = create_initial_pipeline_stages()
    assert len(stages) == 8
    stage_names = [s.name for s in stages]
    assert "Document Upload & Validation" in stage_names
    assert "PDF Inspection & Text Extraction" in stage_names
    assert "Text Cleaning & Normalization" in stage_names
    assert "Semantic Chunking" in stage_names
    assert "Dense Embedding Generation" in stage_names
    assert "Vector & Keyword Indexing" in stage_names
    assert "Retrieval Readiness Verification" in stage_names
    assert "Document Ready" in stage_names


def test_pipeline_execution_valid_pdf(synthetic_multipage_pdf):
    """Test full 8-stage execution on a valid PDF document with real text."""
    settings = get_settings()
    pipeline = make_test_rag_pipeline()

    stage_events = []

    def on_progress(idx, total, msg):
        stage_events.append((idx, total, msg))

    report = execute_document_ingestion_pipeline(
        filename="sample_valid.pdf",
        file_bytes=synthetic_multipage_pdf,
        vector_store=pipeline.vector_store,
        keyword_search=pipeline.keyword_search,
        embedding_provider=pipeline.embedding_provider,
        settings=settings,
        progress_callback=on_progress,
    )

    # Verification of final state
    assert report.is_ready is True
    assert report.error is None
    assert report.processed_doc is not None
    assert report.processed_doc.page_count > 0
    assert len(report.processed_doc.chunks) > 0
    assert pipeline.vector_store.count == len(report.processed_doc.chunks)
    assert pipeline.keyword_search.count == len(report.processed_doc.chunks)

    # Verification of all 8 stages completed
    for stage in report.stages:
        assert stage.status == "completed", f"Stage '{stage.name}' failed with: {stage.error}"
        assert stage.duration_seconds >= 0.0

    # Verification of progress callback calls
    assert len(stage_events) >= 8


def test_pipeline_execution_corrupted_file():
    """Test pipeline rejection on corrupted bytes."""
    settings = get_settings()
    pipeline = make_test_rag_pipeline()

    corrupted_bytes = b"NOT_A_REAL_PDF_DATA_STREAM"

    report = execute_document_ingestion_pipeline(
        filename="corrupted.pdf",
        file_bytes=corrupted_bytes,
        vector_store=pipeline.vector_store,
        keyword_search=pipeline.keyword_search,
        embedding_provider=pipeline.embedding_provider,
        settings=settings,
    )

    assert report.is_ready is False
    assert report.error is not None
    # Stage 1 (Validation) must fail
    assert report.stages[0].status == "failed"
    assert pipeline.vector_store.count == 0


def test_pipeline_execution_empty_file():
    """Test pipeline rejection on empty 0-byte file."""
    settings = get_settings()
    pipeline = make_test_rag_pipeline()

    report = execute_document_ingestion_pipeline(
        filename="empty.pdf",
        file_bytes=b"",
        vector_store=pipeline.vector_store,
        keyword_search=pipeline.keyword_search,
        embedding_provider=pipeline.embedding_provider,
        settings=settings,
    )

    assert report.is_ready is False
    assert report.stages[0].status == "failed"
    assert "empty" in report.error.lower()
