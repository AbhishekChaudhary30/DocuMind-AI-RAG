"""Unit tests for Pydantic schema validation contracts."""

import pytest
from pydantic import ValidationError

from app.models.schemas import (
    Citation,
    DocumentMetadata,
    EvaluationMetric,
    ModelConfig,
    SystemHealth,
)


def test_document_metadata_schema():
    """Verifies DocumentMetadata schema validation and calculations."""
    meta = DocumentMetadata(
        doc_id="abcd1234efgh5678",
        filename="technical_paper.pdf",
        size_bytes=2 * 1024 * 1024,
        sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        page_count=12,
    )
    assert meta.size_mb == 2.0
    assert meta.processing_status == "uploaded"
    assert meta.is_sample is False

    # Negative size should fail
    with pytest.raises(ValidationError):
        DocumentMetadata(
            doc_id="123",
            filename="bad.pdf",
            size_bytes=-100,
            sha256_hash="hash",
        )


def test_model_config_schema():
    """Verifies ModelConfig defaults and temperature clamping."""
    cfg = ModelConfig(provider="openai", model_id="gpt-4o")
    assert cfg.temperature == 0.2
    assert cfg.max_tokens == 1024
    assert cfg.is_connected is False

    # Clamping
    cfg2 = ModelConfig(provider="gemini", model_id="gemini-1.5-pro", temperature=0.7321)
    assert cfg2.temperature == 0.73

    # Temperature out of bounds should fail
    with pytest.raises(ValidationError):
        ModelConfig(temperature=1.5)


def test_citation_schema():
    """Verifies Citation schema attributes and score constraints."""
    citation = Citation(
        citation_id="cit-001",
        chunk_id="chk-042",
        doc_id="doc-999",
        filename="annual_report.pdf",
        page_number=4,
        snippet="Operating revenue increased by 28% year-over-year.",
        relevance_score=0.94,
    )
    assert citation.page_number == 4
    assert citation.relevance_score == 0.94

    # Relevance score > 1.0 or < 0.0 should fail
    with pytest.raises(ValidationError):
        Citation(
            citation_id="cit-002",
            chunk_id="chk-042",
            doc_id="doc-999",
            filename="report.pdf",
            page_number=1,
            snippet="test",
            relevance_score=1.5,
        )


def test_evaluation_metric_schema():
    """Verifies EvaluationMetric schema."""
    metric = EvaluationMetric(
        name="Faithfulness",
        category="Attribution",
        target_threshold=0.95,
        unit="ratio",
        description="Verifiable claims supported strictly by context",
    )
    assert metric.status == "planned_phase_3"
    assert metric.target_threshold == 0.95
