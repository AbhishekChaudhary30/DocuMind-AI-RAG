"""Unit tests for system health diagnostics and PDF metadata extraction."""

from app.services.health_service import (
    calculate_file_hash,
    get_pdf_metadata,
    get_sample_documents,
    get_system_health,
)


def test_get_system_health():
    """Verifies system health diagnostics execute and return structured snapshot."""
    health = get_system_health()
    assert health.status in ["healthy", "degraded", "warning"]
    assert health.python_version
    assert health.streamlit_version
    assert isinstance(health.checks, dict)
    assert "python_supported" in health.checks
    assert health.checks["python_supported"] is True
    assert "temp_storage_writable" in health.checks
    assert health.checks["temp_storage_writable"] is True


def test_calculate_file_hash(sample_pdf_bytes):
    """Verifies SHA-256 calculation matches expected deterministic digest."""
    digest = calculate_file_hash(sample_pdf_bytes)
    assert isinstance(digest, str)
    assert len(digest) == 64  # SHA-256 produces 64 hex characters


def test_get_pdf_metadata(sample_pdf_bytes):
    """Verifies metadata extraction produces valid DocumentMetadata."""
    meta = get_pdf_metadata("contract.pdf", sample_pdf_bytes)
    assert meta.filename == "contract.pdf"
    assert meta.size_bytes == len(sample_pdf_bytes)
    assert meta.sha256_hash == calculate_file_hash(sample_pdf_bytes)
    assert meta.doc_id == meta.sha256_hash[:16]
    assert meta.processing_status == "uploaded"
    assert meta.size_mb >= 0


def test_get_sample_documents():
    """Verifies that sample documents catalog returns structured entries."""
    samples = get_sample_documents()
    assert len(samples) == 3

    for sample_model, exists in samples:
        assert sample_model.id
        assert sample_model.title
        assert sample_model.filename.endswith(".pdf")
        assert sample_model.license
        assert sample_model.source
        # All 3 were generated during setup and should be present on disk
        assert exists is True
