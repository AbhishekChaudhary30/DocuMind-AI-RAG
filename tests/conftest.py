"""Pytest test fixtures and configuration for DocuMind AI."""

import os
import sys
from pathlib import Path
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Returns a minimal valid PDF byte sequence for testing."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000102 00000 n\n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF\n"
    )


@pytest.fixture
def mock_session_dict() -> dict:
    """Returns a fresh dictionary mimicking Streamlit session state."""
    return {
        "uploaded_documents": [],
        "active_document_id": None,
        "selected_provider": "openai",
        "api_key": "sk-dummy-mock-fixture-testing-only-1234",
        "model_config": {
            "provider": "openai",
            "model_id": "gpt-4o-mini",
            "is_connected": True,
        },
    }


@pytest.fixture
def create_test_pdf():
    """Factory fixture to generate valid synthetic PDFs with given text per page."""
    import pymupdf

    def _create(pages_text: list[str]) -> bytes:
        doc = pymupdf.open()
        for text in pages_text:
            page = doc.new_page()
            page.insert_text((72, 72), text)
        pdf_bytes = doc.write()
        doc.close()
        return pdf_bytes

    return _create


@pytest.fixture
def synthetic_multipage_pdf(create_test_pdf) -> bytes:
    """A 3-page valid synthetic PDF with realistic content for end-to-end testing."""
    return create_test_pdf([
        "DocuMind AI Architecture Guide. Page 1 outlines vector indexing with FAISS and dense semantic embeddings.",
        "DocuMind AI Retrieval Subsystem. Page 2 details BM25 sparse keyword search and Reciprocal Rank Fusion hybrid scoring.",
        "DocuMind AI Citation Verification. Page 3 describes deterministic provenance mapping and hallucination prevention.",
    ])

