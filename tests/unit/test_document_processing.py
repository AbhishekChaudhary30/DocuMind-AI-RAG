"""Unit tests for PDF document ingestion, validation, and extraction."""

import pytest
from app.document_processing.validators import validate_pdf_content
from app.document_processing.metadata import calculate_content_hash, generate_doc_id
from app.document_processing.pdf_loader import load_pdf_pages
from app.services.document_service import process_uploaded_pdf
from app.document_processing.text_cleaner import clean_text, estimate_tokens
from app.document_processing.ocr import get_ocr_prerequisites, is_ocr_available


def test_calculate_content_hash_deterministic():
    data1 = b"%PDF-1.4 sample content 123"
    data2 = b"%PDF-1.4 sample content 123"
    data3 = b"%PDF-1.4 different content"
    assert calculate_content_hash(data1) == calculate_content_hash(data2)
    assert calculate_content_hash(data1) != calculate_content_hash(data3)


def test_validate_pdf_content_success(synthetic_multipage_pdf):
    is_valid, err, warnings = validate_pdf_content(
        filename="test_guide.pdf",
        file_bytes=synthetic_multipage_pdf,
        max_size_bytes=20 * 1024 * 1024,
        max_page_count=100,
    )
    assert is_valid is True
    assert err is None
    assert isinstance(warnings, list)


def test_validate_pdf_content_invalid_extension():
    is_valid, err, _ = validate_pdf_content(
        filename="doc.docx",
        file_bytes=b"%PDF-1.4",
        max_size_bytes=20 * 1024 * 1024,
        max_page_count=100,
    )
    assert is_valid is False
    assert "Unsupported file extension" in err


def test_validate_pdf_content_empty_bytes():
    is_valid, err, _ = validate_pdf_content(
        filename="empty.pdf",
        file_bytes=b"",
        max_size_bytes=20 * 1024 * 1024,
        max_page_count=100,
    )
    assert is_valid is False
    assert "empty" in err.lower()


def test_validate_pdf_content_missing_magic_byte():
    is_valid, err, _ = validate_pdf_content(
        filename="fake.pdf",
        file_bytes=b"NOT A REAL PDF CONTENT",
        max_size_bytes=20 * 1024 * 1024,
        max_page_count=100,
    )
    assert is_valid is False
    assert "%PDF-" in err


def test_validate_pdf_content_oversized():
    fake_large = b"%PDF-1.4" + b"A" * (2 * 1024 * 1024)
    is_valid, err, _ = validate_pdf_content(
        filename="huge.pdf",
        file_bytes=fake_large,
        max_size_bytes=1024 * 1024,
        max_page_count=100,
    )
    assert is_valid is False
    assert "exceeds size limit" in err


def test_load_pdf_pages_page_number_preservation(synthetic_multipage_pdf):
    pages, is_scanned, warnings = load_pdf_pages(synthetic_multipage_pdf, doc_id="doc-test-1")
    assert len(pages) == 3
    # Check 1-based page numbering
    assert [p.page_number for p in pages] == [1, 2, 3]
    assert "Architecture Guide" in pages[0].raw_text
    assert "Retrieval Subsystem" in pages[1].raw_text
    assert "Citation Verification" in pages[2].raw_text


def test_load_corrupted_pdf_returns_warning():
    corrupt_bytes = b"%PDF-1.4\nCorrupt header and random bytes that do not form a PDF"
    pages, _, warnings = load_pdf_pages(corrupt_bytes, doc_id="doc-corrupt")
    assert len(pages) == 0
    assert len(warnings) > 0


def test_process_uploaded_pdf_full_flow(synthetic_multipage_pdf):
    processed, err, warnings = process_uploaded_pdf(
        filename="complete_guide.pdf",
        file_bytes=synthetic_multipage_pdf,
    )
    assert err is None
    assert processed is not None
    assert processed.page_count == 3
    assert processed.filename == "complete_guide.pdf"
    assert len(processed.chunks) >= 1
    # Verify chunks have correct 1-based page tracking
    for chunk in processed.chunks:
        assert chunk.start_page in [1, 2, 3]
        assert chunk.end_page in [1, 2, 3]
        assert len(chunk.text) > 0


def test_text_cleaner_removes_null_bytes_and_normalizes():
    raw = "Hello\x00\x08 World!\t\r\n\r\nThis   is    DocuMind.  "
    cleaned = clean_text(raw)
    assert "\x00" not in cleaned
    assert "\x08" not in cleaned
    assert "Hello World!" in cleaned
    assert "This is DocuMind." in cleaned


def test_estimate_tokens():
    text = "The quick brown fox jumps over the lazy dog."
    tok = estimate_tokens(text)
    assert 5 <= tok <= 15


def test_ocr_availability_and_prerequisites():
    available, msg = is_ocr_available()
    assert isinstance(available, bool)
    assert isinstance(msg, str)
    prereqs = get_ocr_prerequisites()
    assert "Windows" in prereqs
    assert "Linux" in prereqs

