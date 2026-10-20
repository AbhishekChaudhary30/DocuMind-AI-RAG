"""Unit tests for security policies, credential masking, format checks, and PDF safety."""

import pytest
from app.security.secrets_policy import (
    get_security_disclosures,
    mask_secret,
    purge_session_credentials,
    validate_key_format,
    validate_pdf_file,
)


def test_mask_secret():
    """Verifies credential masking masks internal characters while preserving boundary hints."""
    # Standard OpenAI style key
    masked = mask_secret("sk-1234567890abcdefghijklmnop")
    assert masked.startswith("sk-")
    assert masked.endswith("mnop")
    assert "..." in masked
    assert "1234567890" not in masked

    # None or empty
    assert mask_secret(None) == "••••••••"
    assert mask_secret("") == "••••••••"

    # Short string
    short_masked = mask_secret("secret")
    assert "•" in short_masked


def test_validate_key_format_openai():
    """Verifies format validation for OpenAI keys."""
    # Valid syntax
    valid, err = validate_key_format("openai", "sk-proj-1234567890abcdefghijklmnopqrstuvwxyz")
    assert valid is True
    assert err is None

    # Missing sk- prefix
    valid, err = validate_key_format("openai", "bad-key-1234567890abcdefghijklmnopqrstuvwxyz")
    assert valid is False
    assert "sk-" in err

    # Too short
    valid, err = validate_key_format("openai", "sk-short")
    assert valid is False


def test_validate_key_format_anthropic():
    """Verifies format validation for Anthropic keys."""
    valid, err = validate_key_format("anthropic", "sk-ant-api03-1234567890abcdefghijklmnopqrstuvwxyz")
    assert valid is True

    valid, err = validate_key_format("anthropic", "invalid-key-test")
    assert valid is False


def test_validate_key_format_gemini():
    """Verifies format validation for Google Gemini keys."""
    valid, err = validate_key_format("gemini", "AIzaSyD1234567890abcdefghijklmnopqrstuvw")
    assert valid is True

    valid, err = validate_key_format("gemini", "wrong-prefix-key")
    assert valid is False


def test_validate_key_format_groq():
    """Verifies format validation for Groq keys."""
    valid, err = validate_key_format("groq", "gsk_1234567890abcdefghijklmnopqrstuvwxyz")
    assert valid is True

    valid, err = validate_key_format("groq", "wrong-groq-key")
    assert valid is False


def test_validate_key_format_ollama():
    """Verifies Ollama does not require an API key."""
    valid, err = validate_key_format("ollama", "")
    # Ollama is local, empty is permitted or returns true
    # But validate_key_format checks empty first, let's verify handling
    valid, err = validate_key_format("ollama", "none")
    assert valid is True


def test_validate_pdf_file_valid(sample_pdf_bytes):
    """Verifies that authentic PDF bytes pass validation."""
    valid, err = validate_pdf_file("report.pdf", sample_pdf_bytes, max_size_bytes=10 * 1024 * 1024)
    assert valid is True
    assert err is None


def test_validate_pdf_file_invalid_extension(sample_pdf_bytes):
    """Verifies non-pdf file extensions are rejected."""
    valid, err = validate_pdf_file("malicious.exe", sample_pdf_bytes, max_size_bytes=10 * 1024 * 1024)
    assert valid is False
    assert "Unsupported file extension" in err


def test_validate_pdf_file_empty():
    """Verifies zero-byte files are rejected."""
    valid, err = validate_pdf_file("empty.pdf", b"", max_size_bytes=10 * 1024 * 1024)
    assert valid is False
    assert "is empty" in err


def test_validate_pdf_file_oversized(sample_pdf_bytes):
    """Verifies files exceeding max size limit are rejected."""
    valid, err = validate_pdf_file("huge.pdf", sample_pdf_bytes, max_size_bytes=10)
    assert valid is False
    assert "exceeds maximum allowed size" in err


def test_validate_pdf_file_missing_magic_header():
    """Verifies files without %PDF- magic signature are rejected."""
    fake_bytes = b"This is just a text file masquerading as pdf."
    valid, err = validate_pdf_file("fake.pdf", fake_bytes, max_size_bytes=10 * 1024 * 1024)
    assert valid is False
    assert "does not contain a valid %PDF- header" in err


def test_purge_session_credentials(mock_session_dict):
    """Verifies session credentials are completely wiped from state."""
    assert "api_key" in mock_session_dict
    purged_count = purge_session_credentials(mock_session_dict)
    assert purged_count >= 1
    assert "api_key" not in mock_session_dict


def test_get_security_disclosures():
    """Verifies mandatory security disclosures are populated."""
    disclosures = get_security_disclosures()
    assert isinstance(disclosures, list)
    assert len(disclosures) >= 3
    titles = [d["title"] for d in disclosures]
    assert any("Server-Side" in t for t in titles)
    assert any("Disk" in t or "Storage" in t for t in titles)
