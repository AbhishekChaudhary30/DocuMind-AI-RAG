"""Unit tests for logging configuration and sensitive-data redaction."""

import logging
from app.core.logging_config import RedactingFormatter, redact_sensitive_text, setup_logging


def test_redact_sensitive_text_openai():
    """Verifies that raw OpenAI keys are redacted."""
    raw = "Failed to connect using key sk-proj1234567890abcdefghijklmnopqrstuvwxyz"
    sanitized = redact_sensitive_text(raw)
    assert "sk-proj1234567890abcdefghijklmnopqrstuvwxyz" not in sanitized
    assert "[REDACTED_OPENAI_KEY]" in sanitized


def test_redact_sensitive_text_anthropic():
    """Verifies that raw Anthropic keys are redacted."""
    raw = "Request failed for token sk-ant-api03-1234567890abcdefghijklmnopqrstuvwxyz"
    sanitized = redact_sensitive_text(raw)
    assert "sk-ant-api03" not in sanitized
    assert "[REDACTED_ANTHROPIC_KEY]" in sanitized


def test_redact_sensitive_text_gemini():
    """Verifies that raw Gemini keys are redacted."""
    raw = "Calling Gemini API endpoint with AIzaSyD1234567890abcdefghijklmnopqrstuvw"
    sanitized = redact_sensitive_text(raw)
    assert "AIzaSyD" not in sanitized
    assert "[REDACTED_GEMINI_KEY]" in sanitized


def test_redact_sensitive_text_bearer():
    """Verifies that Bearer tokens are redacted."""
    raw = "Authorization: Bearer mySecretToken1234567890."
    sanitized = redact_sensitive_text(raw)
    assert "mySecretToken" not in sanitized
    assert "[REDACTED_TOKEN]" in sanitized


def test_redact_sensitive_text_clean_content():
    """Verifies that normal non-sensitive messages remain completely unaltered."""
    normal_msg = "Successfully ingested document annual_report.pdf with 14 pages."
    assert redact_sensitive_text(normal_msg) == normal_msg


def test_redacting_formatter():
    """Verifies that RedactingFormatter scrubs log records."""
    formatter = RedactingFormatter(fmt="%(message)s")
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Debug key: sk-abcdefghijklmnopqrstuvwxyz1234567890",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    assert "sk-abcdefghijklmnopqrstuvwxyz1234567890" not in formatted
    assert "[REDACTED_OPENAI_KEY]" in formatted
