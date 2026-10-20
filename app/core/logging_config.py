"""Logging configuration with mandatory sensitive-data redaction for DocuMind AI."""

import logging
import re
import sys
from typing import Optional

# Regular expressions identifying sensitive keys, authorization tokens, and credentials
SENSITIVE_PATTERNS = [
    # Anthropic API Keys (must precede OpenAI sk- pattern)
    (re.compile(r"sk-ant-[a-zA-Z0-9_\-]{20,}", re.IGNORECASE), "[REDACTED_ANTHROPIC_KEY]"),
    # OpenAI API Keys
    (re.compile(r"sk-[a-zA-Z0-9_\-]{20,}", re.IGNORECASE), "[REDACTED_OPENAI_KEY]"),
    # Google Gemini API Keys
    (re.compile(r"AIza[0-9A-Za-z\-_]{35}", re.IGNORECASE), "[REDACTED_GEMINI_KEY]"),
    # Groq API Keys
    (re.compile(r"gsk_[a-zA-Z0-9_\-]{20,}", re.IGNORECASE), "[REDACTED_GROQ_KEY]"),
    # Generic Bearer Tokens
    (re.compile(r"(Bearer\s+)[A-Za-z0-9_\-\.]{15,}", re.IGNORECASE), r"\1[REDACTED_TOKEN]"),
    # Generic key-value credentials: api_key=..., token=..., password=...
    (
        re.compile(
            r"((?:api[_-]?key|secret|token|password|auth_token)\s*[:=]\s*['\"]?)([a-zA-Z0-9_\-\.]{8,})(['\"]?)",
            re.IGNORECASE,
        ),
        r"\1[REDACTED_CREDENTIAL]\3",
    ),
]


def redact_sensitive_text(text: str) -> str:
    """Sanitizes text by stripping API keys, tokens, and credentials."""
    if not isinstance(text, str):
        return text
    sanitized = text
    for pattern, replacement in SENSITIVE_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


class RedactingFormatter(logging.Formatter):
    """Logging formatter that automatically scrubs sensitive credentials from output."""

    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        return redact_sensitive_text(original)


def setup_logging(
    log_level: str = "INFO",
    app_logger_name: str = "documind",
) -> logging.Logger:
    """Initializes and returns the DocuMind application logger with redaction."""
    logger = logging.getLogger(app_logger_name)
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times
    if not any(isinstance(h.formatter, RedactingFormatter) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = RedactingFormatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


# Default logger instance
logger = setup_logging()
