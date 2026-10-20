"""Secrets policy, credential masking, format validation, and PDF safety controls."""

from typing import Dict, List, Optional, Tuple
import re

from app.core.exceptions import SecurityError, ValidationError


# Magic bytes for valid PDF files (%PDF-)
PDF_MAGIC_BYTES = b"%PDF-"


def mask_secret(secret: Optional[str], unmasked_prefix: int = 3, unmasked_suffix: int = 4) -> str:
    """Masks a secret string, revealing only specified leading and trailing characters.

    Examples:
        sk-1234567890abcdef -> sk-...cdef
        short -> ••••••
    """
    if not secret:
        return "••••••••"

    secret_str = str(secret).strip()
    total_len = len(secret_str)

    if total_len <= (unmasked_prefix + unmasked_suffix + 2):
        return "•" * min(total_len, 8)

    prefix = secret_str[:unmasked_prefix]
    suffix = secret_str[-unmasked_suffix:]
    return f"{prefix}...{suffix}"


def validate_key_format(provider: str, api_key: str) -> Tuple[bool, Optional[str]]:
    """Validates the basic syntax of user-supplied API keys without making network calls.

    Returns:
        (is_valid, error_message_or_none)
    """
    if not api_key or not isinstance(api_key, str) or not api_key.strip():
        return False, "API key cannot be empty."

    key = api_key.strip()
    provider_key = provider.lower()

    if provider_key == "openai":
        if not key.startswith("sk-") or len(key) < 25:
            return False, "OpenAI API keys typically begin with 'sk-' and exceed 25 characters."

    elif provider_key == "anthropic":
        if not key.startswith("sk-ant-") or len(key) < 30:
            return False, "Anthropic API keys typically begin with 'sk-ant-' and exceed 30 characters."

    elif provider_key == "gemini":
        if not key.startswith("AIza") or len(key) < 30:
            return False, "Google Gemini API keys typically begin with 'AIza' and exceed 30 characters."

    elif provider_key == "groq":
        if not key.startswith("gsk_") or len(key) < 25:
            return False, "Groq API keys typically begin with 'gsk_' and exceed 25 characters."

    elif provider_key == "ollama":
        # Ollama local instance does not require an API key
        return True, None

    else:
        if len(key) < 12:
            return False, f"Key appears too short for provider '{provider}'."

    # Verify no accidental whitespace or control characters
    if any(c in key for c in ["\n", "\r", "\t", " "]):
        return False, "API key contains invalid whitespace or newline characters."

    return True, None


def validate_pdf_file(
    filename: str,
    file_bytes: bytes,
    max_size_bytes: int,
) -> Tuple[bool, Optional[str]]:
    """Validates PDF file format, magic header bytes, and size bounds for secure ingestion.

    Returns:
        (is_valid, error_message_or_none)
    """
    if not filename.lower().endswith(".pdf"):
        return False, f"Unsupported file extension for '{filename}'. Only .pdf files are accepted."

    if not file_bytes:
        return False, f"File '{filename}' is empty (0 bytes)."

    if len(file_bytes) > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        actual_mb = len(file_bytes) / (1024 * 1024)
        return False, f"File exceeds maximum allowed size ({actual_mb:.1f} MB > {max_mb:.1f} MB)."

    # Verify PDF magic signature
    if not file_bytes.startswith(PDF_MAGIC_BYTES):
        # Scan initial 1024 bytes in case of leading whitespace/BOM
        if PDF_MAGIC_BYTES not in file_bytes[:1024]:
            return False, f"Invalid PDF format: '{filename}' does not contain a valid %PDF- header."

    return True, None


def purge_session_credentials(session_dict: dict) -> int:
    """Removes all stored API keys, tokens, and active credentials from the given session dictionary.

    Returns:
        Number of purged keys.
    """
    credential_keys = [
        "api_key",
        "openai_api_key",
        "anthropic_api_key",
        "gemini_api_key",
        "groq_api_key",
        "active_credentials",
        "temp_token",
        "gemini_discovered_models",
        "discovered_models",
    ]
    purged = 0
    for key in list(session_dict.keys()):
        if key in credential_keys or "api_key" in key.lower() or "secret" in key.lower():
            del session_dict[key]
            purged += 1
    return purged


def get_security_disclosures() -> List[Dict[str, str]]:
    """Returns the mandatory application security and privacy disclosures."""
    return [
        {
            "title": "Server-Side Execution Notice",
            "body": (
                "Streamlit runs Python code on the host application server. Any API key entered "
                "into this application is transmitted to the host runtime and retained ephemerally "
                "in server-side session memory only while your browser tab remains connected."
            ),
        },
        {
            "title": "Third-Party Provider Communication",
            "body": (
                "When you execute queries or request embeddings, prompt payloads and API keys are sent "
                "directly to the selected provider (OpenAI, Anthropic, Google, Groq). Each provider's "
                "individual data-retention, compliance, and billing policies apply."
            ),
        },
        {
            "title": "Zero Persistent Disk Storage",
            "body": (
                "DocuMind AI never writes user API keys, passwords, or bearer tokens to disk, local files, "
                "databases, browser localStorage, or server logs. Sensitive credentials are scrubbed by "
                "automated log filtering."
            ),
        },
        {
            "title": "Document Isolation and Sandboxing",
            "body": (
                "Uploaded documents are held in memory or ephemeral session storage, never used for model "
                "training, and protected against embedded PDF scripts or unauthorized outbound network calls."
            ),
        },
    ]
