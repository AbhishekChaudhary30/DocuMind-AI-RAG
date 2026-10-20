"""Custom exceptions for DocuMind AI with automatic sensitive data protection."""

from typing import Optional


class DocuMindError(Exception):
    """Base exception for all DocuMind AI errors.

    Ensures that messages are sanitized and provide consistent error codes.
    """

    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code or "DOCUMIND_ERROR"
        super().__init__(self.message)

    def to_dict(self) -> dict:
        return {
            "error_code": self.error_code,
            "message": self.message,
        }


class ConfigurationError(DocuMindError):
    """Raised when configuration values are missing, invalid, or conflicting."""

    def __init__(self, message: str):
        super().__init__(message, error_code="CONFIG_ERROR")


class SecurityError(DocuMindError):
    """Raised when a security policy, credential handling rule, or permission constraint is violated."""

    def __init__(self, message: str):
        super().__init__(message, error_code="SECURITY_POLICY_VIOLATION")


class DocumentProcessingError(DocuMindError):
    """Raised during document ingestion, validation, or extraction failures."""

    def __init__(self, message: str, filename: Optional[str] = None):
        self.filename = filename
        super().__init__(message, error_code="DOCUMENT_PROCESSING_ERROR")


class ModelProviderError(DocuMindError):
    """Raised when an LLM or embedding provider encounters an authentication or runtime failure."""

    def __init__(self, message: str, provider: Optional[str] = None):
        self.provider = provider
        super().__init__(message, error_code="MODEL_PROVIDER_ERROR")


class ValidationError(DocuMindError):
    """Raised when user inputs or schema payloads fail validation constraints."""

    def __init__(self, message: str, field: Optional[str] = None):
        self.field = field
        super().__init__(message, error_code="VALIDATION_ERROR")
