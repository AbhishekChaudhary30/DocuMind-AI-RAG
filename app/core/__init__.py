"""Core system modules for DocuMind AI including configuration, logging, and constants."""

from app.core.config import Settings, get_settings
from app.core.constants import (
    APP_NAME,
    APP_TAGLINE,
    APP_DESCRIPTION,
    DEVELOPER_CREDIT,
    SUPPORTED_EXTENSIONS,
)
from app.core.exceptions import (
    DocuMindError,
    ConfigurationError,
    SecurityError,
    DocumentProcessingError,
    ModelProviderError,
    ValidationError,
)

__all__ = [
    "Settings",
    "get_settings",
    "APP_NAME",
    "APP_TAGLINE",
    "APP_DESCRIPTION",
    "DEVELOPER_CREDIT",
    "SUPPORTED_EXTENSIONS",
    "DocuMindError",
    "ConfigurationError",
    "SecurityError",
    "DocumentProcessingError",
    "ModelProviderError",
    "ValidationError",
]
