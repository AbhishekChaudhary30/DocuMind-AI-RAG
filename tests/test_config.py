"""Unit tests for centralized configuration and settings management."""

import pytest
from pathlib import Path
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.core.constants import APP_NAME, DEFAULT_MAX_FILE_SIZE_MB


def test_default_settings():
    """Verifies default settings are instantiated with correct system constants."""
    settings = Settings()
    assert settings.app_name == APP_NAME
    assert settings.max_file_size_mb == DEFAULT_MAX_FILE_SIZE_MB
    assert settings.max_file_size_bytes == DEFAULT_MAX_FILE_SIZE_MB * 1024 * 1024
    assert ".pdf" in settings.allowed_extensions
    assert settings.environment == "development"


def test_get_settings_caching():
    """Verifies that get_settings() returns a cached singleton instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_file_size_validation():
    """Verifies file size limits enforce bounds."""
    # Negative or zero size should fail
    with pytest.raises(ValidationError):
        Settings(max_file_size_mb=0)

    with pytest.raises(ValidationError):
        Settings(max_file_size_mb=-5)

    # Excessive size should fail
    with pytest.raises(ValidationError):
        Settings(max_file_size_mb=500)

    # Valid size
    s = Settings(max_file_size_mb=50)
    assert s.max_file_size_mb == 50
    assert s.max_file_size_bytes == 50 * 1024 * 1024


def test_page_count_validation():
    """Verifies page count bounds enforcement."""
    with pytest.raises(ValidationError):
        Settings(max_page_count=0)

    with pytest.raises(ValidationError):
        Settings(max_page_count=2000)

    s = Settings(max_page_count=250)
    assert s.max_page_count == 250


def test_resolve_logo_path():
    """Verifies logo path resolution correctly identifies available logo assets."""
    settings = Settings()
    resolved = settings.resolve_logo_path()
    # If a logo exists in workspace or assets/logo, resolved should be a Path object
    if resolved:
        assert isinstance(resolved, Path)
        assert resolved.is_file()
        assert resolved.suffix.lower() in [".png", ".svg", ".jpg", ".jpeg"]
