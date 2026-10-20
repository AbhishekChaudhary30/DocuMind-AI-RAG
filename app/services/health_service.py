"""System health, dependency diagnostics, and sample document verification service."""

import hashlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
import pydantic

from app.core.config import get_settings
from app.core.constants import ROOT_DIR, SAMPLE_DOCUMENTS_CATALOG
from app.core.logging_config import logger
from app.models.schemas import DocumentMetadata, SampleDocument, SystemHealth


def check_dependency(module_name: str) -> bool:
    """Checks whether a Python module is importable in the current environment."""
    return importlib.util.find_spec(module_name) is not None


def check_writable_temp() -> bool:
    """Verifies that the runtime environment can create and delete temporary files."""
    try:
        with tempfile.NamedTemporaryFile(prefix="documind_health_", delete=True) as tf:
            tf.write(b"health_check")
            tf.flush()
        return True
    except Exception as exc:
        logger.warning(f"Temporary file write test failed: {exc}")
        return False


def get_system_health() -> SystemHealth:
    """Evaluates core system health, runtime dependencies, and storage sanity."""
    settings = get_settings()

    # Core checks
    py_version = sys.version.split()[0]
    st_version = getattr(st, "__version__", "unknown")
    pyd_version = getattr(pydantic, "__version__", "unknown")

    has_pymupdf = check_dependency("pymupdf") or check_dependency("fitz")
    has_pandas = check_dependency("pandas")
    has_numpy = check_dependency("numpy")
    has_pillow = check_dependency("PIL")
    is_temp_writable = check_writable_temp()
    logo_exists = settings.resolve_logo_path() is not None

    has_faiss = check_dependency("faiss")
    has_bm25 = check_dependency("rank_bm25")

    checks: Dict[str, bool] = {
        "python_supported": sys.version_info >= (3, 11),
        "streamlit_available": bool(st_version),
        "pydantic_v2": pyd_version.startswith("2."),
        "pymupdf_available": has_pymupdf,
        "faiss_available": has_faiss,
        "bm25_available": has_bm25,
        "pandas_available": has_pandas,
        "numpy_available": has_numpy,
        "pillow_available": has_pillow,
        "temp_storage_writable": is_temp_writable,
        "logo_resolved": logo_exists,
    }

    # Determine overall status
    critical_checks = [
        checks["python_supported"],
        checks["streamlit_available"],
        checks["pydantic_v2"],
        checks["temp_storage_writable"],
    ]

    if all(critical_checks) and has_pymupdf:
        status = "healthy"
    elif all(critical_checks):
        status = "degraded"  # PyMuPDF missing, fallback mode required
    else:
        status = "warning"

    return SystemHealth(
        status=status,
        python_version=py_version,
        streamlit_version=st_version,
        pymupdf_available=has_pymupdf,
        pydantic_version=pyd_version,
        checks=checks,
    )


def calculate_file_hash(content: bytes) -> str:
    """Calculates SHA-256 cryptographic digest of document bytes."""
    return hashlib.sha256(content).hexdigest()


def get_pdf_metadata(filename: str, content: bytes) -> DocumentMetadata:
    """Extracts verified metadata from raw PDF bytes.

    Attempts page count extraction via PyMuPDF; gracefully falls back if unavailable.
    """
    file_size = len(content)
    file_hash = calculate_file_hash(content)
    page_count: Optional[int] = None
    error_msg: Optional[str] = None
    status = "uploaded"

    try:
        # Try importing pymupdf
        import pymupdf

        doc = pymupdf.open(stream=content, filetype="pdf")
        page_count = doc.page_count
        doc.close()
    except ImportError:
        try:
            import fitz

            doc = fitz.open(stream=content, filetype="pdf")
            page_count = doc.page_count
            doc.close()
        except Exception:
            logger.info(f"PyMuPDF not available for page counting '{filename}' - metadata recorded.")
    except Exception as exc:
        logger.warning(f"Could not extract page count from '{filename}': {exc}")
        error_msg = f"Page count extraction deferred: {str(exc)}"

    return DocumentMetadata(
        doc_id=file_hash[:16],
        filename=filename,
        size_bytes=file_size,
        page_count=page_count,
        sha256_hash=file_hash,
        processing_status=status,
        error_message=error_msg,
        is_sample=False,
    )


def get_sample_documents() -> List[Tuple[SampleDocument, bool]]:
    """Returns catalog of sample documents along with disk presence status."""
    results = []
    for entry in SAMPLE_DOCUMENTS_CATALOG:
        sample_model = SampleDocument(**entry)
        doc_path = ROOT_DIR / sample_model.rel_path
        exists_on_disk = doc_path.is_file() and doc_path.stat().st_size > 0
        results.append((sample_model, exists_on_disk))
    return results
