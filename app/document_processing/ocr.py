"""Optical Character Recognition (OCR) detection and execution module."""

import io
import shutil
from typing import Dict, Optional, Tuple
from PIL import Image

from app.core.logging_config import logger


def is_ocr_available() -> Tuple[bool, str]:
    """Verifies whether pytesseract and the Tesseract OCR engine are installed and accessible."""
    try:
        import pytesseract  # type: ignore
    except ImportError:
        return False, "pytesseract Python library is not installed."

    tesseract_bin = shutil.which("tesseract")
    if not tesseract_bin:
        # Check standard Windows installation path
        std_win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        import os
        if os.path.isfile(std_win_path):
            pytesseract.pytesseract.tesseract_cmd = std_win_path
            return True, f"Tesseract found at: {std_win_path}"
        return False, "Tesseract OCR binary ('tesseract') not found in system PATH."

    return True, f"Tesseract executable verified: {tesseract_bin}"


def extract_ocr_from_image_bytes(image_bytes: bytes) -> Tuple[str, bool]:
    """Extracts text from image bytes using pytesseract under safe resource bounds."""
    available, msg = is_ocr_available()
    if not available:
        logger.warning(f"OCR requested but engine unavailable: {msg}")
        return "", False

    try:
        import pytesseract

        img = Image.open(io.BytesIO(image_bytes))
        # Limit image dimensions if excessively large
        if max(img.size) > 4000:
            img.thumbnail((4000, 4000), Image.Resampling.LANCZOS)

        text = pytesseract.image_to_string(img)
        return text.strip(), True
    except Exception as exc:
        logger.error(f"OCR extraction failed on image: {exc}")
        return "", False


def get_ocr_prerequisites() -> Dict[str, str]:
    """Returns platform-specific instructions for enabling Tesseract OCR."""
    return {
        "Windows": (
            "Download and install Tesseract from UB-Mannheim: "
            "https://github.com/UB-Mannheim/tesseract/wiki. "
            "Ensure 'C:\\Program Files\\Tesseract-OCR' is added to system PATH."
        ),
        "Linux": "Run: sudo apt-get update && sudo apt-get install -y tesseract-ocr",
        "macOS": "Run: brew install tesseract",
    }
