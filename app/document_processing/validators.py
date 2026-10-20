"""Document validation, format verification, and safety constraints."""

from typing import List, Optional, Tuple
import pymupdf

from app.security.secrets_policy import PDF_MAGIC_BYTES


def validate_pdf_content(
    filename: str,
    file_bytes: bytes,
    max_size_bytes: int,
    max_page_count: int,
) -> Tuple[bool, Optional[str], List[str]]:
    """Comprehensive validation of PDF structure, integrity, size, and readability.

    Returns:
        (is_valid, error_message, list_of_warnings)
    """
    warnings: List[str] = []

    # 1. Filename and Extension Validation
    if not filename or not isinstance(filename, str):
        return False, "Invalid or missing filename.", warnings

    if not filename.lower().endswith(".pdf"):
        return False, f"Unsupported file extension for '{filename}'. Only .pdf files are accepted.", warnings

    # 2. File Size Validation
    if not file_bytes:
        return False, f"File '{filename}' is empty (0 bytes).", warnings

    actual_size = len(file_bytes)
    if actual_size > max_size_bytes:
        actual_mb = actual_size / (1024 * 1024)
        max_mb = max_size_bytes / (1024 * 1024)
        return False, f"File exceeds size limit ({actual_mb:.1f} MB > {max_mb:.1f} MB).", warnings

    # 3. Magic Byte Signature Validation
    if not file_bytes.startswith(PDF_MAGIC_BYTES):
        if PDF_MAGIC_BYTES not in file_bytes[:1024]:
            return False, f"Invalid format: '{filename}' lacks standard %PDF- header.", warnings
        warnings.append("Leading bytes detected before %PDF- header; structure was normalized.")

    # 4. PyMuPDF Structural Parse & Integrity Check
    try:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        return False, f"Corrupted PDF file '{filename}': could not parse document structure ({str(exc)}).", warnings

    try:
        # Check Encryption / Password Protection
        if doc.is_encrypted:
            doc.close()
            return False, f"Encrypted PDF '{filename}' is password-protected and cannot be parsed.", warnings

        # Page Count Bounds Check
        page_count = doc.page_count
        if page_count == 0:
            doc.close()
            return False, f"PDF '{filename}' contains zero pages.", warnings

        if page_count > max_page_count:
            doc.close()
            return False, f"Document exceeds page limit ({page_count} > {max_page_count} pages).", warnings

        # Check for Extractable Text Content (Scanned Detection)
        total_text_chars = 0
        blank_pages = 0
        for page_idx in range(min(page_count, 10)):  # Sample initial 10 pages for speed
            page = doc.load_page(page_idx)
            text = page.get_text() or ""
            total_text_chars += len(text.strip())
            if len(text.strip()) < 30:
                blank_pages += 1

        if total_text_chars < 20:
            warnings.append(
                f"Document '{filename}' appears to be image-based or scanned (under 20 text characters detected). "
                "Text extraction may require OCR."
            )
        elif blank_pages > 0 and blank_pages >= (min(page_count, 10) // 2):
            warnings.append(
                f"{blank_pages} pages in '{filename}' contain minimal or no native text."
            )

        doc.close()
        return True, None, warnings

    except Exception as exc:
        try:
            doc.close()
        except Exception:
            pass
        return False, f"Failed to inspect PDF '{filename}': {str(exc)}", warnings
