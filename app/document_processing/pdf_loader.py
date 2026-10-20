"""PDF text extraction, multi-page parsing, and page model construction using PyMuPDF."""

from typing import List, Tuple
import pymupdf

from app.core.logging_config import logger
from app.document_processing.ocr import extract_ocr_from_image_bytes, is_ocr_available
from app.document_processing.text_cleaner import clean_text, detect_scanned_text, estimate_tokens
from app.models.schemas import DocumentPage


def load_pdf_pages(
    file_bytes: bytes,
    doc_id: str,
    enable_ocr: bool = False,
) -> Tuple[List[DocumentPage], bool, List[str]]:
    """Extracts text page-by-page from raw PDF bytes preserving 1-based page numbering.

    Returns:
        (pages, is_document_scanned, warnings)
    """
    pages: List[DocumentPage] = []
    warnings: List[str] = []
    scanned_page_count = 0

    try:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        logger.error(f"Failed to open PDF stream for doc {doc_id}: {exc}")
        return [], False, [f"Could not open document: {str(exc)}"]

    ocr_engine_available, _ = is_ocr_available()

    try:
        for idx in range(doc.page_count):
            page_num = idx + 1
            page = doc.load_page(idx)

            raw_text = page.get_text("text") or ""
            cleaned = clean_text(raw_text)
            is_scanned = detect_scanned_text(cleaned)

            if is_scanned:
                scanned_page_count += 1
                if enable_ocr and ocr_engine_available:
                    try:
                        pix = page.get_pixmap(dpi=150)
                        img_bytes = pix.tobytes("png")
                        ocr_text, ocr_success = extract_ocr_from_image_bytes(img_bytes)
                        if ocr_success and ocr_text:
                            raw_text = ocr_text
                            cleaned = clean_text(ocr_text)
                            warnings.append(f"Page {page_num} extracted using OCR fallback.")
                    except Exception as ocr_err:
                        logger.warning(f"OCR failed for page {page_num}: {ocr_err}")

            char_count = len(cleaned)
            est_tokens = estimate_tokens(cleaned)

            page_model = DocumentPage(
                doc_id=doc_id,
                page_number=page_num,
                raw_text=raw_text,
                cleaned_text=cleaned,
                char_count=char_count,
                estimated_tokens=est_tokens,
                is_scanned=is_scanned,
            )
            pages.append(page_model)

        is_doc_scanned = scanned_page_count > 0 and scanned_page_count >= (len(pages) // 2)
        if is_doc_scanned:
            warnings.append(
                f"Document appears predominantly scanned ({scanned_page_count}/{len(pages)} pages have low native text)."
            )

        return pages, is_doc_scanned, warnings

    finally:
        try:
            doc.close()
        except Exception:
            pass
