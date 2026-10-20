"""Semantic, page-aware text chunking with token-boundary overlap and page mapping."""

import re
from typing import List, Tuple
from app.document_processing.text_cleaner import clean_text, estimate_tokens
from app.models.schemas import DocumentChunk, DocumentPage


def split_into_sentences(text: str) -> List[str]:
    """Splits text into sentences while respecting common abbreviations."""
    if not text:
        return []
    # Regex splitting on periods, question marks, or exclamation marks followed by whitespace
    sentence_end = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
    parts = sentence_end.split(text.strip())
    # Sub-split on double newlines to preserve paragraph boundaries
    result = []
    for part in parts:
        paragraphs = part.split("\n\n")
        for para in paragraphs:
            para_clean = para.strip()
            if para_clean:
                result.append(para_clean)
    return result


def chunk_document_pages(
    pages: List[DocumentPage],
    filename: str,
    doc_id: str,
    chunk_size_tokens: int = 800,
    chunk_overlap_tokens: int = 120,
    extraction_method: str = "pymupdf",
) -> List[DocumentChunk]:
    """Partitions document pages into semantic chunks with overlap while preserving 1-based page spans.

    When a chunk spans across page boundaries, start_page, end_page, and the full page_numbers
    list are preserved so citations point strictly to authentic source pages.
    """
    if not pages:
        return []

    if chunk_overlap_tokens >= chunk_size_tokens:
        chunk_overlap_tokens = max(0, chunk_size_tokens // 4)

    # 1. Flatten all pages into sequential units annotated with their authentic 1-based page numbers
    units: List[Tuple[str, int, int]] = []  # (unit_text, page_number, token_count)
    for page in pages:
        page_text = page.cleaned_text.strip()
        if not page_text:
            continue

        sentences = split_into_sentences(page_text)
        for s in sentences:
            tokens = estimate_tokens(s)
            # If an individual unit exceeds chunk_size_tokens, break it by word chunks
            if tokens > chunk_size_tokens:
                words = s.split()
                step = max(10, int(chunk_size_tokens * 0.75))
                for w_idx in range(0, len(words), step):
                    sub_text = " ".join(words[w_idx : w_idx + step])
                    units.append((sub_text, page.page_number, estimate_tokens(sub_text)))
            else:
                units.append((s, page.page_number, tokens))

    if not units:
        return []

    # 2. Sliding window accumulation with overlap
    chunks: List[DocumentChunk] = []
    current_idx = 0
    chunk_counter = 0

    while current_idx < len(units):
        accumulated_units: List[str] = []
        accumulated_pages: List[int] = []
        accumulated_tokens = 0
        end_idx = current_idx

        while end_idx < len(units):
            unit_text, page_num, unit_tokens = units[end_idx]

            # If adding unit exceeds chunk size (and we already have content), break
            if accumulated_tokens + unit_tokens > chunk_size_tokens and accumulated_tokens > 0:
                break

            accumulated_units.append(unit_text)
            accumulated_pages.append(page_num)
            accumulated_tokens += unit_tokens
            end_idx += 1

        if not accumulated_units:
            # Advance at least one unit to prevent infinite loop
            unit_text, page_num, unit_tokens = units[current_idx]
            accumulated_units.append(unit_text)
            accumulated_pages.append(page_num)
            accumulated_tokens = unit_tokens
            end_idx = current_idx + 1

        # Formulate chunk text
        chunk_text = "\n\n".join(accumulated_units).strip()
        unique_pages = sorted(list(set(accumulated_pages)))
        start_page = unique_pages[0] if unique_pages else 1
        end_page = unique_pages[-1] if unique_pages else start_page

        chunk_id = f"{doc_id}_c{chunk_counter:03d}"
        chunk_model = DocumentChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            filename=filename,
            chunk_index=chunk_counter,
            text=chunk_text,
            char_count=len(chunk_text),
            estimated_tokens=accumulated_tokens,
            start_page=start_page,
            end_page=end_page,
            page_numbers=unique_pages,
            extraction_method=extraction_method,
            metadata={"total_spanned_pages": len(unique_pages)},
        )
        chunks.append(chunk_model)
        chunk_counter += 1

        # Calculate next start index based on overlap
        if end_idx >= len(units):
            break

        # Backtrack from end_idx by chunk_overlap_tokens
        backtrack_tokens = 0
        backtrack_idx = end_idx - 1
        while backtrack_idx > current_idx:
            backtrack_tokens += units[backtrack_idx][2]
            if backtrack_tokens >= chunk_overlap_tokens:
                break
            backtrack_idx -= 1

        next_idx = max(current_idx + 1, backtrack_idx)
        current_idx = next_idx

    return chunks
