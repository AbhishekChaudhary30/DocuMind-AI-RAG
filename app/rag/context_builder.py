"""Context window assembly, source identifier binding, and token budget management."""

from typing import Dict, List, Tuple
from app.document_processing.text_cleaner import estimate_tokens
from app.models.schemas import DocumentChunk, RetrievalResult


def build_context_block(
    retrieved_results: List[RetrievalResult],
    max_tokens: int = 3000,
) -> Tuple[str, Dict[str, DocumentChunk], bool]:
    """Assembles prompt context from retrieved chunks with deterministic source labels.

    Returns:
        (context_block, source_registry, is_truncated)
    """
    if not retrieved_results:
        return "", {}, False

    context_segments: List[str] = []
    source_registry: Dict[str, DocumentChunk] = {}
    total_tokens = 0
    is_truncated = False

    for idx, item in enumerate(retrieved_results):
        chunk = item.chunk
        source_id = f"Source {idx + 1}"
        source_label = f"[{source_id}]"

        # Format page span descriptor
        if chunk.start_page == chunk.end_page:
            page_desc = f"Page {chunk.start_page}"
        else:
            page_desc = f"Pages {chunk.start_page}-{chunk.end_page}"

        segment_header = f"{source_label} ({chunk.filename} - {page_desc})"
        segment_body = chunk.text.strip()
        segment = f"{segment_header}\n{segment_body}"

        segment_tokens = estimate_tokens(segment)
        if total_tokens + segment_tokens > max_tokens and context_segments:
            is_truncated = True
            break

        context_segments.append(segment)
        source_registry[source_id.lower()] = chunk
        source_registry[str(idx + 1)] = chunk
        source_registry[source_label.lower()] = chunk
        total_tokens += segment_tokens

    context_block = "\n\n".join(context_segments)
    return context_block, source_registry, is_truncated
