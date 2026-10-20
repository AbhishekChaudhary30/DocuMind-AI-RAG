"""Document summarization engine with hierarchical map-reduce for long texts."""

from typing import List, Optional
from app.llm.base import BaseLLMProvider
from app.llm.prompts import SUMMARY_SYSTEM_PROMPT
from app.models.schemas import DocumentPage


def generate_document_summary(
    pages: List[DocumentPage],
    llm_provider: BaseLLMProvider,
    max_chunk_chars: int = 3500,
) -> str:
    """Generates an executive summary using bounded hierarchical summarization."""
    if not pages:
        return "No document text available for summarization."

    # Filter pages with content
    valid_texts = [p.cleaned_text.strip() for p in pages if p.cleaned_text.strip()]
    if not valid_texts:
        return "Document contains no extractable text."

    full_text = "\n\n".join(valid_texts)

    # 1. Single pass if compact
    if len(full_text) <= max_chunk_chars:
        prompt = f"Please summarize the following document:\n\n{full_text}"
        resp = llm_provider.generate(
            prompt=prompt,
            system_prompt=SUMMARY_SYSTEM_PROMPT,
            temperature=0.2,
        )
        return resp.content

    # 2. Hierarchical map-reduce for long documents
    sections = []
    current_section = []
    current_chars = 0

    for text in valid_texts:
        if current_chars + len(text) > max_chunk_chars and current_section:
            sections.append("\n\n".join(current_section))
            current_section = [text]
            current_chars = len(text)
        else:
            current_section.append(text)
            current_chars += len(text)

    if current_section:
        sections.append("\n\n".join(current_section))

    intermediate_summaries = []
    for idx, sec in enumerate(sections[:5]):  # Bound to max 5 sections for resource safety
        p = f"Summarize section {idx + 1} of the document:\n\n{sec}"
        r = llm_provider.generate(prompt=p, system_prompt="Provide a concise 1-paragraph summary.", max_tokens=250)
        intermediate_summaries.append(r.content)

    combined_intermediate = "\n\n".join(intermediate_summaries)
    reduce_prompt = (
        "Combine the following section summaries into a cohesive executive overview:\n\n"
        f"{combined_intermediate}"
    )
    final_resp = llm_provider.generate(
        prompt=reduce_prompt,
        system_prompt=SUMMARY_SYSTEM_PROMPT,
        temperature=0.2,
    )
    return final_resp.content
