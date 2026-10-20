"""Evidence-grounded answer generation coordinating LLM inference and citations."""

from typing import List, Optional, Tuple
from app.llm.base import BaseLLMProvider
from app.llm.prompts import RAG_SYSTEM_PROMPT, format_rag_prompt
from app.models.schemas import Citation, LLMResponse, RetrievalResult
from app.rag.citations import parse_and_validate_citations
from app.rag.context_builder import build_context_block


def generate_grounded_answer(
    query: str,
    retrieved_results: List[RetrievalResult],
    llm_provider: BaseLLMProvider,
    conversation_history: str = "",
    context_budget_tokens: int = 3000,
    temperature: float = 0.2,
) -> Tuple[LLMResponse, List[Citation]]:
    """Synthesizes an evidence-grounded answer and returns validated citations."""
    if not retrieved_results:
        response = LLMResponse(
            content="No relevant passages were found in the uploaded documents to answer this question.",
            citations=[],
            model_name=llm_provider.model_name,
            provider=llm_provider.provider_name,
            latency_seconds=0.0,
            is_evidence_sufficient=False,
        )
        return response, []

    context_block, source_registry, is_truncated = build_context_block(
        retrieved_results,
        max_tokens=context_budget_tokens,
    )

    prompt = format_rag_prompt(
        query=query,
        context_block=context_block,
        conversation_history=conversation_history,
    )

    llm_resp = llm_provider.generate(
        prompt=prompt,
        system_prompt=RAG_SYSTEM_PROMPT,
        temperature=temperature,
    )

    # Validate citations
    citations = parse_and_validate_citations(
        answer_text=llm_resp.content,
        source_registry=source_registry,
        retrieved_results=retrieved_results,
    )
    llm_resp.citations = citations

    return llm_resp, citations
