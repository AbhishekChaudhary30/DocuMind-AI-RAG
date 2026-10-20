"""Query execution service coordinating retrieval, generation, and telemetry."""

from typing import List, Optional, Tuple
from app.core.exceptions import ValidationError
from app.models.schemas import LLMResponse, RetrievalResult
from app.rag.pipeline import RAGPipeline


def execute_document_query(
    query: str,
    pipeline: RAGPipeline,
    top_k: int = 5,
    retrieval_mode: str = "hybrid",
    doc_ids: Optional[List[str]] = None,
    conversation_history: str = "",
    context_budget_tokens: int = 3000,
    temperature: float = 0.2,
) -> Tuple[LLMResponse, List[RetrievalResult]]:
    """Validates query and coordinates full RAG pipeline execution."""
    if not query or not query.strip():
        raise ValidationError("Query cannot be empty.")

    clean_query = query.strip()

    response, results = pipeline.answer_query(
        query=clean_query,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
        doc_ids=doc_ids,
        conversation_history=conversation_history,
        context_budget_tokens=context_budget_tokens,
        temperature=temperature,
    )

    return response, results
