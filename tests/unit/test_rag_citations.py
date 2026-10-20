"""Unit tests for ContextBuilder, CitationValidator, and grounding rules."""

import pytest
from app.rag.context_builder import build_context_block
from app.rag.citations import parse_and_validate_citations
from app.llm.providers import OfflineExtractiveProvider
from app.models.schemas import DocumentChunk, RetrievalResult


@pytest.fixture
def mock_retrieval_results():
    c1 = DocumentChunk(
        chunk_id="chk-101",
        doc_id="doc-guide",
        filename="system_architecture.pdf",
        chunk_index=0,
        start_page=3,
        end_page=3,
        page_numbers=[3],
        text="The vector database index FlatIP performs cosine similarity on normalized unit vectors.",
        estimated_tokens=14,
    )
    c2 = DocumentChunk(
        chunk_id="chk-102",
        doc_id="doc-guide",
        filename="system_architecture.pdf",
        chunk_index=1,
        start_page=4,
        end_page=5,
        page_numbers=[4, 5],
        text="Hybrid search uses Reciprocal Rank Fusion with constant k=60 to merge dense and keyword search.",
        estimated_tokens=16,
    )
    return [
        RetrievalResult(chunk=c1, score=0.88, retrieval_mode="hybrid", rank=1),
        RetrievalResult(chunk=c2, score=0.75, retrieval_mode="hybrid", rank=2),
    ]


def test_build_context_block_budget_and_mapping(mock_retrieval_results):
    context_text, source_registry, is_truncated = build_context_block(mock_retrieval_results, max_tokens=1000)
    
    assert "[Source 1]" in context_text
    assert "[Source 2]" in context_text
    assert len(source_registry) > 0
    assert "source 1" in source_registry
    assert source_registry["source 1"].chunk_id == "chk-101"
    assert is_truncated is False


def test_citation_validator_valid_and_invalid_sources(mock_retrieval_results):
    _, source_registry, _ = build_context_block(mock_retrieval_results)
    
    # Model text references [Source 1] and an invented [Source 99]
    answer_text = "The system uses cosine similarity [Source 1] and something completely fabricated [Source 99]."
    
    validated_citations = parse_and_validate_citations(
        answer_text=answer_text,
        source_registry=source_registry,
        retrieved_results=mock_retrieval_results,
    )
    
    assert len(validated_citations) == 2
    # First citation is valid
    cit_valid = [c for c in validated_citations if c.is_valid][0]
    assert cit_valid.doc_id == "doc-guide"
    assert cit_valid.page_number == 3
    assert cit_valid.verification_status == "verified"
    
    # Second citation is invalid
    cit_invalid = [c for c in validated_citations if not c.is_valid][0]
    assert cit_invalid.verification_status == "invalid_source"


def test_citation_validator_fallback_when_no_tags(mock_retrieval_results):
    _, source_registry, _ = build_context_block(mock_retrieval_results)
    
    # Model answered without citation tags
    answer_text = "The vector database performs cosine similarity on normalized vectors."
    validated_citations = parse_and_validate_citations(
        answer_text=answer_text,
        source_registry=source_registry,
        retrieved_results=mock_retrieval_results,
    )
    
    # Should fall back to associating available context sources with disclaimer
    assert len(validated_citations) >= 1
    assert validated_citations[0].verification_status == "retrieved_evidence"


def test_offline_extractive_provider_answers_grounded(mock_retrieval_results):
    context_text, _, _ = build_context_block(mock_retrieval_results)
    prompt = f"<context>\n{context_text}\n</context>\nUSER QUESTION: Tell me about cosine similarity in vector database\n\nPlease answer:"
    
    provider = OfflineExtractiveProvider()
    response = provider.generate(
        prompt=prompt,
        system_prompt="System prompt",
    )
    
    assert len(response.content) > 0
    assert "[Source 1]" in response.content
    assert response.provider == "local-offline"


def test_offline_extractive_provider_insufficient_evidence():
    prompt = "<context>\n\n</context>\nUSER QUESTION: What is the stock price of Apple on NASDAQ?\n\nPlease answer:"
    
    provider = OfflineExtractiveProvider()
    response = provider.generate(
        prompt=prompt,
        system_prompt="System prompt",
    )
    
    assert "insufficient" in response.content.lower()
