"""Unit tests for quantitative RAG evaluation metrics."""

import pytest
from app.evaluation.metrics import (
    calculate_hit_rate,
    calculate_mrr,
    calculate_precision_recall,
    evaluate_citation_precision,
    evaluate_grounding_and_abstention,
)
from app.models.schemas import Citation, DocumentChunk, RetrievalResult


@pytest.fixture
def sample_retrieval_candidates():
    c1 = DocumentChunk(
        chunk_id="c1",
        doc_id="doc1",
        filename="report_2025.pdf",
        chunk_index=0,
        text="Sample text 1",
        start_page=1,
        end_page=1,
        page_numbers=[1],
    )
    c2 = DocumentChunk(
        chunk_id="c2",
        doc_id="doc2",
        filename="guide_arch.pdf",
        chunk_index=0,
        text="Sample text 2",
        start_page=2,
        end_page=3,
        page_numbers=[2, 3],
    )
    c3 = DocumentChunk(
        chunk_id="c3",
        doc_id="doc3",
        filename="policy_sec.pdf",
        chunk_index=0,
        text="Sample text 3",
        start_page=1,
        end_page=1,
        page_numbers=[1],
    )
    return [
        RetrievalResult(chunk=c1, score=0.92, rank=1, retrieval_mode="hybrid"),
        RetrievalResult(chunk=c2, score=0.85, rank=2, retrieval_mode="hybrid"),
        RetrievalResult(chunk=c3, score=0.74, rank=3, retrieval_mode="hybrid"),
    ]


def test_calculate_hit_rate(sample_retrieval_candidates):
    # Hit at rank 1
    hit = calculate_hit_rate(sample_retrieval_candidates, "report_2025", [1], k=3)
    assert hit == 1.0

    # Hit at rank 2
    hit_2 = calculate_hit_rate(sample_retrieval_candidates, "guide_arch", [2], k=3)
    assert hit_2 == 1.0

    # Page mismatch
    hit_miss_page = calculate_hit_rate(sample_retrieval_candidates, "guide_arch", [99], k=3)
    assert hit_miss_page == 0.0

    # Unrelated document
    hit_miss_doc = calculate_hit_rate(sample_retrieval_candidates, "nonexistent", [1], k=3)
    assert hit_miss_doc == 0.0


def test_calculate_mrr(sample_retrieval_candidates):
    # First candidate matches -> MRR = 1/1 = 1.0
    mrr_1 = calculate_mrr(sample_retrieval_candidates, "report_2025", [1])
    assert mrr_1 == 1.0

    # Second candidate matches -> MRR = 1/2 = 0.5
    mrr_2 = calculate_mrr(sample_retrieval_candidates, "guide_arch", [2])
    assert mrr_2 == 0.5

    # Third candidate matches -> MRR = 1/3 = 0.3333
    mrr_3 = calculate_mrr(sample_retrieval_candidates, "policy_sec", [1])
    assert mrr_3 == 0.3333

    # Miss -> MRR = 0.0
    mrr_0 = calculate_mrr(sample_retrieval_candidates, "nonexistent", [1])
    assert mrr_0 == 0.0


def test_calculate_precision_recall(sample_retrieval_candidates):
    # In top 3, 1 out of 3 is relevant to report_2025
    prec, rec = calculate_precision_recall(sample_retrieval_candidates, "report_2025", [1], k=3)
    assert round(prec, 2) == 0.33
    assert rec == 1.0


def test_evaluate_citation_precision():
    citations = [
        Citation(
            citation_id="cit1",
            chunk_id="c1",
            doc_id="d1",
            filename="annual_report.pdf",
            page_number=1,
            snippet="Revenue grew 28%",
            relevance_score=0.9,
            is_valid=True,
        ),
        Citation(
            citation_id="cit2",
            chunk_id="c2",
            doc_id="d2",
            filename="unknown.pdf",
            page_number=99,
            snippet="Hallucinated text",
            relevance_score=0.0,
            is_valid=False,
        ),
    ]

    prec = evaluate_citation_precision(citations, "annual_report", [1])
    assert prec == 0.5


def test_evaluate_grounding_and_abstention():
    # 1. Answerable query
    passed, cov = evaluate_grounding_and_abstention(
        answer_text="Enterprise Tech Group achieved 28% year-over-year revenue growth in FY2025.",
        expected_behavior="answer",
        expected_keywords=["28%", "growth", "revenue"],
    )
    assert passed is True
    assert cov == 1.0

    # 2. Unanswerable query correctly abstaining
    passed_abs, _ = evaluate_grounding_and_abstention(
        answer_text="The provided document evidence is insufficient to answer this question.",
        expected_behavior="abstain",
        expected_keywords=[],
    )
    assert passed_abs is True

    # 3. Unanswerable query hallucinating without abstention
    passed_halluc, _ = evaluate_grounding_and_abstention(
        answer_text="Tesla stock closed at 248.50 on NASDAQ yesterday.",
        expected_behavior="abstain",
        expected_keywords=[],
    )
    assert passed_halluc is False
