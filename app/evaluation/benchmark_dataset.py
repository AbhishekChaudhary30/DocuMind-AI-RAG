"""Curated benchmark dataset for deterministic RAG evaluation across sample documents."""

from typing import List, Optional
from pydantic import BaseModel, Field


class BenchmarkCase(BaseModel):
    """Encapsulates a ground-truth benchmark evaluation test case."""

    case_id: str = Field(description="Unique benchmark test case identifier, e.g. TC-01")
    category: str = Field(description="Test scenario category (factual, keyword, multi-doc, unanswerable, etc.)")
    question: str = Field(description="User query submitted to the RAG pipeline")
    expected_doc_substring: str = Field(description="Expected filename substring containing gold evidence")
    expected_pages: List[int] = Field(default_factory=list, description="1-based source pages containing gold evidence")
    expected_answer_keywords: List[str] = Field(default_factory=list, description="Mandatory factual terms expected in response")
    expected_source_excerpt: str = Field(description="Key ground-truth snippet from the source document")
    expected_behavior: str = Field(default="answer", description="'answer' for answerable, 'abstain' for unanswerable")
    evaluation_criteria: str = Field(description="Explicit human-readable rubric for assessing answer correctness")


BENCHMARK_CASES: List[BenchmarkCase] = [
    # 1. Direct Factual Question
    BenchmarkCase(
        case_id="TC-01",
        category="Direct Factual",
        question="What was the year-over-year revenue growth rate reported in the FY2025 Financial Performance Report?",
        expected_doc_substring="business_financial_report",
        expected_pages=[1],
        expected_answer_keywords=["28%", "growth", "revenue"],
        expected_source_excerpt="Enterprise Tech Group demonstrated sustained 28% year-over-year revenue growth across FY2025",
        expected_behavior="answer",
        evaluation_criteria="Answer must state that revenue grew by 28% year-over-year in FY2025 and cite Page 1.",
    ),
    # 2. Exact Technical Keyword Question
    BenchmarkCase(
        case_id="TC-02",
        category="Exact Keyword",
        question="Which library does the DocuMind AI ingestion engine use for structural document extraction?",
        expected_doc_substring="technical_architecture_guide",
        expected_pages=[1],
        expected_answer_keywords=["PyMuPDF", "extraction"],
        expected_source_excerpt="In DocuMind AI, the ingestion engine uses PyMuPDF for structural document extraction.",
        expected_behavior="answer",
        evaluation_criteria="Answer must identify PyMuPDF as the structural extraction library and cite the technical architecture guide.",
    ),
    # 3. Governance Policy Mandate (Clause Correctness)
    BenchmarkCase(
        case_id="TC-03",
        category="Policy Compliance",
        question="What is the policy identifier and effective date for the Enterprise AI Governance policy?",
        expected_doc_substring="ai_governance_policy",
        expected_pages=[1],
        expected_answer_keywords=["POL-SEC-2025-04", "January 1, 2025"],
        expected_source_excerpt="Policy ID: POL-SEC-2025-04 — Effective Date: January 1, 2025",
        expected_behavior="answer",
        evaluation_criteria="Answer must report Policy ID 'POL-SEC-2025-04' and Effective Date 'January 1, 2025' with page citation.",
    ),
    # 4. Semantic Paraphrase Question
    BenchmarkCase(
        case_id="TC-04",
        category="Semantic Paraphrase",
        question="How does the system ensure high fidelity across specialized technical terms during search?",
        expected_doc_substring="technical_architecture_guide",
        expected_pages=[1],
        expected_answer_keywords=["BM25", "hybrid", "dense vector"],
        expected_source_excerpt="Retrieval uses dense vector search augmented with BM25 hybrid indexing to ensure maximum precision across technical vocabularies.",
        expected_behavior="answer",
        evaluation_criteria="Answer must explain that BM25 hybrid indexing augments dense vector search to capture technical terminology.",
    ),
    # 5. Financial Tabular Extraction
    BenchmarkCase(
        case_id="TC-05",
        category="Tabular Data",
        question="What was the reported R&D expense in Q1 and Q3 of FY25?",
        expected_doc_substring="business_financial_report",
        expected_pages=[1],
        expected_answer_keywords=["28.1", "33.8"],
        expected_source_excerpt="R&D Expense: Q1 FY25 28.1M, Q3 FY25 33.8M",
        expected_behavior="answer",
        evaluation_criteria="Answer must state Q1 R&D was 28.1M and Q3 was 33.8M based on the financial report table.",
    ),
    # 6. Multi-Document Synthesis
    BenchmarkCase(
        case_id="TC-06",
        category="Multi-Document",
        question="What grounding threshold triggers refusal in the architecture guide, and what policy section governs secret logging in the AI governance policy?",
        expected_doc_substring="technical_architecture_guide|ai_governance_policy",  # Multi-document spans both guide and policy
        expected_pages=[1],
        expected_answer_keywords=["0.70", "2.1", "logging"],
        expected_source_excerpt="Responses with grounding scores below 0.70 trigger automated fallback ... Section 2.1 Zero Persistent Secret Logging",
        expected_behavior="answer",
        evaluation_criteria="Answer must synthesize findings from both documents: the 0.70 threshold and Section 2.1 for secret logging.",
    ),
    # 7. Unanswerable Question / Required Refusal (Abstention)
    BenchmarkCase(
        case_id="TC-07",
        category="Unanswerable Abstention",
        question="What was the closing price of Tesla (TSLA) stock on the NASDAQ exchange yesterday?",
        expected_doc_substring="",
        expected_pages=[],
        expected_answer_keywords=["insufficient", "not", "evidence"],
        expected_source_excerpt="",
        expected_behavior="abstain",
        evaluation_criteria="Application must explicitly abstain or state that the document evidence is insufficient rather than hallucinating.",
    ),
    # 8. Data Isolation Requirements (Clause Correctness)
    BenchmarkCase(
        case_id="TC-08",
        category="Policy Compliance",
        question="Under Section 2.2 of the governance policy, when must user document data be purged?",
        expected_doc_substring="ai_governance_policy",
        expected_pages=[1],
        expected_answer_keywords=["session", "disconnect", "isolated"],
        expected_source_excerpt="User documents processed within temporary sessions must be isolated per tenant and purged upon session disconnect.",
        expected_behavior="answer",
        evaluation_criteria="Answer must identify that documents must be purged upon session disconnect under Section 2.2.",
    ),
]
