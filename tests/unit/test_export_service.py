"""Unit tests for the export service across Markdown, JSON, and CSV formats."""

import json
import pytest
from app.services.export_service import (
    export_analysis_json,
    export_analysis_markdown,
    export_answer_json,
    export_answer_markdown,
    export_benchmark_csv,
    export_benchmark_json,
    export_benchmark_markdown,
    sanitize_filename,
)
from app.evaluation.runner import BenchmarkCaseResult, BenchmarkSuiteResult
from app.models.schemas import AnalysisResult, Citation, DocumentChunk, RetrievalResult


def test_sanitize_filename():
    assert sanitize_filename("Simple File.pdf") == "Simple_File.pdf"
    assert sanitize_filename("bad/\\:*?\"<>|path") == "bad_path"
    assert sanitize_filename("") == "documind_export"


def test_export_answer_markdown_and_json():
    cit = Citation(
        citation_id="cit1",
        chunk_id="c1",
        doc_id="d1",
        filename="guide.pdf",
        page_number=1,
        snippet="System uses PyMuPDF",
        relevance_score=0.95,
        is_valid=True,
    )
    chunk = DocumentChunk(
        chunk_id="c1",
        doc_id="d1",
        filename="guide.pdf",
        chunk_index=0,
        text="Full chunk text here",
        start_page=1,
        end_page=1,
    )
    ret = RetrievalResult(chunk=chunk, score=0.95, rank=1, retrieval_mode="hybrid")

    # 1. Markdown Export
    md = export_answer_markdown(
        question="What library is used?",
        answer="The system uses PyMuPDF [Source 1].",
        citations=[cit],
        retrieved_results=[ret],
        provider="local-offline",
        model="local-extractive",
        retrieval_mode="hybrid",
        latency_seconds=0.042,
    )
    assert "# DocuMind AI — Q&A Evidence Report" in md
    assert "What library is used?" in md
    assert "PyMuPDF" in md
    assert "Page 1" in md
    assert "sk-" not in md  # No secrets

    # 2. JSON Export
    json_str = export_answer_json(
        question="What library is used?",
        answer="The system uses PyMuPDF [Source 1].",
        citations=[cit],
        retrieved_results=[ret],
        provider="local-offline",
        model="local-extractive",
        retrieval_mode="hybrid",
        latency_seconds=0.042,
    )
    parsed = json.loads(json_str)
    assert parsed["report_type"] == "qa_evidence_report"
    assert parsed["query"] == "What library is used?"
    assert len(parsed["citations"]) == 1
    assert parsed["citations"][0]["filename"] == "guide.pdf"


def test_export_analysis_markdown_and_json():
    analysis = AnalysisResult(
        document_titles=["tech_guide.pdf"],
        summary="This document outlines microservices.",
        key_topics=["Architecture", "Docker"],
        important_facts=["Achieved 99.9% uptime."],
        key_entities={"Technical Terms": ["PYMUPDF", "FAISS"]},
        suggested_questions=["How is Docker configured?"],
        structural_stats={"Total Documents": 1, "Total Pages": 3},
    )

    md = export_analysis_markdown(analysis)
    assert "# DocuMind AI — Document Intelligence Report" in md
    assert "microservices" in md
    assert "Docker" in md

    json_str = export_analysis_json(analysis)
    data = json.loads(json_str)
    assert data["report_type"] == "document_intelligence_report"
    assert data["key_topics"] == ["Architecture", "Docker"]


def test_export_benchmark_formats():
    case = BenchmarkCaseResult(
        case_id="TC-01",
        category="Direct Factual",
        question="Growth rate?",
        expected_doc="report.pdf",
        expected_pages=[1],
        retrieved_doc="report.pdf",
        retrieved_pages=[1],
        hit_at_3=1.0,
        mrr=1.0,
        precision_at_k=1.0,
        recall_at_k=1.0,
        citation_precision=1.0,
        grounding_passed=True,
        keyword_coverage=1.0,
        answer_preview="28% revenue growth",
        latency_seconds=0.015,
        status="PASS",
    )
    suite = BenchmarkSuiteResult(
        total_cases=1,
        passed_cases=1,
        failed_cases=0,
        overall_hit_rate_at_3=1.0,
        overall_mrr=1.0,
        overall_precision=1.0,
        overall_recall=1.0,
        overall_citation_precision=1.0,
        grounding_accuracy=1.0,
        avg_latency_seconds=0.015,
        retrieval_mode="hybrid",
        top_k=3,
        provider="local-offline",
        case_results=[case],
    )

    csv_data = export_benchmark_csv(suite)
    assert "Case ID,Category,Status" in csv_data
    assert "TC-01,Direct Factual,PASS" in csv_data

    json_data = export_benchmark_json(suite)
    parsed = json.loads(json_data)
    assert parsed["overall_hit_rate_at_3"] == 1.0

    md_data = export_benchmark_markdown(suite)
    assert "# DocuMind AI — RAG Evaluation & Benchmark Audit Report" in md_data
    assert "TC-01" in md_data
