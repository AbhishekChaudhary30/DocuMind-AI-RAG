"""Evaluation execution engine for running deterministic RAG benchmarks."""

import time
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

from app.evaluation.benchmark_dataset import BENCHMARK_CASES, BenchmarkCase
from app.evaluation.metrics import (
    calculate_hit_rate,
    calculate_mrr,
    calculate_precision_recall,
    evaluate_citation_precision,
    evaluate_grounding_and_abstention,
)
from app.rag.pipeline import RAGPipeline


class BenchmarkCaseResult(BaseModel):
    """Encapsulates the evaluation outcome for a single benchmark query."""

    case_id: str
    category: str
    question: str
    expected_doc: str
    expected_pages: List[int]
    retrieved_doc: str
    retrieved_pages: List[int]
    hit_at_3: float
    mrr: float
    precision_at_k: float
    recall_at_k: float
    citation_precision: float
    grounding_passed: bool
    keyword_coverage: float
    answer_preview: str
    latency_seconds: float
    status: str  # PASS / FAIL


class BenchmarkSuiteResult(BaseModel):
    """Aggregate benchmark results across the entire test collection."""

    total_cases: int
    passed_cases: int
    failed_cases: int
    overall_hit_rate_at_3: float
    overall_mrr: float
    overall_precision: float
    overall_recall: float
    overall_citation_precision: float
    grounding_accuracy: float
    avg_latency_seconds: float
    retrieval_mode: str
    top_k: int
    provider: str
    case_results: List[BenchmarkCaseResult] = Field(default_factory=list)


def run_rag_benchmark(
    pipeline: RAGPipeline,
    cases: Optional[List[BenchmarkCase]] = None,
    top_k: int = 5,
    retrieval_mode: str = "hybrid",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> BenchmarkSuiteResult:
    """Executes deterministic benchmark suite against the active RAG pipeline.

    Computes Hit Rate@3, MRR, Precision, Recall, Citation Precision, and Grounding Pass Rate.
    """
    eval_cases = cases or BENCHMARK_CASES
    results: List[BenchmarkCaseResult] = []

    for idx, case in enumerate(eval_cases):
        if progress_callback:
            progress_callback(idx + 1, len(eval_cases), f"Evaluating {case.case_id}: {case.question[:40]}...")

        t0 = time.time()
        response, retrieved = pipeline.answer_query(
            query=case.question,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
        )
        latency = round(time.time() - t0, 3)

        # 1. Retrieval Metrics
        hit_3 = calculate_hit_rate(retrieved, case.expected_doc_substring, case.expected_pages, k=3)
        mrr = calculate_mrr(retrieved, case.expected_doc_substring, case.expected_pages)
        prec, rec = calculate_precision_recall(retrieved, case.expected_doc_substring, case.expected_pages, k=top_k)

        # 2. Citation Metrics
        cit_prec = evaluate_citation_precision(response.citations, case.expected_doc_substring, case.expected_pages)

        # 3. Grounding & Abstention Checks
        grounding_pass, kw_cov = evaluate_grounding_and_abstention(
            answer_text=response.content,
            expected_behavior=case.expected_behavior,
            expected_keywords=case.expected_answer_keywords,
        )

        # Overall Case Status
        is_pass = (hit_3 > 0.0 or case.expected_behavior == "abstain") and grounding_pass
        status = "PASS" if is_pass else "FAIL"

        top_retrieved_doc = retrieved[0].chunk.filename if retrieved else "None"
        top_retrieved_pages = retrieved[0].chunk.page_numbers if retrieved else []

        results.append(
            BenchmarkCaseResult(
                case_id=case.case_id,
                category=case.category,
                question=case.question,
                expected_doc=case.expected_doc_substring or "N/A (Abstention)",
                expected_pages=case.expected_pages,
                retrieved_doc=top_retrieved_doc,
                retrieved_pages=top_retrieved_pages,
                hit_at_3=hit_3,
                mrr=mrr,
                precision_at_k=prec,
                recall_at_k=rec,
                citation_precision=cit_prec,
                grounding_passed=grounding_pass,
                keyword_coverage=kw_cov,
                answer_preview=response.content[:160] + ("..." if len(response.content) > 160 else ""),
                latency_seconds=latency,
                status=status,
            )
        )

    # Compute Aggregates
    n = len(results) if results else 1
    total_passed = sum(1 for r in results if r.status == "PASS")
    total_failed = n - total_passed

    mean_hit_3 = round(sum(r.hit_at_3 for r in results) / n, 4)
    mean_mrr = round(sum(r.mrr for r in results) / n, 4)
    mean_prec = round(sum(r.precision_at_k for r in results) / n, 4)
    mean_rec = round(sum(r.recall_at_k for r in results) / n, 4)
    mean_cit_prec = round(sum(r.citation_precision for r in results) / n, 4)
    mean_grounding = round(sum(1.0 for r in results if r.grounding_passed) / n, 4)
    avg_latency = round(sum(r.latency_seconds for r in results) / n, 3)

    return BenchmarkSuiteResult(
        total_cases=len(results),
        passed_cases=total_passed,
        failed_cases=total_failed,
        overall_hit_rate_at_3=mean_hit_3,
        overall_mrr=mean_mrr,
        overall_precision=mean_prec,
        overall_recall=mean_rec,
        overall_citation_precision=mean_cit_prec,
        grounding_accuracy=mean_grounding,
        avg_latency_seconds=avg_latency,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        provider=getattr(pipeline.llm_provider, "provider_name", getattr(pipeline.llm_provider, "name", "local-provider")),
        case_results=results,
    )
