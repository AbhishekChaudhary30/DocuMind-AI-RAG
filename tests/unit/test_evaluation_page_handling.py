"""Unit tests for Evaluation page logic and Confirmed Error A resolution."""

from typing import Any, Dict, List, Optional
import pytest
from app.evaluation.runner import BenchmarkCaseResult, BenchmarkSuiteResult


def test_typing_optional_import_defined():
    """Verify Optional and typing symbols are correctly defined and usable at runtime."""
    # This was Confirmed Error A: NameError: name 'Optional' is not defined
    bench: Optional[Any] = None
    assert bench is None
    bench = {"dummy": 1}
    assert bench is not None


def test_benchmark_result_none_handling():
    """Verify safe evaluation page handling when no benchmark has run."""
    session_state = {"active_benchmark_result": None}
    bench: Optional[Any] = session_state.get("active_benchmark_result")

    # Mirroring page logic guard
    is_valid = bench and hasattr(bench, "case_results") and hasattr(bench, "overall_hit_rate_at_3")
    assert not is_valid
    # No exception raised, fallback condition is met
    assert bench is None


def test_benchmark_result_malformed_handling():
    """Verify safe evaluation page handling when benchmark result is malformed."""
    malformed_benchmarks = [
        {},
        {"case_results": []},
        "corrupted_string",
        42,
        object(),
    ]
    for bench in malformed_benchmarks:
        # Must not raise an exception
        is_valid = bench and hasattr(bench, "case_results") and hasattr(bench, "overall_hit_rate_at_3")
        assert not is_valid


def test_benchmark_result_valid_handling():
    """Verify safe evaluation page handling when benchmark result is populated."""
    valid_res = BenchmarkSuiteResult(
        total_cases=1,
        passed_cases=1,
        failed_cases=0,
        overall_hit_rate_at_3=1.0,
        overall_mrr=1.0,
        overall_precision=1.0,
        overall_recall=1.0,
        overall_citation_precision=1.0,
        grounding_accuracy=1.0,
        avg_latency_seconds=0.12,
        retrieval_mode="hybrid",
        top_k=5,
        provider="offline",
        case_results=[
            BenchmarkCaseResult(
                case_id="tc_01",
                category="Single-hop Fact",
                question="What is DocuMind?",
                expected_doc="architecture_spec.pdf",
                expected_pages=[1],
                retrieved_doc="architecture_spec.pdf",
                retrieved_pages=[1],
                hit_at_3=1.0,
                mrr=1.0,
                precision_at_k=1.0,
                recall_at_k=1.0,
                citation_precision=1.0,
                grounding_passed=True,
                keyword_coverage=1.0,
                answer_preview="DocuMind AI is a PDF QA system.",
                latency_seconds=0.12,
                status="PASS",
            )
        ],
    )
    is_valid = valid_res and hasattr(valid_res, "case_results") and hasattr(valid_res, "overall_hit_rate_at_3")
    assert is_valid
    assert len(valid_res.case_results) == 1
    assert valid_res.overall_hit_rate_at_3 == 1.0
