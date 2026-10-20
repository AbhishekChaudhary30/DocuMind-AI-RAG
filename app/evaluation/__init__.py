"""Evaluation subsystem for quantitative RAG benchmark execution and metrics."""

from app.evaluation.benchmark_dataset import BENCHMARK_CASES, BenchmarkCase
from app.evaluation.metrics import (
    calculate_hit_rate,
    calculate_mrr,
    calculate_precision_recall,
    evaluate_citation_precision,
    evaluate_grounding_and_abstention,
)
from app.evaluation.runner import run_rag_benchmark

__all__ = [
    "BENCHMARK_CASES",
    "BenchmarkCase",
    "calculate_hit_rate",
    "calculate_mrr",
    "calculate_precision_recall",
    "evaluate_citation_precision",
    "evaluate_grounding_and_abstention",
    "run_rag_benchmark",
]
