"""Integration test for the RAG benchmark runner and metrics suite."""

from pathlib import Path
import pytest
from app.embeddings.local_embeddings import FastLocalEmbedder
from app.evaluation.benchmark_dataset import BENCHMARK_CASES
from app.evaluation.runner import run_rag_benchmark
from app.llm.providers import OfflineExtractiveProvider
from app.rag.pipeline import RAGPipeline
from app.retrieval.keyword_search import BM25KeywordSearch
from app.retrieval.vector_store import FaissVectorStore
from app.services.document_service import process_uploaded_pdf
from app.services.indexing_service import index_processed_document


@pytest.fixture
def populated_rag_pipeline():
    vstore = FaissVectorStore(dimension=384)
    bm25 = BM25KeywordSearch()
    embedder = FastLocalEmbedder(dimension=384)
    llm = OfflineExtractiveProvider()

    # Index the 3 sample documents
    samples_dir = Path("data/samples")
    sample_files = ["technical_architecture_guide.pdf", "business_financial_report.pdf", "ai_governance_policy.pdf"]
    for fname in sample_files:
        fpath = samples_dir / fname
        if fpath.exists():
            with open(fpath, "rb") as f:
                pdoc, _, _ = process_uploaded_pdf(fname, f.read())
                if pdoc:
                    index_processed_document(pdoc, vstore, bm25, embedder)

    pipeline = RAGPipeline(
        vector_store=vstore,
        keyword_search=bm25,
        embedding_provider=embedder,
        llm_provider=llm,
    )
    return pipeline


def test_run_rag_benchmark_full_suite(populated_rag_pipeline):
    suite_res = run_rag_benchmark(
        pipeline=populated_rag_pipeline,
        cases=BENCHMARK_CASES,
        top_k=4,
        retrieval_mode="hybrid",
    )

    assert suite_res.total_cases == len(BENCHMARK_CASES)
    assert suite_res.passed_cases > 0
    assert suite_res.overall_hit_rate_at_3 >= 0.80
    assert suite_res.overall_mrr >= 0.70
    assert suite_res.overall_citation_precision >= 0.80
    assert suite_res.grounding_accuracy >= 0.80
    assert suite_res.avg_latency_seconds < 1.0

    # Ensure every case result has non-empty fields
    for cr in suite_res.case_results:
        assert cr.case_id.startswith("TC-")
        assert len(cr.question) > 0
        assert cr.status in ["PASS", "FAIL"]
