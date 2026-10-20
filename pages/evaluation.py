"""Evaluation Page: Real-time pipeline diagnostics, benchmark audit suite, and evaluation framework."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import streamlit as st

from app.core.config import get_settings
from app.evaluation.benchmark_dataset import BENCHMARK_CASES
from app.evaluation.runner import run_rag_benchmark
from app.services.document_service import process_uploaded_pdf
from app.services.export_service import (
    export_benchmark_csv,
    export_benchmark_json,
    export_benchmark_markdown,
)
from app.services.indexing_service import index_processed_document
from app.ui.branding import render_footer
from app.ui.components import (
    render_metric_card,
    render_next_page_button,
    render_section_header,
)
from app.ui.layout import get_active_rag_pipeline, page_setup

settings = get_settings()

page_setup(
    page_title="RAG Evaluation & Benchmark Telemetry",
    subtitle="Quantitative benchmark auditing, citation verification metrics, and retrieval diagnostics.",
)

pipeline = get_active_rag_pipeline()

# Section 1: Interactive Benchmark Runner
render_section_header(
    title="Quantitative RAG Benchmark Suite",
    subtitle="Execute deterministic evaluation across verified ground-truth test cases.",
    badge_text="Automated Evaluation",
)

st.markdown(
    """
    <div class="dm-card" style="border-left: 3px solid #7C3AED; margin-bottom: 1rem;">
        <div style="font-weight: 700; color: #FFFFFF; font-size: 1.05rem; margin-bottom: 0.25rem;">
            Deterministic Benchmark Suite (8 Multi-Scenario Cases)
        </div>
        <div style="font-size: 0.86rem; color: #94A3B8; line-height: 1.5;">
            Evaluates factual retrieval, technical keyword matching, clause policy grounding, semantic paraphrasing,
            multi-document synthesis, and unanswerable question refusal across the curated enterprise sample collection.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_run1, col_run2 = st.columns([2, 1])

with col_run1:
    run_bench_btn = st.button("🚀 Run RAG Benchmark Suite", type="primary", use_container_width=True)

with col_run2:
    if st.session_state.get("active_benchmark_result"):
        if st.button("Clear Benchmark Results ✕", use_container_width=True):
            st.session_state.active_benchmark_result = None
            st.rerun()

# Execution logic
if run_bench_btn:
    # Check if sample documents are indexed
    samples_dir = Path("data/samples")
    sample_files = ["technical_architecture_guide.pdf", "business_financial_report.pdf", "ai_governance_policy.pdf"]
    
    # Auto-index samples if index is currently empty
    if st.session_state.vector_store.count == 0:
        with st.spinner("Pre-indexing curated sample documents for benchmark suite..."):
            for fname in sample_files:
                fpath = samples_dir / fname
                if fpath.exists():
                    with open(fpath, "rb") as f:
                        file_bytes = f.read()
                    pdoc, _, _ = process_uploaded_pdf(fname, file_bytes, settings)
                    if pdoc:
                        st.session_state.processed_documents[pdoc.doc_id] = pdoc
                        index_processed_document(
                            pdoc,
                            st.session_state.vector_store,
                            st.session_state.keyword_search,
                            pipeline.embedding_provider,
                        )

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def update_prog(cur: int, total: int, msg: str):
        progress_bar.progress(cur / total)
        status_text.text(f"Progress [{cur}/{total}]: {msg}")

    with st.spinner("Executing benchmark queries, calculating MRR, Precision, Recall, and Citation Grounding..."):
        suite_res = run_rag_benchmark(
            pipeline=pipeline,
            cases=BENCHMARK_CASES,
            top_k=st.session_state.get("retrieval_top_k", 5),
            retrieval_mode=st.session_state.get("retrieval_mode", "hybrid"),
            progress_callback=update_prog,
        )
        st.session_state.active_benchmark_result = suite_res
        progress_bar.empty()
        status_text.empty()
        st.success(f"✓ Benchmark completed: {suite_res.passed_cases}/{suite_res.total_cases} test cases passed!")

# Render Benchmark Results if Available
bench: Optional[Any] = st.session_state.get("active_benchmark_result")
if bench and hasattr(bench, "case_results") and hasattr(bench, "overall_hit_rate_at_3"):
    # Summary Metric Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        hit_pct = f"{bench.overall_hit_rate_at_3 * 100:.1f}%"
        render_metric_card("Hit Rate @ 3", hit_pct, "Gold doc in top 3", "Retrieval")
    with m2:
        render_metric_card("MRR", f"{bench.overall_mrr:.3f}", "Mean Reciprocal Rank", "Ranking")
    with m3:
        cit_pct = f"{bench.overall_citation_precision * 100:.1f}%"
        render_metric_card("Citation Precision", cit_pct, "Page-accurate citations", "Grounding")
    with m4:
        ground_pct = f"{bench.grounding_accuracy * 100:.1f}%"
        render_metric_card("Grounding / Refusal", ground_pct, "Factual pass rate", "Accuracy")
    with m5:
        render_metric_card("Avg Latency", f"{bench.avg_latency_seconds:.3f}s", "Roundtrip query time", "Performance")

    # Detailed Per-Case Table
    st.markdown("#### Detailed Case-by-Case Benchmark Audit")
    
    table_data = []
    for c in bench.case_results:
        table_data.append({
            "Case ID": getattr(c, "case_id", "N/A"),
            "Category": getattr(c, "category", "N/A"),
            "Status": "✓ PASS" if getattr(c, "status", "") == "PASS" else "❌ FAIL",
            "Hit @ 3": f"{getattr(c, 'hit_at_3', 0.0):.1f}",
            "MRR": f"{getattr(c, 'mrr', 0.0):.2f}",
            "Cit. Prec.": f"{getattr(c, 'citation_precision', 0.0) * 100:.0f}%",
            "Latency": f"{getattr(c, 'latency_seconds', 0.0)}s",
            "Question": getattr(c, "question", ""),
            "Expected Target": getattr(c, "expected_doc", ""),
            "Top Retrieved": getattr(c, "retrieved_doc", ""),
        })
    
    st.dataframe(table_data, use_container_width=True)

    # Export Controls for Benchmark Suite
    st.markdown("##### 📥 Export Benchmark Audit Report")
    exp_col1, exp_col2, exp_col3 = st.columns(3)
    
    with exp_col1:
        csv_str = export_benchmark_csv(bench)
        st.download_button(
            "Download Benchmark (CSV) 📊",
            data=csv_str,
            file_name="documind_rag_benchmark.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with exp_col2:
        json_str = export_benchmark_json(bench)
        st.download_button(
            "Download Benchmark (JSON) 📦",
            data=json_str,
            file_name="documind_rag_benchmark.json",
            mime="application/json",
            use_container_width=True,
        )
    with exp_col3:
        md_str = export_benchmark_markdown(bench)
        st.download_button(
            "Download Audit Report (MD) 📝",
            data=md_str,
            file_name="documind_rag_benchmark.md",
            mime="text/markdown",
            use_container_width=True,
        )
elif bench:
    st.warning("⚠️ Stored benchmark result is in an unexpected format. Click '🚀 Run RAG Benchmark Suite' above to generate fresh results.")
else:
    st.info("ℹ️ No active benchmark results in this session. Click **'🚀 Run RAG Benchmark Suite'** above to execute evaluation across all 8 ground-truth test cases.")

# Section 2: Active Pipeline Operational Diagnostics
render_section_header(
    title="Active Pipeline Operational Diagnostics",
    subtitle="Live quantitative telemetry from the current session's retrieval and inference engine.",
    badge_text="Live Telemetry",
)

docs = st.session_state.uploaded_documents
vector_count = st.session_state.vector_store.count if "vector_store" in st.session_state and st.session_state.vector_store else 0
bm25_count = st.session_state.keyword_search.count if "keyword_search" in st.session_state and st.session_state.keyword_search else 0
last_lat = st.session_state.get("last_query_latency")
last_latency_str = f"{last_lat} s" if last_lat is not None else "No queries yet"

col1, col2, col3, col4 = st.columns(4)
with col1:
    render_metric_card("Corpus Documents", str(len(docs)), "Loaded in active session", "Documents")
with col2:
    render_metric_card("Searchable Chunks", str(vector_count), "In FAISS & BM25 indices", "Vector Store")
with col3:
    ret_mode = st.session_state.get("retrieval_mode", "hybrid").upper()
    render_metric_card("Active Retrieval", ret_mode, f"Top-K: {st.session_state.get('retrieval_top_k', 5)} candidates", "Engine")
with col4:
    render_metric_card("Last Query Latency", last_latency_str, "End-to-end roundtrip", "Performance")

# Recent Query Citation Audit
st.markdown("#### Live Citation Verification Audit")
chat_history = st.session_state.get("chat_history", [])
if chat_history:
    last_interaction = chat_history[-1]
    cits = last_interaction.get("citations", [])
    verified_count = sum(1 for c in cits if c.is_valid)
    invalid_count = sum(1 for c in cits if not c.is_valid)

    ca1, ca2, ca3 = st.columns(3)
    with ca1:
        render_metric_card("Total Citations Emitted", str(len(cits)), f"Query: '{last_interaction['question'][:30]}...'", "Citations")
    with ca2:
        render_metric_card("Verified Source Anchors", str(verified_count), "Matched authentic context chunk", "Grounding")
    with ca3:
        status_label = "0 Invalid" if invalid_count == 0 else f"{invalid_count} Flagged"
        badge = "Clean" if invalid_count == 0 else "Audit Alert"
        render_metric_card("Hallucinated Source IDs", status_label, "Out-of-context source references", badge)
else:
    st.info("ℹ️ Execute a query in Workspace to inspect live citation provenance auditing.")

# Section 3: Deep Dive Metric Specifications Table
render_section_header(
    title="RAG Evaluation Framework Specifications",
    subtitle="Formulations and threshold criteria evaluated in DocuMind AI.",
    badge_text="Ragas / TruLens Alignment",
)

evaluation_catalog = [
    {
        "Metric": "Hit Rate @ 3",
        "Pipeline Stage": "Dense & Hybrid Retrieval",
        "Methodology": "Binary indicator whether gold passage appears in top-3 candidates",
        "Target SLA": "≥ 90.0%",
        "Phase 3 Status": "Active & Measured",
    },
    {
        "Metric": "Mean Reciprocal Rank (MRR)",
        "Pipeline Stage": "Candidate Re-ranking",
        "Methodology": "1 / rank of highest relevant document chunk",
        "Target SLA": "≥ 0.82",
        "Phase 3 Status": "Active & Measured",
    },
    {
        "Metric": "Groundedness / Faithfulness",
        "Pipeline Stage": "Answer Synthesis",
        "Methodology": "Ratio of claims directly corroborated by cited source snippets",
        "Target SLA": "≥ 95.0%",
        "Phase 3 Status": "Audited via Keywords & Refusal",
    },
    {
        "Metric": "Citation Exact-Match",
        "Pipeline Stage": "Provenance Verification",
        "Methodology": "Verbatim string alignment of quoted excerpt against 1-based source page",
        "Target SLA": "100.0%",
        "Phase 3 Status": "Enforced by Source Registry",
    },
    {
        "Metric": "End-to-End Latency",
        "Pipeline Stage": "System Performance",
        "Methodology": "Total roundtrip latency from query input to final answer",
        "Target SLA": "< 2.50 s",
        "Phase 3 Status": f"Measured ({last_latency_str})",
    },
]

st.dataframe(evaluation_catalog, use_container_width=True)

# Section 4: Failure Mode Defense Matrix
render_section_header(
    title="Failure Mode Defense Matrix",
    subtitle="Defensive engineering guards active in DocuMind AI.",
    badge_text="Defensive AI",
)

col_f1, col_f2 = st.columns(2)
with col_f1:
    with st.container(border=True):
        st.markdown("**✓ Retrieval Gaps: Guarded by Hybrid RRF**")
        st.caption(
            "Reciprocal Rank Fusion blends dense FAISS semantic vectors with sparse BM25 keyword matching, "
            "ensuring that technical jargon, numbers, and conceptual queries both resolve successfully."
        )

    with st.container(border=True):
        st.markdown("**✓ Premise Hallucination: Guarded by Prompt Constraints**")
        st.caption(
            "System prompts mandate strict refusal when retrieved evidence is absent, returning explicit "
            "insufficient evidence disclosures rather than fabricated assertions."
        )

with col_f2:
    with st.container(border=True):
        st.markdown("**✓ Phantom Citations: Guarded by Source Registry**")
        st.caption(
            "Every citation emitted by the LLM is validated against the active request's authenticated chunk registry. "
            "Hallucinated page numbers or invented sources are immediately intercepted and flagged."
        )

    with st.container(border=True):
        st.markdown("**✓ Prompt Injection: Guarded by Content Neutralization**")
        st.caption(
            "Extracted PDF text is enclosed within strict XML boundaries and scanned for adversarial instructions "
            "(e.g., 'ignore previous instructions') which are neutralized before prompt assembly."
        )

# Navigation
render_next_page_button(
    label="View System Architecture & About",
    target_page="pages/about.py",
    icon="ℹ️",
)

render_footer()
