"""Overview Page: Introduces DocuMind AI, the RAG architecture, and workflow."""

import streamlit as st

from app.core.constants import (
    APP_NAME,
    APP_TAGLINE,
    APP_DESCRIPTION,
    DEVELOPER_CREDIT,
)
from app.ui.branding import render_footer
from app.ui.components import (
    render_metric_card,
    render_next_page_button,
    render_pipeline_steps,
    render_section_header,
)
from app.ui.layout import page_setup

# Setup page layout, styles, and headers
page_setup(
    page_title="Product Overview",
    subtitle="Intelligent document comprehension backed by page-level citation evidence.",
)

# Hero Section
hero_col1, hero_col2 = st.columns([3, 1])
with hero_col1:
    with st.container(border=True):
        st.markdown(f"### {APP_TAGLINE}")
        st.markdown(APP_DESCRIPTION)
        st.markdown(
            """<span class="dm-badge dm-badge-violet">Phase 3 · Production Architecture</span>
            <span class="dm-badge dm-badge-cyan">Zero Secret Persistence</span>
            <span class="dm-badge dm-badge-success">Page-Level Attribution Ready</span>""",
            unsafe_allow_html=True,
        )

with hero_col2:
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    if st.button("🚀 Open Workspace", type="primary", use_container_width=True):
        st.switch_page("pages/workspace.py")
    if st.button("⚙️ Configure Model", use_container_width=True):
        st.switch_page("pages/model_settings.py")
    if st.button("📁 Browse Documents", use_container_width=True):
        st.switch_page("pages/documents.py")

# Document-to-Answer Workflow Pipeline
render_section_header(
    title="Document-to-Answer Workflow",
    subtitle="How DocuMind AI transforms unstructured PDF files into verified answers.",
    badge_text="RAG Architecture",
)

workflow_steps = [
    {
        "title": "Document Ingestion",
        "desc": "Upload standard PDFs. Extracts structural text, layout hierarchy, and tabular elements using PyMuPDF.",
    },
    {
        "title": "Semantic Chunking",
        "desc": "Splits content into coherent contextual blocks with token boundary overlap to preserve semantic relationships.",
    },
    {
        "title": "Vector Indexing",
        "desc": "Generates dense vector embeddings and stores them in an ephemeral vector space for semantic search.",
    },
    {
        "title": "Evidence Retrieval",
        "desc": "Matches queries against high-similarity chunks and ranks passages by relevance score.",
    },
    {
        "title": "Grounded Generation",
        "desc": "Synthesizes precise answers strictly constrained to retrieved context with exact page citations.",
    },
]
render_pipeline_steps(workflow_steps, active_step_index=0)

# Capabilities: What DocuMind Can and Cannot Do
render_section_header(
    title="System Scope & Operating Boundaries",
    subtitle="Clear distinctions between supported capabilities and system limitations.",
)

col_can, col_cannot = st.columns(2)
with col_can:
    with st.container(border=True):
        st.markdown("**:green[✓ What DocuMind AI Does]**")
        st.markdown(
            """
- Extracts clean text and metadata from arbitrary native PDF files.
- Retrieves specific passages relevant to natural language questions.
- Provides verifiable, page-numbered source references for every claim.
- Supports multiple leading LLM providers (OpenAI, Anthropic, Gemini, Groq, Ollama).
- Protects user credentials via strict in-memory ephemeral handling.
- Operates on sample documents without requiring immediate API keys.
"""
        )

with col_cannot:
    with st.container(border=True):
        st.markdown("**:red[✕ What DocuMind AI Does Not Do]**")
        st.markdown(
            """
- Does not generate ungrounded answers without source text backing.
- Does not store your API credentials, keys, or tokens to disk.
- Does not send your documents for AI model training.
- Does not execute embedded JavaScript or malicious links inside PDFs.
- Does not fabricate retrieval accuracy benchmarks or metrics.
- Does not require paid third-party hosting or mandatory paid APIs.
"""
        )

# Understanding Retrieval-Augmented Generation (RAG) & Evidence Citations
render_section_header(
    title="Evidence-Backed Architecture",
    subtitle="Eliminating hallucination through verified source attribution.",
)

rag_col1, rag_col2 = st.columns(2)
with rag_col1:
    with st.container(border=True):
        st.markdown("**:violet[Why Retrieval-Augmented Generation (RAG)?]**")
        st.caption(
            "Standard LLMs are trained on static historical data and can hallucinate plausible-sounding falsehoods. "
            "RAG solves this by providing the model with real-time extracted excerpts from your specific documents "
            "as authoritative grounding context at inference time."
        )

with rag_col2:
    with st.container(border=True):
        st.markdown("**:blue[Page-Level Source Attributions]**")
        st.caption(
            "Every synthesized answer in DocuMind AI links directly to verbatim document excerpts, "
            "specifying the exact page number and text segment. This allows human verification "
            "of legal, financial, and technical statements with zero ambiguity."
        )

# Security & Architecture Snapshot
render_section_header(
    title="Security & Technology Summary",
    subtitle="Engineered for privacy, reliability, and deployment neutrality.",
)

s_col1, s_col2, s_col3 = st.columns(3)
with s_col1:
    render_metric_card(
        title="Privacy Model",
        value="Ephemeral",
        subtitle="Keys & documents held in session memory only",
        badge="Zero Disk Keys",
    )
with s_col2:
    render_metric_card(
        title="PDF Parser",
        value="PyMuPDF",
        subtitle="High-fidelity C-backed text extraction",
        badge="Native Engine",
    )
with s_col3:
    render_metric_card(
        title="LLM Adapters",
        value="Multi-Provider",
        subtitle="OpenAI, Anthropic, Gemini, Groq, Ollama",
        badge="Configurable",
    )

# Next Button Navigation to Workspace
render_next_page_button(
    label="Proceed to Workspace",
    target_page="pages/workspace.py",
    icon="🔬",
)

# Footer
render_footer()
