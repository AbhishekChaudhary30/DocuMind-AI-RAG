"""About Page: Product specifications, system architecture, runtime diagnostics, and attribution."""

import streamlit as st

from app.core.config import get_settings
from app.core.constants import (
    APP_NAME,
    APP_VERSION,
    APP_TAGLINE,
    APP_DESCRIPTION,
    DEVELOPER_NAME,
    DEVELOPER_CREDIT,
)
from app.services.health_service import get_system_health
from app.ui.branding import render_footer
from app.ui.components import (
    render_metric_card,
    render_section_header,
)
from app.ui.layout import page_setup

settings = get_settings()

page_setup(
    page_title="About DocuMind AI",
    subtitle="Product design, architectural principles, runtime diagnostics, and developer attribution.",
)

# Product Overview Card
with st.container(border=True):
    st.markdown(f"### {APP_NAME} `v{APP_VERSION}`")
    st.markdown(f"**{APP_TAGLINE}**")
    st.write(APP_DESCRIPTION)

# Live System Diagnostics Check
render_section_header(
    title="Live Runtime System Health Diagnostics",
    subtitle="Verified status of runtime dependencies and server environment.",
    badge_text="Live Telemetry",
)

health = get_system_health()
h_status_color = "#10B981" if health.status == "healthy" else "#F59E0B"
h_badge = "Healthy" if health.status == "healthy" else "Degraded"

col_h1, col_h2, col_h3, col_h4 = st.columns(4)
with col_h1:
    render_metric_card("System Status", health.status.capitalize(), f"Overall state: {h_badge}", h_badge)
with col_h2:
    render_metric_card("Python Runtime", f"v{health.python_version}", "Verified ≥ 3.11", "Host")
with col_h3:
    render_metric_card("Vector Engine", "FAISS-CPU Active", "Dense similarity index", "Vector")
with col_h4:
    render_metric_card("Keyword Engine", "BM25Okapi Active", "Sparse lexical retrieval", "Lexical")

# Diagnostic Checks Table
st.markdown("#### Environmental Dependency Audit")
checks_table = [
    {"Subsystem Check": "Python Version ≥ 3.11", "Passed": health.checks.get("python_supported", False)},
    {"Subsystem Check": "Streamlit Modern Routing", "Passed": health.checks.get("streamlit_available", False)},
    {"Subsystem Check": "Pydantic v2 Contract Validation", "Passed": health.checks.get("pydantic_v2", False)},
    {"Subsystem Check": "PyMuPDF Native PDF Parser", "Passed": health.checks.get("pymupdf_available", False)},
    {"Subsystem Check": "FAISS-CPU Vector Indexer", "Passed": health.checks.get("faiss_available", False)},
    {"Subsystem Check": "BM25Okapi Keyword Indexer", "Passed": health.checks.get("bm25_available", False)},
    {"Subsystem Check": "Pandas Data Analytics", "Passed": health.checks.get("pandas_available", False)},
    {"Subsystem Check": "NumPy Numerical Operations", "Passed": health.checks.get("numpy_available", False)},
    {"Subsystem Check": "Pillow Asset Processing", "Passed": health.checks.get("pillow_available", False)},
    {"Subsystem Check": "Ephemeral Storage Writable", "Passed": health.checks.get("temp_storage_writable", False)},
    {"Subsystem Check": "Brand Logo Asset Resolved", "Passed": health.checks.get("logo_resolved", False)},
]
st.dataframe(checks_table, use_container_width=True)

# Implementation Phase Roadmap & Deliverables
render_section_header(
    title="Implementation Phase Roadmap",
    subtitle="Sequential release milestones and development roadmap.",
)

c_p1, c_p2, c_p3 = st.columns(3)
with c_p1:
    with st.container(border=True):
        st.markdown("**Phase 1: Foundation (Completed)**")
        st.caption(
            "• Modular Python architecture\n"
            "• Streamlit modern navigation\n"
            "• Design system & brand identity\n"
            "• Centralized Pydantic settings\n"
            "• Sensitive data redaction logging\n"
            "• PDF metadata & SHA256 integrity\n"
            "• Curated sample documents\n"
            "• Automated test suite (pytest)"
        )

with c_p2:
    with st.container(border=True):
        st.markdown("**Phase 2: RAG Pipeline (Completed)**")
        st.caption(
            "• PyMuPDF 1-based page extraction\n"
            "• Token-aware semantic chunking\n"
            "• FAISS vector indexing (Cosine FlatIP)\n"
            "• BM25Okapi sparse keyword retrieval\n"
            "• Reciprocal Rank Fusion (RRF) search\n"
            "• Multi-provider LLM adapters + local fallback\n"
            "• Verified page-level citations engine\n"
            "• Automated document analysis suite"
        )

with c_p3:
    with st.container(border=True):
        st.markdown("**Phase 3: Evaluation & Polish (Completed)**")
        st.caption(
            "• 8-stage document processing pipeline\n"
            "• Automated RAG benchmark runner\n"
            "• Context relevance & MRR evaluation\n"
            "• Citation precision & recall scoring\n"
            "• Ephemeral credential sanitization\n"
            "• Light enterprise UI redesign\n"
            "• Exportable evidence reports"
        )

# Developer Attribution & Follow Links
render_section_header(
    title="Developer Attribution & Contact",
    subtitle="Engineered by Abhishek Chaudhary.",
    badge_text="Creator",
)

with st.container(border=True):
    st.markdown(f"### {DEVELOPER_CREDIT}")
    st.write(
        "AI Application Engineer & Full-Stack Developer specializing in Retrieval-Augmented Generation, "
        "document intelligence systems, and secure LLM application architecture."
    )
    c_gh, c_li = st.columns(2)
    with c_gh:
        st.link_button("GitHub Profile ↗", settings.social_links.get("GitHub", "https://github.com"), use_container_width=True)
    with c_li:
        st.link_button("LinkedIn Profile ↗", settings.social_links.get("LinkedIn", "https://linkedin.com"), use_container_width=True)

render_footer()
