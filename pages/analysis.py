"""Analysis Page: Automated document intelligence, executive summarization, and concept modeling."""

import streamlit as st

from app.analysis.document_insights import analyze_processed_documents
from app.services.export_service import export_analysis_json, export_analysis_markdown
from app.ui.branding import render_footer
from app.ui.components import (
    render_empty_state,
    render_metric_card,
    render_next_page_button,
    render_section_header,
)
from app.ui.layout import get_active_rag_pipeline, page_setup

page_setup(
    page_title="Document Intelligence & Analysis",
    subtitle="Automated executive summarization, key entity modeling, and structural corpus diagnostics.",
)

pipeline = get_active_rag_pipeline()
docs = list(st.session_state.processed_documents.values())

if not docs:
    render_empty_state(
        title="No Ingested Documents to Analyze",
        description="Load a document in Workspace or Documents to trigger the automatic document intelligence suite.",
        icon="📊",
        cta_label="Browse Documents",
        cta_target_page="pages/documents.py",
    )
else:
    # Action Banner to Trigger Real Analysis
    with st.container(border=True):
        st.markdown("### ⚡ Automated Intelligence Engine")
        st.write(
            "Execute deep structural telemetry, hierarchical map-reduce executive summarization, "
            "entity recognition, and intelligent question discovery across all currently ingested documents."
        )

    col_act1, col_act2 = st.columns([2, 1])
    with col_act1:
        analyze_btn = st.button("🚀 Analyze Document Corpus", type="primary", use_container_width=True)
    with col_act2:
        if st.session_state.active_analysis:
            if st.button("Clear Analysis Results ✕", use_container_width=True):
                st.session_state.active_analysis = None
                st.rerun()

    if analyze_btn:
        with st.spinner("Executing structural audit and generating executive intelligence report..."):
            analysis_result = analyze_processed_documents(
                documents=docs,
                llm_provider=pipeline.llm_provider,
            )
            st.session_state.active_analysis = analysis_result
            st.success("✓ Document analysis generated successfully!")

    # Display Analysis Results
    res = st.session_state.active_analysis
    if res:
        # 1. Structural Corpus Statistics Cards
        render_section_header(
            title="Corpus Structural Telemetry",
            subtitle="Verified quantitative measurements from parsed document text.",
            badge_text="Deterministic Telemetry",
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_metric_card("Analyzed Documents", str(res.structural_stats.get("Total Documents", 0)), ", ".join(res.document_titles[:2]), "Corpus")
        with c2:
            render_metric_card("Total Page Volume", str(res.structural_stats.get("Total Pages", 0)), "Across parsed files", "Pages")
        with c3:
            render_metric_card("Semantic Chunks", str(res.structural_stats.get("Total Chunks", 0)), "Partitioned for retrieval", "Index")
        with c4:
            render_metric_card("Word Count", f"{res.structural_stats.get('Total Words', 0):,}", f"{res.structural_stats.get('Vocabulary Size', 0):,} unique terms", "Tokens")

        # 2. Executive Overview Section
        render_section_header(
            title="Executive Summary",
            subtitle="Synthesized overview generated from extracted document sections.",
            badge_text="Executive Digest",
        )
        with st.container(border=True):
            st.markdown(res.summary)

        # 3. Key Topics and Concepts
        render_section_header(
            title="Primary Thematic Concepts",
            subtitle="Prominent domains identified across the corpus.",
            badge_text="Thematic Modeling",
        )
        pills_html = " ".join([f'<span class="dm-badge dm-badge-violet" style="font-size:0.85rem; padding:0.35rem 0.8rem; margin:3px;">{t}</span>' for t in res.key_topics])
        st.markdown(f'<div style="margin-bottom: 1.25rem;">{pills_html}</div>', unsafe_allow_html=True)

        # 4. Key Entities Breakdown
        if res.key_entities:
            render_section_header(
                title="Extracted Domain Entities & Financials",
                subtitle="Categorized organizations, monetary amounts, and technical specifications.",
                badge_text="Named Entities",
            )
            e_cols = st.columns(len(res.key_entities))
            for idx, (cat_name, items) in enumerate(res.key_entities.items()):
                with e_cols[idx]:
                    st.markdown(f"**{cat_name}**")
                    for item in items:
                        st.markdown(f"- `{item}`")

        # 5. Important Facts and Verifiable Statements
        if res.important_facts:
            render_section_header(
                title="Critical Factual Assertions",
                subtitle="High-density quantitative metrics and governance directives.",
                badge_text="Factual Evidence",
            )
            for fact in res.important_facts:
                with st.container(border=True):
                    st.markdown(f"📌 {fact}")

        # 6. Suggested Exploratory Questions
        if res.suggested_questions:
            render_section_header(
                title="Suggested Exploratory Queries",
                subtitle="Recommended questions derived from document evidence.",
                badge_text="Question Discovery",
            )
            for q in res.suggested_questions:
                st.markdown(f"- 💡 *{q}*")

        # 7. Export Controls for Intelligence Report
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        render_section_header(
            title="Export Intelligence Findings",
            subtitle="Download synthesized document intelligence and structural telemetry reports.",
            badge_text="Report Export",
        )
        col_an_md, col_an_json = st.columns(2)
        with col_an_md:
            an_md_str = export_analysis_markdown(res)
            st.download_button(
                "Download Intelligence Report (Markdown) 📝",
                data=an_md_str,
                file_name="documind_intelligence_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col_an_json:
            an_json_str = export_analysis_json(res)
            st.download_button(
                "Download Intelligence Report (JSON) 📦",
                data=an_json_str,
                file_name="documind_intelligence_report.json",
                mime="application/json",
                use_container_width=True,
            )

# Navigation
render_next_page_button(
    label="Configure Model Settings",
    target_page="pages/model_settings.py",
    icon="⚙️",
)

render_footer()
