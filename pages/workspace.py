"""Workspace Page: Interactive document query workbench with real RAG execution and citations."""

import time
import streamlit as st

from app.core.config import get_settings
from app.core.logging_config import logger
from app.models.schemas import DocumentMetadata
from app.services.cleanup_service import reset_user_workspace
from app.services.document_service import process_uploaded_pdf
from app.services.export_service import export_answer_json, export_answer_markdown, sanitize_filename
from app.services.indexing_service import index_processed_document
from app.services.pipeline_tracker import (
    execute_document_ingestion_pipeline,
    render_ingestion_pipeline_stepper,
)
from app.ui.branding import render_footer
from app.ui.components import (
    render_empty_state,
    render_metric_card,
    render_next_page_button,
    render_section_header,
    render_status_badge,
)
from app.ui.layout import get_active_rag_pipeline, page_setup

settings = get_settings()

page_setup(
    page_title="Document Workspace",
    subtitle="Interactive workbench for semantic document search, question answering, and citation evidence.",
)

# Sidebar / Top Workflow Configuration
with st.expander("🛠️ Retrieval & Pipeline Configuration", expanded=False):
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        retrieval_mode = st.selectbox(
            "Retrieval Strategy",
            options=["hybrid", "dense", "keyword"],
            index=["hybrid", "dense", "keyword"].index(st.session_state.retrieval_mode),
            format_func=lambda x: {
                "hybrid": "Hybrid (Dense Vector + BM25 Keywords via RRF)",
                "dense": "Dense Semantic (FAISS Vector Similarity)",
                "keyword": "Sparse Keyword (BM25Okapi Exact Match)",
            }[x],
            help="Select the candidate search algorithm.",
        )
        st.session_state.retrieval_mode = retrieval_mode

    with col_c2:
        top_k = st.slider(
            "Passage Count (Top-K)",
            min_value=1,
            max_value=10,
            value=st.session_state.retrieval_top_k,
            help="Number of relevant chunk passages provided as evidence context.",
        )
        st.session_state.retrieval_top_k = top_k

    with col_c3:
        temperature = st.slider(
            "Generation Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.model_config.get("temperature", 0.2),
            step=0.05,
            help="Lower values yield more factual, deterministic answers.",
        )
        st.session_state.model_config["temperature"] = temperature

# Main Workspace Layout: Ingestion Column (Left) and Q&A Column (Right)
col_doc, col_qa = st.columns([1, 2], gap="large")

with col_doc:
    st.markdown("### 📄 Document Ingestion")

    uploaded_files = st.file_uploader(
        "Upload PDF Document(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Accepts standard PDF files up to {settings.max_file_size_mb} MB.",
        key="workspace_pdf_uploader_multi",
    )

    if uploaded_files:
        pipeline = get_active_rag_pipeline()
        for uf in uploaded_files:
            file_bytes = uf.getvalue()
            already_processed = any(d.filename == uf.name for d in st.session_state.uploaded_documents)
            if not already_processed:
                prog_bar = st.progress(0.0)
                status_box = st.empty()

                def update_prog(cur, total, msg):
                    prog_bar.progress(cur / total)
                    status_box.caption(f"Ingestion [{cur}/{total}]: {msg}")

                report = execute_document_ingestion_pipeline(
                    filename=uf.name,
                    file_bytes=file_bytes,
                    vector_store=st.session_state.vector_store,
                    keyword_search=st.session_state.keyword_search,
                    embedding_provider=pipeline.embedding_provider,
                    settings=settings,
                    existing_documents=st.session_state.processed_documents,
                    progress_callback=update_prog,
                )
                prog_bar.empty()
                status_box.empty()

                if "ingestion_reports" not in st.session_state:
                    st.session_state.ingestion_reports = {}
                st.session_state.ingestion_reports[report.filename] = report

                if report.is_ready and report.processed_doc:
                    pdoc = report.processed_doc
                    st.session_state.processed_documents[pdoc.doc_id] = pdoc
                    meta = DocumentMetadata(
                        doc_id=pdoc.doc_id,
                        filename=pdoc.filename,
                        size_bytes=pdoc.size_bytes,
                        page_count=pdoc.page_count,
                        sha256_hash=pdoc.content_hash,
                        processing_status="indexed",
                        chunk_count=len(pdoc.chunks),
                        is_indexed=True,
                        warnings=report.warnings,
                    )
                    st.session_state.uploaded_documents.append(meta)
                    st.session_state.active_document_id = pdoc.doc_id

                render_ingestion_pipeline_stepper(report)

    # Active Document Selection Filter
    if st.session_state.uploaded_documents:
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown("#### Retrieval Corpus Scope")

        doc_options = {"ALL": "📚 Entire Active Corpus (All Documents)"}
        for d in st.session_state.uploaded_documents:
            doc_options[d.doc_id] = f"📄 {d.filename} ({d.chunk_count} chunks)"

        selected_scope = st.selectbox(
            "Query target:",
            options=list(doc_options.keys()),
            format_func=lambda x: doc_options[x],
            key="workspace_doc_scope_select",
        )
        st.session_state.active_document_id = None if selected_scope == "ALL" else selected_scope

        # Ingested Corpus Summary Pill
        total_chunks = st.session_state.vector_store.count
        with st.container(border=True):
            st.caption("SEARCHABLE VECTOR INDEX")
            st.markdown(f"### {total_chunks} Chunks Indexed")
            st.caption(f"Mode: **{st.session_state.retrieval_mode.upper()}** · Engine: **FAISS + BM25**")
    else:
        st.info("ℹ️ No documents indexed yet. Upload a PDF above or load a sample from Documents.")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Browse Samples 📁", use_container_width=True):
            st.switch_page("pages/documents.py")
    with col_btn2:
        if st.button("Reset Workspace 🗑️", use_container_width=True, type="secondary"):
            reset_user_workspace(st.session_state)
            st.success("Workspace reset.")
            st.rerun()

with col_qa:
    st.markdown("### 🔍 Evidence-Grounded Question Answering")

    # Inform user of active model configuration
    cfg = st.session_state.model_config
    has_api_key = bool(st.session_state.get("api_key"))
    prov = cfg.get("provider", "openai")

    if not has_api_key and prov != "ollama":
        st.info(
            "💡 **Local Extractive RAG Mode Active**: No remote API key is configured. "
            "DocuMind AI is using its deterministic local extractive engine to retrieve passages "
            "and cite authentic source pages at zero cost. Configure a cloud provider in **Model Settings** "
            "to enable generative LLM synthesis."
        )

    # Question Input
    query_input = st.text_input(
        "Ask a question about your document(s):",
        placeholder="e.g., What are the key architectural constraints or revenue figures?",
        key="workspace_query_input_active",
    )

    btn_ask_col, btn_clear_col = st.columns([3, 1])
    with btn_ask_col:
        ask_btn = st.button("Generate Answer with Evidence ⚡", type="primary", use_container_width=True)
    with btn_clear_col:
        if st.button("Clear Chat ✕", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # RAG Execution
    if ask_btn:
        if not st.session_state.uploaded_documents or st.session_state.vector_store.count == 0:
            st.warning("⚠️ Please upload and index a PDF document before asking questions.")
        elif not query_input.strip():
            st.warning("⚠️ Please enter a question.")
        else:
            pipeline = get_active_rag_pipeline()
            target_docs = None
            if st.session_state.active_document_id:
                target_docs = [st.session_state.active_document_id]

            with st.spinner("Searching document index & synthesizing evidence-backed answer..."):
                t0 = time.time()
                # Assemble brief recent conversation history for follow-ups
                conv_history = ""
                for msg in st.session_state.chat_history[-2:]:
                    conv_history += f"Q: {msg['question']}\nA: {msg['answer'][:300]}\n"

                response, retrieved_results = pipeline.answer_query(
                    query=query_input.strip(),
                    top_k=st.session_state.retrieval_top_k,
                    retrieval_mode=st.session_state.retrieval_mode,
                    doc_ids=target_docs,
                    conversation_history=conv_history,
                    context_budget_tokens=settings.context_budget_tokens,
                    temperature=st.session_state.model_config.get("temperature", 0.2),
                )
                latency = round(time.time() - t0, 3)
                st.session_state.last_query_latency = latency
                st.session_state.last_retrieved_results = retrieved_results

                # Append to chat history
                st.session_state.chat_history.append(
                    {
                        "question": query_input.strip(),
                        "answer": response.content,
                        "citations": response.citations,
                        "retrieved": retrieved_results,
                        "latency": latency,
                        "provider": response.provider,
                        "model": response.model_name,
                        "is_sufficient": response.is_evidence_sufficient,
                    }
                )

    # Render Current / Recent Answers
    if st.session_state.chat_history:
        for idx, entry in enumerate(reversed(st.session_state.chat_history)):
            is_latest = idx == 0
            with st.container(border=True):
                c_q, c_meta = st.columns([3, 1.2])
                with c_q:
                    st.markdown(f"#### ❓ {entry['question']}")
                with c_meta:
                    st.caption(f"**{entry['provider']}** · `{entry['model']}` · **{entry['latency']}s**")

                st.markdown(entry["answer"])

                # Verified Citations Display
                citations = entry.get("citations", [])
                if citations:
                    st.markdown("##### 📌 Verified Evidence Citations")
                    for cit in citations:
                        with st.container(border=True):
                            c_src, c_stat = st.columns([3, 1])
                            with c_src:
                                label = cit.source_label or "[Source]"
                                pages = cit.page_range or f"Page {cit.page_number}"
                                st.markdown(f"**{label}** — {cit.filename} ({pages})")
                            with c_stat:
                                if cit.is_valid:
                                    st.success("✓ Verified Passage")
                                else:
                                    st.warning("⚠ Unverified")
                            st.caption(f'"{cit.snippet}"')

                # Inspect Retrieved Passages
                retrieved = entry.get("retrieved", [])
                if retrieved:
                    with st.expander(f"🔎 Inspect Retrieved Passages ({len(retrieved)} Candidates)"):
                        for r in retrieved:
                            st.markdown(
                                f"**Rank {r.rank}** (`{r.source_label}`) · **{r.chunk.filename}** (Pages {r.chunk.start_page}-{r.chunk.end_page}) · "
                                f"Score: `{r.score}` · Engine: `{r.retrieval_mode}`"
                            )
                            snippet = r.chunk.text[:280] + "..." if len(r.chunk.text) > 280 else r.chunk.text
                            st.caption(snippet)
                            st.divider()

            # Export Q&A Evidence Report Buttons
            col_exp_md, col_exp_json = st.columns(2)
            clean_q = sanitize_filename(entry["question"])
            with col_exp_md:
                md_content = export_answer_markdown(
                    question=entry["question"],
                    answer=entry["answer"],
                    citations=citations,
                    retrieved_results=entry.get("retrieved", []),
                    provider=entry["provider"],
                    model=entry["model"],
                    retrieval_mode=st.session_state.retrieval_mode,
                    latency_seconds=entry["latency"],
                )
                st.download_button(
                    "Export Answer (Markdown) 📝",
                    data=md_content,
                    file_name=f"{clean_q}_evidence.md",
                    mime="text/markdown",
                    key=f"ws_exp_md_{idx}",
                    use_container_width=True,
                )
            with col_exp_json:
                json_content = export_answer_json(
                    question=entry["question"],
                    answer=entry["answer"],
                    citations=citations,
                    retrieved_results=entry.get("retrieved", []),
                    provider=entry["provider"],
                    model=entry["model"],
                    retrieval_mode=st.session_state.retrieval_mode,
                    latency_seconds=entry["latency"],
                )
                st.download_button(
                    "Export Answer (JSON) 📦",
                    data=json_content,
                    file_name=f"{clean_q}_evidence.json",
                    mime="application/json",
                    key=f"ws_exp_json_{idx}",
                    use_container_width=True,
                )
    else:
        render_empty_state(
            title="Evidence Workbench Ready",
            description="Upload PDF documents on the left or select a sample document to explore questions with verifiable page citations.",
            icon="💬",
        )

# Navigation
render_next_page_button(
    label="Manage Document Inventory",
    target_page="pages/documents.py",
    icon="📁",
)

render_footer()
