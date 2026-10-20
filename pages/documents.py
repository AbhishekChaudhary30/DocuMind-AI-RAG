"""Documents Page: Ingested document catalog, real vector index status, and sample loader."""

from pathlib import Path
import streamlit as st

from app.core.config import get_settings
from app.core.constants import ROOT_DIR
from app.models.schemas import DocumentMetadata
from app.services.cleanup_service import reset_user_workspace
from app.services.health_service import get_sample_documents
from app.services.indexing_service import (
    rebuild_indices_from_documents,
    remove_document_from_indices,
)
from app.services.pipeline_tracker import execute_document_ingestion_pipeline
from app.ui.branding import render_footer
from app.ui.components import (
    render_empty_state,
    render_metric_card,
    render_next_page_button,
    render_section_header,
)
from app.ui.layout import get_active_rag_pipeline, page_setup

settings = get_settings()

page_setup(
    page_title="Document Management & Indexing",
    subtitle="Inspect metadata, manage active vector indexes, and load curated sample documents.",
)

pipeline = get_active_rag_pipeline()

# Top Metrics Row (Live Measurements)
m1, m2, m3, m4 = st.columns(4)
with m1:
    render_metric_card(
        title="Session Documents",
        value=str(len(st.session_state.uploaded_documents)),
        subtitle="Loaded in ephemeral memory",
        badge="Active",
    )
with m2:
    total_bytes = sum(d.size_bytes for d in st.session_state.uploaded_documents)
    total_mb = round(total_bytes / (1024 * 1024), 2)
    render_metric_card(
        title="Ingested Volume",
        value=f"{total_mb} MB",
        subtitle=f"Limit: {settings.max_file_size_mb} MB/file",
        badge="In-Memory",
    )
with m3:
    total_pages = sum(d.page_count for d in st.session_state.uploaded_documents if d.page_count)
    render_metric_card(
        title="Extracted Pages",
        value=str(total_pages) if total_pages > 0 else "0",
        subtitle="Parsed via PyMuPDF C-engine",
        badge="Verified",
    )
with m4:
    total_chunks = st.session_state.vector_store.count
    render_metric_card(
        title="Indexed Chunks",
        value=str(total_chunks),
        subtitle="In FAISS & BM25 indexes",
        badge="Searchable",
    )

# Section: Active Document Registry
render_section_header(
    title="Active Document Inventory",
    subtitle="Real-time metadata, cryptographic SHA-256 integrity, and index states.",
    badge_text="Live Corpus",
)

if not st.session_state.uploaded_documents:
    render_empty_state(
        title="No Documents in Active Corpus",
        description="Load a curated sample document below or upload a PDF in the Workspace to begin.",
        icon="📂",
        cta_label="Open Workspace to Ingest",
        cta_target_page="pages/workspace.py",
    )
else:
    for idx, doc in enumerate(st.session_state.uploaded_documents):
        with st.container(border=True):
            c_info, c_actions = st.columns([4, 1.2])
            with c_info:
                sample_tag = " · (Curated Sample)" if doc.is_sample else ""
                pages_str = f"{doc.page_count} pages" if doc.page_count else "Pending"
                st.markdown(f"**📄 {doc.filename}** {sample_tag}")
                st.caption(
                    f"**Volume:** {pages_str} · **Size:** {doc.size_mb} MB ({doc.size_bytes:,} bytes) · "
                    f"**Chunks:** {doc.chunk_count} indexed · **Uploaded:** {doc.upload_timestamp[:19]} UTC"
                )
                st.code(f"SHA-256: {doc.sha256_hash}", language="text")
            with c_actions:
                st.write("")
                if st.button("Remove ✕", key=f"rm_doc_{doc.doc_id}", use_container_width=True):
                    # Remove from vector store and BM25 index
                    remove_document_from_indices(
                        doc_id=doc.doc_id,
                        vector_store=st.session_state.vector_store,
                        keyword_search=st.session_state.keyword_search,
                    )
                    # Remove from session registries
                    st.session_state.uploaded_documents = [
                        d for d in st.session_state.uploaded_documents if d.doc_id != doc.doc_id
                    ]
                    if doc.doc_id in st.session_state.processed_documents:
                        del st.session_state.processed_documents[doc.doc_id]

                    if st.session_state.active_document_id == doc.doc_id:
                        st.session_state.active_document_id = None

                    st.success(f"Removed '{doc.filename}' and purged from search indexes.")
                    st.rerun()

    # Bulk Index Actions
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        if st.button("Rebuild All Indexes 🔄", type="secondary"):
            all_docs = list(st.session_state.processed_documents.values())
            rebuilt_chunks = rebuild_indices_from_documents(
                documents=all_docs,
                vector_store=st.session_state.vector_store,
                keyword_search=st.session_state.keyword_search,
                embedding_provider=pipeline.embedding_provider,
            )
            st.success(f"✓ Rebuilt FAISS & BM25 indexes with {rebuilt_chunks} chunks.")
            st.rerun()
    with b_col2:
        if st.button("Purge All Documents & Reset 🗑️", type="secondary"):
            reset_user_workspace(st.session_state)
            st.success("All session documents and search indexes purged.")
            st.rerun()

# Section: Curated Sample Document Catalog
render_section_header(
    title="Curated Sample Documents Catalog",
    subtitle="Pre-verified test files with zero external setup required.",
    badge_text="Instant Ingestion",
)

sample_docs = get_sample_documents()

for sample, exists in sample_docs:
    with st.container(border=True):
        col_desc, col_btn = st.columns([3, 1.2])
        with col_desc:
            st.markdown(f"**📖 {sample.title}** `[{sample.category}]`")
            st.write(sample.description)
            st.caption(
                f"**Provenance:** {sample.source} · **License:** {sample.license} · **Focus:** {sample.intended_purpose}"
            )
        with col_btn:
            st.write("")
            if exists:
                btn_key = f"load_sample_p2_{sample.id}"
                if st.button("Load & Index 📥", key=btn_key, use_container_width=True):
                    sample_path = ROOT_DIR / sample.rel_path
                    with open(sample_path, "rb") as sf:
                        pdf_bytes = sf.read()

                    with st.spinner(f"Running pipeline for '{sample.title}'..."):
                        ingestion_report = execute_document_ingestion_pipeline(
                            filename=sample.filename,
                            file_bytes=pdf_bytes,
                            vector_store=st.session_state.vector_store,
                            keyword_search=st.session_state.keyword_search,
                            embedding_provider=pipeline.embedding_provider,
                            settings=settings,
                            existing_documents=st.session_state.processed_documents,
                        )

                    if ingestion_report.is_ready and ingestion_report.processed_doc:
                        pdoc = ingestion_report.processed_doc
                        st.session_state.processed_documents[pdoc.doc_id] = pdoc
                        existing_ids = [d.doc_id for d in st.session_state.uploaded_documents]
                        if pdoc.doc_id not in existing_ids:
                            meta = DocumentMetadata(
                                doc_id=pdoc.doc_id,
                                filename=pdoc.filename,
                                size_bytes=pdoc.size_bytes,
                                page_count=pdoc.page_count,
                                sha256_hash=pdoc.content_hash,
                                processing_status="indexed",
                                chunk_count=len(pdoc.chunks),
                                is_indexed=True,
                                is_sample=True,
                                warnings=ingestion_report.warnings,
                            )
                            st.session_state.uploaded_documents.append(meta)
                        if "ingestion_reports" not in st.session_state:
                            st.session_state.ingestion_reports = {}
                        st.session_state.ingestion_reports[ingestion_report.filename] = ingestion_report
                        st.session_state.active_document_id = pdoc.doc_id
                        st.success(f"✓ Successfully indexed '{sample.title}' ({len(pdoc.chunks)} chunks).")
                        st.rerun()
                    else:
                        st.error(f"Pipeline error: {ingestion_report.error}")
            else:
                st.button("Unavailable", disabled=True, key=f"disabled_s_{sample.id}", use_container_width=True)

# Navigation
render_next_page_button(
    label="Proceed to Automatic Analysis",
    target_page="pages/analysis.py",
    icon="📊",
)

render_footer()
