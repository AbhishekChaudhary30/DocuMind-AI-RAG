"""Page layout scaffolding, session state initialization, and wrappers."""

from typing import Optional
import streamlit as st

from app.core.config import get_settings
from app.core.constants import APP_NAME
from app.embeddings import get_embedding_provider
from app.llm import get_llm_provider
from app.rag.pipeline import RAGPipeline
from app.retrieval.keyword_search import BM25KeywordSearch
from app.retrieval.vector_store import FaissVectorStore
from app.ui.branding import render_brand_header, render_footer
from app.ui.styles import inject_custom_styles


def init_session_state() -> None:
    """Guarantees essential session state variables are initialized."""
    settings = get_settings()

    if "uploaded_documents" not in st.session_state:
        st.session_state.uploaded_documents = []

    if "processed_documents" not in st.session_state:
        st.session_state.processed_documents = {}

    if "active_document_id" not in st.session_state:
        st.session_state.active_document_id = None

    if "selected_provider" not in st.session_state:
        st.session_state.selected_provider = "openai"

    if "model_config" not in st.session_state:
        st.session_state.model_config = {
            "provider": "openai",
            "model_id": "gpt-4o-mini",
            "is_connected": False,
            "custom_endpoint": None,
            "temperature": 0.2,
        }

    if "api_key" not in st.session_state:
        st.session_state.api_key = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Vector store and keyword retrieval index singletons in session
    if "vector_store" not in st.session_state or st.session_state.vector_store is None:
        st.session_state.vector_store = FaissVectorStore(dimension=384)

    if "keyword_search" not in st.session_state or st.session_state.keyword_search is None:
        st.session_state.keyword_search = BM25KeywordSearch()

    if "embedding_mode" not in st.session_state:
        st.session_state.embedding_mode = "local"

    if "retrieval_mode" not in st.session_state:
        st.session_state.retrieval_mode = settings.default_retrieval_mode

    if "retrieval_top_k" not in st.session_state:
        st.session_state.retrieval_top_k = settings.retrieval_top_k

    if "active_analysis" not in st.session_state:
        st.session_state.active_analysis = None

    if "last_retrieved_results" not in st.session_state:
        st.session_state.last_retrieved_results = []

    if "last_query_latency" not in st.session_state:
        st.session_state.last_query_latency = None


def get_active_rag_pipeline() -> RAGPipeline:
    """Constructs or returns the active RAG pipeline configured for the session."""
    init_session_state()

    cfg = st.session_state.model_config
    provider_name = cfg.get("provider", "openai")
    model_id = cfg.get("model_id", "gpt-4o-mini")
    api_key = st.session_state.get("api_key")
    endpoint = cfg.get("custom_endpoint")

    llm = get_llm_provider(
        provider=provider_name,
        model_id=model_id,
        api_key=api_key,
        endpoint=endpoint,
    )

    embed_mode = st.session_state.get("embedding_mode", "local")
    embedder = get_embedding_provider(
        provider_type=embed_mode,
        api_key=api_key if embed_mode in ["openai", "gemini"] else None,
        endpoint=endpoint if embed_mode == "ollama" else None,
    )

    return RAGPipeline(
        vector_store=st.session_state.vector_store,
        keyword_search=st.session_state.keyword_search,
        embedding_provider=embedder,
        llm_provider=llm,
    )


def render_page_header(page_title: str, subtitle: Optional[str] = None) -> None:
    """Renders page title banner and breadcrumb descriptor."""
    sub_html = f'<div style="font-size:0.95rem; color:#94A3B8; margin-top:0.3rem;">{subtitle}</div>' if subtitle else ""
    html = f"""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="font-size: 1.75rem; font-weight: 700; color: #FFFFFF; margin: 0; letter-spacing: -0.02em;">{page_title}</h2>
        {sub_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def page_setup(page_title: str, subtitle: Optional[str] = None) -> None:
    """Unified entry point for every page: injects CSS, sets up state, and renders the brand header."""
    init_session_state()
    inject_custom_styles()
    render_brand_header(page_title)
    render_page_header(page_title, subtitle)
