"""Session cleanup, memory isolation, and index purging service."""

from typing import Any, Dict
from app.security.secrets_policy import purge_session_credentials


def reset_user_workspace(session: Dict[str, Any]) -> None:
    """Safely resets the active workspace without persisting artifacts or cross-session leaks."""
    # 1. Clear vector store & BM25 index if present
    if "vector_store" in session and session["vector_store"] is not None:
        try:
            session["vector_store"].clear()
        except Exception:
            pass
        session["vector_store"] = None

    if "keyword_search" in session and session["keyword_search"] is not None:
        try:
            session["keyword_search"].clear()
        except Exception:
            pass
        session["keyword_search"] = None

    # 2. Reset session document registries
    session["uploaded_documents"] = []
    session["processed_documents"] = {}
    session["active_document_id"] = None
    session["chat_history"] = []
    session["active_analysis"] = None

    # 3. Purge session credentials
    purge_session_credentials(session)
