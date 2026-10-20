"""Modern Streamlit page navigation configuration and routing for DocuMind AI."""

from typing import Dict, List
import streamlit as st

# Central Registry of Application Pages
PAGE_REGISTRY: List[Dict[str, any]] = [
    {
        "section": "Document Intelligence",
        "file": "pages/overview.py",
        "title": "Overview",
        "icon": "🎯",
        "default": True,
    },
    {
        "section": "Document Intelligence",
        "file": "pages/workspace.py",
        "title": "Workspace",
        "icon": "🔬",
        "default": False,
    },
    {
        "section": "Document Intelligence",
        "file": "pages/documents.py",
        "title": "Documents",
        "icon": "📁",
        "default": False,
    },
    {
        "section": "Document Intelligence",
        "file": "pages/analysis.py",
        "title": "Analysis",
        "icon": "📊",
        "default": False,
    },
    {
        "section": "System & Intelligence",
        "file": "pages/model_settings.py",
        "title": "Model Settings",
        "icon": "⚙️",
        "default": False,
    },
    {
        "section": "System & Intelligence",
        "file": "pages/evaluation.py",
        "title": "Evaluation",
        "icon": "🧪",
        "default": False,
    },
    {
        "section": "System & Intelligence",
        "file": "pages/about.py",
        "title": "About",
        "icon": "ℹ️",
        "default": False,
    },
]


def get_navigation_pages() -> Dict[str, List[st.Page]]:
    """Builds and returns the structured navigation dictionary of st.Page objects."""
    nav: Dict[str, List[st.Page]] = {}
    for entry in PAGE_REGISTRY:
        section = entry["section"]
        if section not in nav:
            nav[section] = []

        page_obj = st.Page(
            entry["file"],
            title=entry["title"],
            icon=entry["icon"],
            default=entry["default"],
        )
        nav[section].append(page_obj)

    return nav


def setup_navigation() -> st.navigation:
    """Configures and returns the active Streamlit navigation router."""
    nav_structure = get_navigation_pages()
    return st.navigation(nav_structure)


def switch_to_page(page_rel_path: str) -> None:
    """Safely transitions between pages preserving session state."""
    st.switch_page(page_rel_path)
