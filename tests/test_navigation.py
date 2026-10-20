"""Unit tests for modern multi-page navigation routing and page availability."""

from pathlib import Path
from app.core.constants import ROOT_DIR
from app.ui.navigation import PAGE_REGISTRY, get_navigation_pages


def test_page_registry_structure():
    """Verifies that PAGE_REGISTRY defines all 7 pages with required metadata."""
    assert len(PAGE_REGISTRY) == 7

    sections = {p["section"] for p in PAGE_REGISTRY}
    assert "Document Intelligence" in sections
    assert "System & Intelligence" in sections

    titles = [p["title"] for p in PAGE_REGISTRY]
    expected_titles = [
        "Overview",
        "Workspace",
        "Documents",
        "Analysis",
        "Model Settings",
        "Evaluation",
        "About",
    ]
    for expected in expected_titles:
        assert expected in titles

    # Verify single default page
    defaults = [p for p in PAGE_REGISTRY if p.get("default")]
    assert len(defaults) == 1
    assert defaults[0]["title"] == "Overview"


def test_page_files_exist_on_disk():
    """Verifies that all registered page files physically exist in the workspace."""
    for entry in PAGE_REGISTRY:
        file_rel = entry["file"]
        script_path = ROOT_DIR / file_rel
        assert script_path.is_file(), f"Page script '{script_path}' was registered but not found on disk."


def test_get_navigation_pages():
    """Verifies that get_navigation_pages returns a valid section mapping."""
    nav = get_navigation_pages()
    assert isinstance(nav, dict)
    assert "Document Intelligence" in nav
    assert "System & Intelligence" in nav
    assert len(nav["Document Intelligence"]) == 4
    assert len(nav["System & Intelligence"]) == 3
