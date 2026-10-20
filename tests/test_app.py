"""Integration test verifying application initialization via Streamlit AppTest."""

import pytest
from streamlit.testing.v1 import AppTest


def test_streamlit_app_starts_without_exception():
    """Verifies that streamlit_app.py executes cleanly with zero runtime exceptions."""
    at = AppTest.from_file("streamlit_app.py", default_timeout=20)
    at.run()
    assert not at.exception, f"AppTest raised unexpected exception(s): {at.exception}"


def test_app_py_starts_without_exception():
    """Verifies that app.py executes cleanly with zero runtime exceptions."""
    at = AppTest.from_file("app.py", default_timeout=20)
    at.run()
    assert not at.exception, f"AppTest raised unexpected exception(s): {at.exception}"


@pytest.mark.parametrize(
    "page_file",
    [
        "pages/overview.py",
        "pages/workspace.py",
        "pages/documents.py",
        "pages/analysis.py",
        "pages/model_settings.py",
        "pages/evaluation.py",
        "pages/about.py",
    ],
)
def test_all_pages_start_without_exception(page_file):
    """Verifies that each individual page executes cleanly with zero exceptions on initial load."""
    at = AppTest.from_file(page_file, default_timeout=20)
    at.run()
    assert not at.exception, f"AppTest on {page_file} raised unexpected exception(s): {at.exception}"

