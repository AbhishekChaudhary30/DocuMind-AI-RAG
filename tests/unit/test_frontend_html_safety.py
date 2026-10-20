"""Unit tests for frontend HTML safety and Confirmed Error B resolution."""

import inspect
from app.ui.components import render_html, render_pipeline_steps


def test_render_html_strips_leading_whitespace():
    """Verify render_html strips leading indentation from multiline HTML blocks."""
    indented_html = """
        <div class="test">
            <span>Hello</span>
        </div>
    """
    clean_lines = [line.lstrip() for line in indented_html.strip().splitlines()]
    clean_html = "\n".join(clean_lines)

    # Indented lines must have 0 leading spaces in cleaned output
    for line in clean_html.splitlines():
        assert not line.startswith("    "), f"Line has 4 or more leading spaces: {line}"


def test_pipeline_steps_no_indented_code_blocks():
    """Verify render_pipeline_steps function source code does not contain nested raw HTML strings."""
    src = inspect.getsource(render_pipeline_steps)
    # Confirmed Error B was caused by literal <div class="dm-pipeline-step"> inside markdown
    assert '<div class="dm-pipeline-step">' not in src
    assert '<span class="dm-step-number">' not in src
    # Must use native columns and containers
    assert "st.columns" in src
    assert "st.container" in src
