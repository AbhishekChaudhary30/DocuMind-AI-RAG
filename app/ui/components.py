"""Reusable UI components adhering to DocuMind AI's design system."""

from typing import Dict, List, Optional
import streamlit as st


def render_section_header(
    title: str,
    subtitle: Optional[str] = None,
    badge_text: Optional[str] = None,
) -> None:
    """Renders a standardized section header with optional descriptor and badge."""
    badge_html = f'<span class="dm-badge dm-badge-cyan">{badge_text}</span>' if badge_text else ""
    subtitle_html = f'<div style="font-size:0.9rem; color:#94A3B8; margin-top:0.25rem;">{subtitle}</div>' if subtitle else ""

    html = f"""
    <div style="margin-top: 1.5rem; margin-bottom: 1rem; border-bottom: 1px solid #2E3646; padding-bottom: 0.5rem;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <h3 style="margin: 0; font-size: 1.25rem; font-weight: 600; color: #F1F5F9;">{title}</h3>
            {badge_html}
        </div>
        {subtitle_html}
    </div>
    """
    render_html(html)


def render_metric_card(
    title: str,
    value: str,
    subtitle: Optional[str] = None,
    badge: Optional[str] = None,
) -> None:
    """Renders a structured KPI/diagnostic card."""
    badge_html = f'<span class="dm-badge dm-badge-violet" style="float: right;">{badge}</span>' if badge else ""
    sub_html = f'<div class="dm-card-subtitle">{subtitle}</div>' if subtitle else ""

    html = f"""
    <div class="dm-card">
        {badge_html}
        <div class="dm-card-title">{title}</div>
        <div class="dm-card-value">{value}</div>
        {sub_html}
    </div>
    """
    render_html(html)


def render_html(html_str: str) -> None:
    """Safely renders HTML without triggering markdown block-level code indentation."""
    clean_html = "\n".join(line.strip() for line in html_str.strip().splitlines() if line.strip())
    if hasattr(st, "html"):
        st.html(clean_html)
    else:
        st.markdown(clean_html, unsafe_allow_html=True)


def render_pipeline_steps(steps: List[Dict[str, str]], active_step_index: int = 0) -> None:
    """Renders the document-to-answer workflow progression pipeline using native Streamlit cards."""
    if not steps:
        return
    cols = st.columns(len(steps))
    for idx, (col, step) in enumerate(zip(cols, steps)):
        with col:
            is_active = idx == active_step_index
            step_num = f"Step 0{idx + 1}"
            with st.container(border=True):
                if is_active:
                    st.markdown(f"<span class='dm-badge dm-badge-violet'>{step_num} · Active</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span class='dm-badge dm-badge-cyan'>{step_num}</span>", unsafe_allow_html=True)
                st.markdown(f"**{step.get('title', '')}**")
                st.caption(step.get("desc", ""))


def render_empty_state(
    title: str,
    description: str,
    icon: str = "📁",
    cta_label: Optional[str] = None,
    cta_target_page: Optional[str] = None,
    key: Optional[str] = None,
) -> None:
    """Renders an honest, informative empty state."""
    html = f"""
    <div class="dm-empty-state">
        <div class="dm-empty-icon">{icon}</div>
        <div class="dm-empty-title">{title}</div>
        <div class="dm-empty-desc">{description}</div>
    </div>
    """
    render_html(html)

    if cta_label and cta_target_page:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(cta_label, key=key or f"cta_{cta_target_page}", use_container_width=True, type="primary"):
                st.switch_page(cta_target_page)


def render_status_badge(text: str, status_type: str = "violet") -> None:
    """Renders a small pill status badge."""
    type_class = f"dm-badge-{status_type}"
    html = f'<span class="dm-badge {type_class}">{text}</span>'
    render_html(html)


def render_next_page_button(
    label: str,
    target_page: str,
    icon: str = "➡️",
    key: Optional[str] = None,
) -> None:
    """Renders a visible Next button at the bottom of a sequential page workflow."""
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    cols = st.columns([2, 1])
    with cols[1]:
        if st.button(f"{label} {icon}", key=key or f"next_to_{target_page}", type="primary", use_container_width=True):
            st.switch_page(target_page)
