"""Branding, logo rendering, and header/footer components for DocuMind AI."""

import base64
from pathlib import Path
from typing import Optional

import streamlit as st

from app.core.config import get_settings
from app.core.constants import (
    APP_NAME,
    APP_TAGLINE,
    DEVELOPER_CREDIT,
    DEVELOPER_FOOTER_TEXT,
)
from app.ui.components import render_html


def get_brand_logo_base64() -> Optional[str]:
    """Reads the resolved logo file into a base64 encoded data URI string.

    Returns None if no logo file exists or if reading fails, guaranteeing no broken images.
    """
    settings = get_settings()
    logo_path = settings.resolve_logo_path()
    if not logo_path or not logo_path.is_file():
        return None

    try:
        suffix = logo_path.suffix.lower().replace(".", "")
        mime = f"image/{suffix}" if suffix in ["png", "jpeg", "jpg", "svg+xml", "webp"] else "image/png"
        if suffix == "svg":
            mime = "image/svg+xml"

        with open(logo_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return None


def get_brand_logo_or_fallback() -> str:
    """Returns HTML for either the image logo or the clean typographic brand mark."""
    logo_data_uri = get_brand_logo_base64()
    if logo_data_uri:
        return f'<img src="{logo_data_uri}" class="dm-brand-logo-img" alt="DocuMind AI Logo" />'
    return '<div class="dm-brand-typographic">DM</div>'


def render_brand_header(page_title: Optional[str] = None) -> None:
    """Renders the top application brand banner."""
    logo_html = get_brand_logo_or_fallback()

    header_html = f"""
    <div class="dm-header-container">
        <div class="dm-brand-identity">
            {logo_html}
            <div class="dm-brand-titles">
                <div class="dm-brand-title">{APP_NAME}</div>
                <div class="dm-brand-subtitle">{APP_TAGLINE}</div>
            </div>
        </div>
        <div class="dm-header-meta">
            <span class="dm-badge dm-badge-violet">Phase 1 · Foundation</span>
            <span class="dm-badge dm-badge-success">● Ready</span>
        </div>
    </div>
    """
    render_html(header_html)


def render_footer() -> None:
    """Renders the unobtrusive footer area with developer attribution and configurable social links."""
    settings = get_settings()
    social_links = settings.social_links

    links_html = "".join(
        f'<a href="{url}" target="_blank" rel="noopener noreferrer" class="dm-footer-link">{name}</a>'
        for name, url in social_links.items()
    )

    logo_data_uri = get_brand_logo_base64()
    footer_logo_html = (
        f'<img src="{logo_data_uri}" style="width:28px; height:28px; border-radius:6px; margin-bottom:4px;" alt="Developer Logo" />'
        if logo_data_uri
        else ""
    )

    footer_html = f"""
    <div class="dm-footer">
        {footer_logo_html}
        <div><strong>{DEVELOPER_FOOTER_TEXT}</strong></div>
        <div class="dm-footer-socials">
            {links_html}
        </div>
        <div style="font-size: 0.72rem; color: #64748B; margin-top: 4px;">
            DocuMind AI · Document Intelligence & RAG Architecture · Ephemeral Session Security Active
        </div>
    </div>
    """
    render_html(footer_html)
