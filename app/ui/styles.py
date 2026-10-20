"""Custom CSS design system for DocuMind AI.

Implements the Graphite/Slate/Violet/Cyan palette with premium readability.
"""

import streamlit as st


def inject_custom_styles() -> None:
    """Injects carefully scoped CSS styling into the Streamlit document head."""
    custom_css = """
    <style>
    /* -------------------------------------------------------------------------
       DocuMind AI Design Tokens & Global Surfaces
       Palette: Off-White (#F8FAFC), Clean White (#FFFFFF), Enterprise Violet (#6D28D9)
       ------------------------------------------------------------------------- */

    :root {
        --dm-bg-main: #F8FAFC;
        --dm-bg-surface: #FFFFFF;
        --dm-bg-card: #FFFFFF;
        --dm-bg-card-hover: #F1F5F9;
        --dm-accent-violet: #6D28D9;
        --dm-accent-violet-light: #7C3AED;
        --dm-accent-cyan: #0284C7;
        --dm-accent-cyan-light: #0369A1;
        --dm-text-primary: #0F172A;
        --dm-text-secondary: #334155;
        --dm-text-muted: #64748B;
        --dm-border: #E2E8F0;
        --dm-border-highlight: #C4B5FD;
        --dm-success: #059669;
        --dm-warning: #D97706;
        --dm-error: #DC2626;
        --dm-shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --dm-shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    }

    /* Overall Streamlit App background and typography refinement */
    .stApp {
        background-color: var(--dm-bg-main);
        color: var(--dm-text-primary);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Primary and Secondary Headers */
    h1, h2, h3, h4, h5, h6 {
        color: var(--dm-text-primary) !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }

    /* Scoped Brand Header styling */
    .dm-header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.85rem 1.35rem;
        background: #FFFFFF;
        border: 1px solid var(--dm-border);
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: var(--dm-shadow-sm);
    }

    .dm-brand-identity {
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .dm-brand-logo-img {
        width: 44px;
        height: 44px;
        border-radius: 8px;
        object-fit: cover;
        border: 1px solid var(--dm-border);
    }

    .dm-brand-typographic {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 44px;
        height: 44px;
        background: linear-gradient(135deg, var(--dm-accent-violet), var(--dm-accent-cyan));
        color: #FFFFFF;
        font-weight: 700;
        font-size: 1.25rem;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(109, 40, 217, 0.25);
    }

    .dm-brand-titles {
        display: flex;
        flex-direction: column;
    }

    .dm-brand-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--dm-text-primary);
        letter-spacing: -0.01em;
        line-height: 1.2;
    }

    .dm-brand-subtitle {
        font-size: 0.82rem;
        color: var(--dm-text-muted);
        line-height: 1.2;
        margin-top: 2px;
    }

    .dm-header-meta {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    /* Badges */
    .dm-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        letter-spacing: 0.02em;
        border: 1px solid transparent;
    }

    .dm-badge-violet {
        background-color: #EDE9FE;
        color: #6D28D9;
        border-color: #DDD6FE;
    }

    .dm-badge-cyan {
        background-color: #E0F2FE;
        color: #0369A1;
        border-color: #BAE6FD;
    }

    .dm-badge-success {
        background-color: #ECFDF5;
        color: #047857;
        border-color: #A7F3D0;
    }

    .dm-badge-warning {
        background-color: #FFFBEB;
        color: #B45309;
        border-color: #FDE68A;
    }

    /* Scoped Metric and Feature Cards */
    .dm-card {
        background-color: var(--dm-bg-card);
        border: 1px solid var(--dm-border);
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: var(--dm-shadow-sm);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .dm-card:hover {
        border-color: var(--dm-border-highlight);
        box-shadow: var(--dm-shadow-md);
    }

    .dm-card-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--dm-text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }

    .dm-card-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--dm-text-primary);
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }

    .dm-card-subtitle {
        font-size: 0.8rem;
        color: var(--dm-text-muted);
    }

    /* Workflow Pipeline Steps */
    .dm-pipeline-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.85rem;
        margin: 1.25rem 0;
    }

    .dm-pipeline-step {
        background: #FFFFFF;
        border: 1px solid var(--dm-border);
        border-radius: 8px;
        padding: 1rem;
        position: relative;
        box-shadow: var(--dm-shadow-sm);
    }

    .dm-pipeline-step.active {
        border-color: var(--dm-accent-violet);
        background: #FAF5FF;
    }

    .dm-step-number {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 700;
        color: var(--dm-accent-cyan);
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .dm-step-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--dm-text-primary);
        margin-bottom: 0.35rem;
    }

    .dm-step-desc {
        font-size: 0.78rem;
        color: var(--dm-text-muted);
        line-height: 1.35;
    }

    /* Empty state styling */
    .dm-empty-state {
        text-align: center;
        padding: 2.5rem 1.5rem;
        background: #FFFFFF;
        border: 1px dashed var(--dm-border);
        border-radius: 12px;
        margin: 1.25rem 0;
        box-shadow: var(--dm-shadow-sm);
    }

    .dm-empty-icon {
        font-size: 2.2rem;
        margin-bottom: 0.75rem;
        opacity: 0.85;
    }

    .dm-empty-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--dm-text-primary);
        margin-bottom: 0.4rem;
    }

    .dm-empty-desc {
        font-size: 0.85rem;
        color: var(--dm-text-muted);
        max-width: 520px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* Clean Footer */
    .dm-footer {
        margin-top: 3.5rem;
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        border-top: 1px solid var(--dm-border);
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.6rem;
        color: var(--dm-text-muted);
        font-size: 0.82rem;
        text-align: center;
    }

    .dm-footer-socials {
        display: flex;
        gap: 1.25rem;
        margin-top: 0.3rem;
    }

    .dm-footer-link {
        color: var(--dm-accent-violet);
        text-decoration: none;
        transition: color 0.15s ease;
    }

    .dm-footer-link:hover {
        color: #4C1D95;
        text-decoration: underline;
    }

    /* Sidebar polish */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid var(--dm-border);
    }

    /* Clean Streamlit Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.15s ease;
    }

    .stButton > button[kind="primary"] {
        background-color: var(--dm-accent-violet) !important;
        border-color: var(--dm-accent-violet) !important;
        color: #FFFFFF !important;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #5B21B6 !important;
        border-color: #5B21B6 !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
