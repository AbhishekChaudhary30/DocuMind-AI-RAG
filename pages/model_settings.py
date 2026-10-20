"""Model Settings Page: Configure LLM providers, model parameters, and ephemeral credentials."""

import streamlit as st

from app.core.constants import SUPPORTED_PROVIDERS
from app.llm import get_llm_provider
from app.llm.gemini_discovery import (
    DEFAULT_FALLBACK_GEMINI_MODELS,
    discover_gemini_models,
)
from app.security.secrets_policy import (
    get_security_disclosures,
    mask_secret,
    purge_session_credentials,
    validate_key_format,
)
from app.ui.branding import render_footer
from app.ui.components import (
    render_metric_card,
    render_next_page_button,
    render_section_header,
)
from app.ui.layout import page_setup

page_setup(
    page_title="Model Provider Configuration",
    subtitle="Manage LLM inference adapters, embedding models, and session-held API credentials.",
)

# Active Provider Status Summary Cards
c_prov, c_model, c_sec = st.columns(3)
cfg = st.session_state.model_config

with c_prov:
    prov_name = SUPPORTED_PROVIDERS.get(cfg.get("provider", "openai"), {}).get("display_name", "OpenAI")
    render_metric_card("Active Provider", prov_name, "Selected inference engine", "Provider")

with c_model:
    render_metric_card("Generation Model", cfg.get("model_id", "gpt-4o-mini"), "Primary reasoning model", "Model")

with c_sec:
    has_key = bool(st.session_state.get("api_key"))
    is_ollama = cfg.get("provider") == "ollama"
    status_text = "Key Active (Masked)" if has_key else ("Local Engine (Key Free)" if is_ollama else "Offline Fallback Active")
    badge_type = "Ready" if (has_key or is_ollama) else "Local Only"
    render_metric_card("Credential Status", status_text, "Held ephemerally in session memory", badge_type)

# Provider Selection Form
render_section_header(
    title="Provider & Model Selection",
    subtitle="Configure connection parameters for commercial cloud or local open-weights engines.",
    badge_text="Adapter Interface",
)

provider_keys = list(SUPPORTED_PROVIDERS.keys())
current_provider = st.session_state.selected_provider
provider_idx = provider_keys.index(current_provider) if current_provider in provider_keys else 0

selected_prov_key = st.selectbox(
    "Select Model Provider",
    options=provider_keys,
    format_func=lambda k: f"{SUPPORTED_PROVIDERS[k]['display_name']} — {SUPPORTED_PROVIDERS[k]['description']}",
    index=provider_idx,
    key="select_provider_dropdown_p2",
)
st.session_state.selected_provider = selected_prov_key
prov_info = SUPPORTED_PROVIDERS[selected_prov_key]

# Available Models for selected provider (with dynamic Gemini discovery)
model_col, temp_col = st.columns([2, 1])
with model_col:
    if selected_prov_key == "gemini":
        discovered_models = st.session_state.get("gemini_discovered_models")
        if discovered_models:
            available_models = [m.model_id for m in discovered_models]
            format_dict = {
                m.model_id: f"{m.model_id} ({m.display_name})" + (" · ⚡ Flash" if m.is_flash else (" · 🧠 Pro" if m.is_pro else ""))
                for m in discovered_models
            }
            format_fn = lambda m: format_dict.get(m, m)
            st.caption(f"✓ {len(discovered_models)} models dynamically discovered via API (Flash models prioritized).")
        else:
            available_models = prov_info["available_models"]
            format_fn = lambda m: m
            st.caption("ℹ️ Standard models shown. Enter your key below and click 'Discover Available Models' for dynamic discovery.")
    else:
        available_models = prov_info["available_models"]
        format_fn = lambda m: m

    current_model = cfg.get("model_id")
    model_idx = available_models.index(current_model) if current_model in available_models else 0
    selected_model = st.selectbox(
        "Select Model Architecture",
        options=available_models,
        index=model_idx,
        format_func=format_fn,
    )

with temp_col:
    temperature = st.slider(
        "Sampling Temperature",
        min_value=0.0,
        max_value=1.0,
        value=cfg.get("temperature", 0.2),
        step=0.05,
        help="Lower values produce more deterministic, factual answers.",
    )

# Conditional Endpoint Configuration
custom_endpoint = None
if prov_info["allows_custom_endpoint"]:
    default_ep = prov_info["default_endpoint"]
    custom_endpoint = st.text_input(
        "API Base Endpoint URL",
        value=cfg.get("custom_endpoint") or default_ep,
        help="Custom base URL for local servers (Ollama) or private OpenAI-compatible gateways.",
    )

# API Key Configuration Area
st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
if prov_info["requires_api_key"]:
    st.markdown(f"#### Authentication: {prov_info['display_name']} API Key")
    st.caption(f"Expected format starts with <code>{prov_info['key_prefix_hint']}</code>. Never saved to disk or logs.")

    current_key = st.session_state.get("api_key", "")
    key_input = st.text_input(
        f"Enter {prov_info['display_name']} API Key",
        type="password",
        value=current_key if current_key else "",
        placeholder=f"Paste {prov_info['key_prefix_hint']} here",
        key="api_key_input_field_p2",
    )

    # Invalidate cached models if credentials change
    if selected_prov_key == "gemini" and st.session_state.get("gemini_last_key"):
        if key_input.strip() and key_input.strip() != st.session_state.get("gemini_last_key"):
            st.session_state["gemini_discovered_models"] = None
            st.session_state["gemini_last_key"] = None

    # Dynamic model discovery button for Gemini
    if selected_prov_key == "gemini":
        c_disc1, c_disc2 = st.columns([1.5, 3])
        with c_disc1:
            if st.button("🔄 Discover Available Models", use_container_width=True):
                target_key = (key_input or current_key).strip()
                if not target_key:
                    st.error("Please enter your Gemini API key first.")
                else:
                    with st.spinner("Querying Google Gemini API for models available to your key..."):
                        models, disc_err = discover_gemini_models(target_key)
                        if disc_err:
                            st.warning(f"⚠️ Model discovery note: {disc_err}")
                        elif models:
                            st.session_state["gemini_discovered_models"] = models
                            st.session_state["gemini_last_key"] = target_key
                            st.success(f"✓ Discovered {len(models)} compatible models! Flash models prioritized.")
                            st.rerun()

    btn_col1, btn_col2, btn_col3 = st.columns([1.2, 1.2, 1])
    with btn_col1:
        if st.button("Apply & Test Connection 🔐", type="primary", use_container_width=True):
            is_valid, err = validate_key_format(selected_prov_key, key_input)
            if not is_valid:
                st.error(f"❌ Key Format Error: {err}")
            else:
                k_clean = key_input.strip()
                # For Gemini: if models not yet discovered for this key, run discovery
                if selected_prov_key == "gemini" and not st.session_state.get("gemini_discovered_models"):
                    with st.spinner("Discovering accessible Gemini models..."):
                        models, _ = discover_gemini_models(k_clean)
                        if models:
                            st.session_state["gemini_discovered_models"] = models
                            st.session_state["gemini_last_key"] = k_clean

                # Store in session state
                st.session_state.api_key = k_clean
                st.session_state.model_config = {
                    "provider": selected_prov_key,
                    "model_id": selected_model,
                    "temperature": temperature,
                    "custom_endpoint": custom_endpoint,
                    "is_connected": False,
                }
                # Instantiate provider and run real test
                test_provider = get_llm_provider(
                    provider=selected_prov_key,
                    model_id=selected_model,
                    api_key=k_clean,
                    endpoint=custom_endpoint,
                )
                with st.spinner(f"Sending minimal test generation request to {prov_info['display_name']} ({selected_model})..."):
                    success, test_msg = test_provider.test_connection()
                    if success:
                        st.session_state.model_config["is_connected"] = True
                        masked = mask_secret(k_clean)
                        st.success(f"✓ {test_msg} Registered key: `{masked}`")
                    else:
                        st.error(f"❌ Connection Test Failed: {test_msg}")
                st.rerun()

    with btn_col2:
        if st.button("Use Offline Extractive Mode 💻", type="secondary", use_container_width=True):
            st.session_state.api_key = None
            st.session_state.model_config = {
                "provider": "offline",
                "model_id": "offline-extractive-fallback",
                "temperature": 0.2,
                "custom_endpoint": None,
                "is_connected": True,
            }
            st.success("✓ Configured local offline extractive engine. No external API key required.")
            st.rerun()

    with btn_col3:
        if st.button("Purge Credentials 🗑️", type="secondary", use_container_width=True):
            purged = purge_session_credentials(st.session_state)
            st.session_state.api_key = None
            st.session_state.model_config["is_connected"] = False
            st.session_state["gemini_discovered_models"] = None
            st.session_state["gemini_last_key"] = None
            st.info(f"Purged {purged} session credential item(s). Active memory cleared.")
            st.rerun()
else:
    st.success(f"✓ {prov_info['display_name']} executes locally on your hardware. No remote API key required.")
    col_ol1, col_ol2 = st.columns(2)
    with col_ol1:
        if st.button("Apply & Test Local Connection 💻", type="primary", use_container_width=True):
            st.session_state.model_config = {
                "provider": selected_prov_key,
                "model_id": selected_model,
                "temperature": temperature,
                "custom_endpoint": custom_endpoint,
                "is_connected": False,
            }
            ollama_llm = get_llm_provider("ollama", selected_model, endpoint=custom_endpoint)
            with st.spinner("Pinging local Ollama endpoint..."):
                ok, msg = ollama_llm.test_connection()
                if ok:
                    st.session_state.model_config["is_connected"] = True
                    st.success(f"✓ {msg}")
                else:
                    st.warning(f"⚠️ {msg}")
            st.rerun()

# Dense Vector Embedding Architecture
render_section_header(
    title="Vector Embedding Configuration",
    subtitle="Configure the dense vector representations for document chunks.",
    badge_text="Embedding Engine",
)

embed_options = {
    "local": "Local SentenceTransformers (all-MiniLM-L6-v2, 384-dim)",
    "fast_local": "Fast Local Projection (Zero-Download Offline, 384-dim)",
    "openai": "OpenAI text-embedding-3-small (Cloud, 1536-dim)",
    "ollama": "Ollama nomic-embed-text (Local Server, 768-dim)",
}

curr_embed_mode = st.session_state.get("embedding_mode", "local")
selected_embed = st.selectbox(
    "Embedding Backend",
    options=list(embed_options.keys()),
    index=list(embed_options.keys()).index(curr_embed_mode) if curr_embed_mode in embed_options else 0,
    format_func=lambda x: embed_options[x],
)
if selected_embed != curr_embed_mode:
    st.session_state.embedding_mode = selected_embed
    st.info("Embedding backend updated. Note: Rebuild indexes in Documents to update existing chunks.")

# Security Disclosures
render_section_header(
    title="Security & Privacy Disclosures",
    subtitle="Mandatory disclosures regarding server-side execution and credential handling.",
    badge_text="Trust & Transparency",
)

disclosures = get_security_disclosures()
d_col1, d_col2 = st.columns(2)
for idx, item in enumerate(disclosures):
    target_col = d_col1 if idx % 2 == 0 else d_col2
    with target_col:
        with st.container(border=True):
            st.markdown(f"**🛡️ {item['title']}**")
            st.caption(item["body"])

# Navigation
render_next_page_button(
    label="View Pipeline Diagnostics in Evaluation",
    target_page="pages/evaluation.py",
    icon="🧪",
)

render_footer()
