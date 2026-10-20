"""DocuMind AI — Main Application Entry Point.

Modular PDF Question Answering and Document Intelligence Application.
Created by Abhishek Chaudhary.

Supports standard execution via:
    streamlit run app.py
or
    streamlit run streamlit_app.py
"""

import streamlit as st

from app.core.config import get_settings
from app.ui.layout import init_session_state
from app.ui.navigation import setup_navigation
from app.ui.styles import inject_custom_styles

# Central Streamlit page configuration
settings = get_settings()
st.set_page_config(
    page_title=f"{settings.app_name} · Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize application session and styling
init_session_state()
inject_custom_styles()

# Execute modern multi-page routing
navigation = setup_navigation()
navigation.run()
