"""Constants, system metadata, and default configuration values for DocuMind AI."""

from pathlib import Path
from typing import Dict, List, Any

# Root and directory paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = ROOT_DIR / "assets"
LOGO_DIR = ASSETS_DIR / "logo"
DATA_DIR = ROOT_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
EXPORTS_DIR = DATA_DIR / "exports"

# Brand and Product Identity
APP_NAME = "DocuMind AI"
APP_VERSION = "0.1.0"
APP_TAGLINE = "Understand Your Documents. Find Answers with Evidence."
APP_DESCRIPTION = (
    "DocuMind AI transforms PDF documents into searchable knowledge. "
    "Ask questions, explore important information, and generate evidence-backed "
    "answers with page-level source references."
)

DEVELOPER_NAME = "Abhishek Chaudhary"
DEVELOPER_CREDIT = "Created by Abhishek Chaudhary"
DEVELOPER_FOOTER_TEXT = "Created by Abhishek Chaudhary · Follow & Connect"

# Configurable developer profile links (placeholders per specification)
DEFAULT_SOCIAL_LINKS: Dict[str, str] = {
    "GitHub": "https://github.com/AbhishekChaudhary-dev",
    "LinkedIn": "https://www.linkedin.com/in/abhishek-chaudhary-ai",
}

# Supported file types and resource constraints
SUPPORTED_EXTENSIONS: List[str] = [".pdf"]
DEFAULT_MAX_FILE_SIZE_MB: int = 20
DEFAULT_MAX_FILE_SIZE_BYTES: int = DEFAULT_MAX_FILE_SIZE_MB * 1024 * 1024
DEFAULT_MAX_PAGE_COUNT: int = 100
ALLOWED_MIME_TYPES: List[str] = ["application/pdf"]

# Supported Model Providers Matrix
SUPPORTED_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "display_name": "OpenAI",
        "description": "Commercial cloud provider supporting GPT-4o, GPT-4o-mini, and text-embedding-3.",
        "requires_api_key": True,
        "key_env_var": "OPENAI_API_KEY",
        "key_prefix_hint": "sk-...",
        "allows_custom_endpoint": True,
        "default_endpoint": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "available_models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "embedding_models": ["text-embedding-3-small", "text-embedding-3-large"],
        "supports_streaming": True,
    },
    "anthropic": {
        "display_name": "Anthropic",
        "description": "State-of-the-art reasoning via Claude 3.5 Sonnet and Haiku.",
        "requires_api_key": True,
        "key_env_var": "ANTHROPIC_API_KEY",
        "key_prefix_hint": "sk-ant-...",
        "allows_custom_endpoint": False,
        "default_endpoint": "https://api.anthropic.com/v1",
        "default_model": "claude-3-5-sonnet-20241022",
        "available_models": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
        "embedding_models": [],  # Anthropic does not provide native embeddings; requires paired embedding provider
        "supports_streaming": True,
    },
    "gemini": {
        "display_name": "Google Gemini",
        "description": "Multimodal reasoning and long-context capabilities via Gemini Flash and Pro.",
        "requires_api_key": True,
        "key_env_var": "GEMINI_API_KEY",
        "key_prefix_hint": "AIzaSy...",
        "allows_custom_endpoint": False,
        "default_endpoint": "https://generativelanguage.googleapis.com/v1beta",
        "default_model": "gemini-1.5-flash",
        "available_models": ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
        "embedding_models": ["text-embedding-004"],
        "supports_streaming": True,
    },
    "groq": {
        "display_name": "Groq LPU",
        "description": "Ultra-low-latency open-weights inference (Llama 3.3, Mixtral).",
        "requires_api_key": True,
        "key_env_var": "GROQ_API_KEY",
        "key_prefix_hint": "gsk_...",
        "allows_custom_endpoint": True,
        "default_endpoint": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "available_models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        "embedding_models": [],
        "supports_streaming": True,
    },
    "ollama": {
        "display_name": "Ollama (Local / Self-Hosted)",
        "description": "Private local execution with zero third-party data egress.",
        "requires_api_key": False,
        "key_env_var": "",
        "key_prefix_hint": "Not required",
        "allows_custom_endpoint": True,
        "default_endpoint": "http://localhost:11434",
        "default_model": "llama3.2:latest",
        "available_models": ["llama3.2:latest", "mistral:latest", "qwen2.5:latest", "phi3:latest"],
        "embedding_models": ["nomic-embed-text:latest", "all-minilm:latest"],
        "supports_streaming": True,
    },
}

# Curated Sample Documents Catalog
SAMPLE_DOCUMENTS_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "sample-tech-guide",
        "title": "Cloud-Native Microservices Architecture Guide",
        "filename": "technical_architecture_guide.pdf",
        "rel_path": "data/samples/technical_architecture_guide.pdf",
        "category": "Technical Guide",
        "source": "DocuMind Engineering Specification",
        "license": "Creative Commons Zero (CC0) / Public Domain",
        "intended_purpose": "Validates system topology retrieval, service boundary parsing, and technical vocabulary queries.",
        "description": "Short architectural specification detailing distributed RAG topology, vector retrieval, and verification SLAs.",
    },
    {
        "id": "sample-business-report",
        "title": "Global Tech Market Financial Performance (FY25)",
        "filename": "business_financial_report.pdf",
        "rel_path": "data/samples/business_financial_report.pdf",
        "category": "Business Report",
        "source": "Synthetic Enterprise Report (modeled on public SEC 10-Q)",
        "license": "Creative Commons Zero (CC0) / Public Domain",
        "intended_purpose": "Validates multi-column tabular data extraction, quarterly revenue queries, and executive summary synthesis.",
        "description": "Structured corporate financial report with tabular metrics, operating margins, and executive commentary.",
    },
    {
        "id": "sample-governance-policy",
        "title": "Enterprise AI Security and Acceptable Use Policy",
        "filename": "ai_governance_policy.pdf",
        "rel_path": "data/samples/ai_governance_policy.pdf",
        "category": "Governance & Policy",
        "source": "Enterprise Compliance Template (POL-SEC-2025-04)",
        "license": "Creative Commons Zero (CC0) / Public Domain",
        "intended_purpose": "Validates numbered clause citation, compliance rule verification, and evidence-grounded refusal testing.",
        "description": "Formal corporate policy outlining zero-persistent secret logging, tenant data isolation, and citation mandates.",
    },
]

# Color tokens for consistent design system
THEME_COLORS = {
    "bg_main": "#0F1117",
    "bg_surface": "#1A1D24",
    "bg_surface_card": "#222733",
    "accent_primary": "#7C3AED",
    "accent_cyan": "#06B6D4",
    "text_primary": "#F1F5F9",
    "text_muted": "#94A3B8",
    "border_subtle": "#2E3646",
    "status_success": "#10B981",
    "status_warning": "#F59E0B",
    "status_error": "#EF4444",
}
