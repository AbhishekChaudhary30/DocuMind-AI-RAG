"""Model provider registry, capability definitions, and validation logic."""

from typing import Dict, List, Optional
from app.core.constants import SUPPORTED_PROVIDERS
from app.models.schemas import ProviderCapabilities


def get_all_providers() -> List[str]:
    """Returns list of supported provider keys."""
    return list(SUPPORTED_PROVIDERS.keys())


def get_provider_capabilities(provider: str) -> Optional[ProviderCapabilities]:
    """Retrieves validated capability specification for a provider."""
    info = SUPPORTED_PROVIDERS.get(provider.lower())
    if not info:
        return None

    return ProviderCapabilities(
        provider=provider.lower(),
        display_name=info["display_name"],
        description=info["description"],
        requires_api_key=info["requires_api_key"],
        key_env_var=info["key_env_var"],
        key_prefix_hint=info["key_prefix_hint"],
        allows_custom_endpoint=info["allows_custom_endpoint"],
        default_endpoint=info["default_endpoint"],
        default_model=info["default_model"],
        available_models=info["available_models"],
        embedding_models=info["embedding_models"],
        supports_streaming=info.get("supports_streaming", True),
    )


def validate_provider_and_model(provider: str, model_id: str) -> bool:
    """Verifies that the requested model belongs to the supported list or is a valid custom model."""
    caps = get_provider_capabilities(provider)
    if not caps:
        return False

    if model_id in caps.available_models:
        return True

    # For Ollama and Groq, permit custom user-pulled models
    if provider.lower() in ["ollama", "groq"] and len(model_id.strip()) > 1:
        return True

    return False
