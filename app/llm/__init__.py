"""LLM inference providers, prompt templates, and adapter factory."""

from typing import Optional
from app.llm.base import BaseLLMProvider
from app.llm.model_registry import get_all_providers, get_provider_capabilities, validate_provider_and_model
from app.llm.prompts import RAG_SYSTEM_PROMPT, SUMMARY_SYSTEM_PROMPT, format_rag_prompt
from app.llm.providers import (
    AnthropicProvider,
    GeminiProvider,
    OfflineExtractiveProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
)


def get_llm_provider(
    provider: str,
    model_id: Optional[str] = None,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> BaseLLMProvider:
    """Factory creating configured LLM provider instances with seamless offline fallback."""
    p = (provider or "openai").lower()
    m = model_id or ("gpt-4o-mini" if p == "openai" else "llama3.2:latest")

    # If provider requires an API key but none is present in session, engage offline engine
    if p in ["openai", "anthropic", "gemini", "groq"] and not api_key:
        return OfflineExtractiveProvider(model_name=f"{p}-offline-extractive")

    if p == "openai":
        return OpenAICompatibleProvider(
            provider_name="openai",
            model_name=m,
            api_key=api_key or "",
            base_url=endpoint or "https://api.openai.com/v1",
        )
    elif p == "groq":
        return OpenAICompatibleProvider(
            provider_name="groq",
            model_name=m,
            api_key=api_key or "",
            base_url=endpoint or "https://api.groq.com/openai/v1",
        )
    elif p == "anthropic":
        return AnthropicProvider(model_name=m, api_key=api_key or "")
    elif p == "gemini":
        return GeminiProvider(model_name=m, api_key=api_key or "")
    elif p == "ollama":
        return OllamaProvider(model_name=m, endpoint=endpoint)
    else:
        return OfflineExtractiveProvider(model_name=f"{p}-offline-extractive")


__all__ = [
    "BaseLLMProvider",
    "get_all_providers",
    "get_provider_capabilities",
    "validate_provider_and_model",
    "RAG_SYSTEM_PROMPT",
    "SUMMARY_SYSTEM_PROMPT",
    "format_rag_prompt",
    "get_llm_provider",
    "OfflineExtractiveProvider",
]
