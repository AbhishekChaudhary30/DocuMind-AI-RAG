"""Vector embedding providers and factory methods."""

from typing import Optional
from app.embeddings.base import BaseEmbeddingProvider, normalize_vectors
from app.embeddings.local_embeddings import FastLocalEmbedder, SentenceTransformersEmbedding
from app.embeddings.provider_embeddings import ProviderEmbeddingAdapter


def get_embedding_provider(
    provider_type: str = "local",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> BaseEmbeddingProvider:
    """Factory creating configured embedding provider instances."""
    p_type = (provider_type or "local").lower()

    if p_type in ["local", "sentence_transformers"]:
        m_name = model_name or "all-MiniLM-L6-v2"
        return SentenceTransformersEmbedding(model_name=m_name)
    elif p_type == "fast_local":
        return FastLocalEmbedder()
    elif p_type in ["openai", "gemini", "ollama"]:
        m_name = model_name or ("text-embedding-3-small" if p_type == "openai" else "nomic-embed-text")
        return ProviderEmbeddingAdapter(
            provider=p_type,
            model_name=m_name,
            api_key=api_key,
            endpoint=endpoint,
        )
    else:
        return FastLocalEmbedder()


__all__ = [
    "BaseEmbeddingProvider",
    "normalize_vectors",
    "FastLocalEmbedder",
    "SentenceTransformersEmbedding",
    "ProviderEmbeddingAdapter",
    "get_embedding_provider",
]
