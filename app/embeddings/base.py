"""Abstract base class and validation contracts for embedding providers."""

from abc import ABC, abstractmethod
from typing import List
import numpy as np

from app.core.exceptions import ValidationError


class BaseEmbeddingProvider(ABC):
    """Abstract interface for dense vector embedding generation."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Generates dense vector embeddings for a list of document chunk texts.

        Returns:
            np.ndarray of shape (len(texts), dimension), normalized to unit length.
        """
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Alias for embed_documents for API convenience."""
        return self.embed_documents(texts)

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Generates a dense vector embedding for a single user query.

        Returns:
            np.ndarray of shape (1, dimension), normalized to unit length.
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the fixed vector dimension of the embedding space."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the identifier of the active embedding model."""
        pass

    def validate_vectors(self, vectors: np.ndarray, expected_count: int) -> None:
        """Validates numerical finiteness, dimensionality, and shape of generated vectors."""
        if not isinstance(vectors, np.ndarray):
            raise ValidationError("Embeddings output must be a numpy ndarray.")

        if vectors.ndim != 2:
            raise ValidationError(f"Embeddings must have 2 dimensions (samples, dim), got {vectors.ndim}.")

        if vectors.shape[0] != expected_count:
            raise ValidationError(
                f"Embedding count mismatch: expected {expected_count}, got {vectors.shape[0]}."
            )

        if vectors.shape[1] != self.dimension:
            raise ValidationError(
                f"Vector dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}."
            )

        if not np.all(np.isfinite(vectors)):
            raise ValidationError("Embeddings contain non-finite values (NaN or Inf).")


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2-normalizes vectors so inner product corresponds directly to cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0  # Avoid division by zero
    return (vectors / norms).astype(np.float32)
