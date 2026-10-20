"""Local privacy-preserving dense embedding implementations."""

from typing import Any, Dict, List, Optional
import numpy as np

from app.core.logging_config import logger
from app.embeddings.base import BaseEmbeddingProvider, normalize_vectors

# Global cache for heavy neural embedding models to prevent reloading across Streamlit reruns
_MODEL_CACHE: Dict[str, Any] = {}


class FastLocalEmbedder(BaseEmbeddingProvider):
    """Deterministic, lightweight local embedding engine requiring zero external model downloads.

    Uses character/word n-gram sublinear term frequency with fixed orthogonal random projection
    to produce 384-dimensional normalized vectors. Ideal for offline execution, air-gapped
    deployments, and fast automated test suites.
    """

    def __init__(self, dimension: int = 384, seed: int = 42):
        self._dim = dimension
        self._seed = seed
        self._name = "fast-local-projection-384"
        # Deterministic projection matrix
        rng = np.random.RandomState(self._seed)
        # 4096 sparse hash space projected into target dimension
        self._hash_dim = 4096
        self._projection = rng.randn(self._hash_dim, self._dim).astype(np.float32)
        # Normalize projection columns
        q, _ = np.linalg.qr(self._projection)
        self._projection = q[:, : self._dim]

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._name

    def _hash_text_to_sparse(self, text: str) -> np.ndarray:
        """Hashes words and character 3-grams into a fixed sparse count vector."""
        vec = np.zeros(self._hash_dim, dtype=np.float32)
        if not text:
            return vec

        words = text.lower().split()
        for w in words:
            # Word hash
            h_w = abs(hash(w)) % self._hash_dim
            vec[h_w] += 1.0
            # Character n-grams for subword robustness
            for i in range(len(w) - 2):
                ngram = w[i : i + 3]
                h_ng = abs(hash(ngram)) % self._hash_dim
                vec[h_ng] += 0.5

        # Sublinear term frequency scaling: 1 + log(tf)
        non_zero = vec > 0
        vec[non_zero] = 1.0 + np.log(vec[non_zero])
        return vec

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        dense_rows = []
        for t in texts:
            sparse = self._hash_text_to_sparse(t)
            projected = np.dot(sparse, self._projection)
            dense_rows.append(projected)

        matrix = np.vstack(dense_rows).astype(np.float32)
        normalized = normalize_vectors(matrix)
        self.validate_vectors(normalized, len(texts))
        return normalized

    def embed_query(self, text: str) -> np.ndarray:
        sparse = self._hash_text_to_sparse(text)
        projected = np.dot(sparse, self._projection).reshape(1, -1).astype(np.float32)
        normalized = normalize_vectors(projected)
        self.validate_vectors(normalized, 1)
        return normalized


class SentenceTransformersEmbedding(BaseEmbeddingProvider):
    """Dense neural embedding provider using SentenceTransformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32):
        self._model_name = model_name
        self._batch_size = batch_size
        self._dimension = 384 if "mini" in model_name.lower() else 768
        self._fallback_embedder = FastLocalEmbedder(dimension=self._dimension)
        self._model = None
        self._use_fallback = False

    def _get_model(self):
        """Lazy-loads the model singleton from cache or attempts instantiation."""
        if self._use_fallback:
            return None

        if self._model is not None:
            return self._model

        if self._model_name in _MODEL_CACHE:
            self._model = _MODEL_CACHE[self._model_name]
            return self._model

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            logger.info(f"Loading local SentenceTransformer model '{self._model_name}'...")
            model = SentenceTransformer(self._model_name)
            dim_fn = getattr(model, "get_embedding_dimension", getattr(model, "get_sentence_embedding_dimension", None))
            self._dimension = dim_fn() if dim_fn else (384 if "mini" in self._model_name.lower() else 768)
            _MODEL_CACHE[self._model_name] = model
            self._model = model
            return self._model
        except Exception as exc:
            logger.warning(
                f"Could not load SentenceTransformer '{self._model_name}': {exc}. "
                "Engaging deterministic FastLocalEmbedder fallback to preserve system availability."
            )
            self._use_fallback = True
            return None

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        if self._use_fallback:
            return f"{self._model_name} (FastLocalFallback)"
        return self._model_name

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        model = self._get_model()
        if model is None or self._use_fallback:
            return self._fallback_embedder.embed_documents(texts)

        try:
            embeddings = model.encode(
                texts,
                batch_size=self._batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            dense = embeddings.astype(np.float32)
            self.validate_vectors(dense, len(texts))
            return dense
        except Exception as exc:
            logger.warning(f"SentenceTransformer encoding failed: {exc}. Using fallback.")
            return self._fallback_embedder.embed_documents(texts)

    def embed_query(self, text: str) -> np.ndarray:
        model = self._get_model()
        if model is None or self._use_fallback:
            return self._fallback_embedder.embed_query(text)

        try:
            embedding = model.encode(
                [text],
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            dense = embedding.astype(np.float32)
            self.validate_vectors(dense, 1)
            return dense
        except Exception as exc:
            logger.warning(f"SentenceTransformer query encoding failed: {exc}. Using fallback.")
            return self._fallback_embedder.embed_query(text)
