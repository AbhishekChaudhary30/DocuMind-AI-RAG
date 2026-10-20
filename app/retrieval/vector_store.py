"""FAISS-based vector store with exact cosine search and session isolation."""

from typing import Dict, List, Optional
import numpy as np

from app.core.exceptions import ValidationError
from app.core.logging_config import logger
from app.models.schemas import DocumentChunk, RetrievalResult

try:
    import faiss  # type: ignore
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class FaissVectorStore:
    """In-memory, session-isolated FAISS vector store.

    Uses IndexFlatIP on L2-normalized embeddings for exact cosine similarity.
    Maintains a pure NumPy fallback in case FAISS is unavailable in constrained environments.
    """

    def __init__(self, dimension: int = 384):
        self._dim = dimension
        self._chunks: List[DocumentChunk] = []
        self._vectors: Optional[np.ndarray] = None
        self._index = None
        self._init_index()

    def _init_index(self) -> None:
        if HAS_FAISS:
            self._index = faiss.IndexFlatIP(self._dim)
        else:
            self._index = None

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def count(self) -> int:
        return len(self._chunks)

    def add_chunks(self, chunks: List[DocumentChunk], vectors: np.ndarray) -> None:
        """Adds chunks and their normalized dense vectors to the active index."""
        if not chunks:
            return

        if len(chunks) != vectors.shape[0]:
            raise ValidationError(
                f"Count mismatch: received {len(chunks)} chunks and {vectors.shape[0]} vectors."
            )

        if vectors.shape[1] != self._dim:
            # If incoming vectors have different dimension, re-initialize index for this dimension
            self._dim = vectors.shape[1]
            self._chunks = []
            self._vectors = None
            self._init_index()

        vectors_f32 = vectors.astype(np.float32)

        if HAS_FAISS and self._index is not None:
            self._index.add(vectors_f32)

        # Store in-memory vectors for NumPy fallback and document removal
        if self._vectors is None:
            self._vectors = vectors_f32
        else:
            self._vectors = np.vstack([self._vectors, vectors_f32])

        self._chunks.extend(chunks)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        doc_ids: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        """Searches index for nearest neighbors by cosine similarity."""
        if self.count == 0 or self._vectors is None:
            return []

        q_vec = query_vector.astype(np.float32)
        if q_vec.ndim == 1:
            q_vec = q_vec.reshape(1, -1)

        # Candidate pool size: retrieve more candidates if filtering by doc_ids
        search_k = min(len(self._chunks), top_k * 4 if doc_ids else top_k)

        indices: np.ndarray
        scores: np.ndarray

        if HAS_FAISS and self._index is not None:
            scores_raw, indices_raw = self._index.search(q_vec, search_k)
            scores = scores_raw[0]
            indices = indices_raw[0]
        else:
            # NumPy dot product fallback
            sims = np.dot(self._vectors, q_vec[0])
            sorted_idx = np.argsort(sims)[::-1][:search_k]
            indices = sorted_idx
            scores = sims[sorted_idx]

        results: List[RetrievalResult] = []
        rank = 1
        for idx, score in zip(indices, scores):
            if idx < 0 or idx >= len(self._chunks):
                continue

            chunk = self._chunks[idx]
            if doc_ids and chunk.doc_id not in doc_ids:
                continue

            clamped_score = float(max(0.0, min(1.0, score)))
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=clamped_score,
                    rank=rank,
                    retrieval_mode="dense",
                    dense_score=clamped_score,
                    source_label=f"[Source {rank}]",
                )
            )
            rank += 1
            if len(results) >= top_k:
                break

        return results

    def delete_document(self, doc_id: str) -> int:
        """Removes all vectors belonging to the specified document and rebuilds the index."""
        if self.count == 0 or self._vectors is None:
            return 0

        keep_indices = [i for i, c in enumerate(self._chunks) if c.doc_id != doc_id]
        deleted_count = len(self._chunks) - len(keep_indices)

        if deleted_count == 0:
            return 0

        if not keep_indices:
            self.clear()
            return deleted_count

        self._chunks = [self._chunks[i] for i in keep_indices]
        self._vectors = self._vectors[keep_indices]

        # Rebuild FAISS index
        self._init_index()
        if HAS_FAISS and self._index is not None:
            self._index.add(self._vectors)

        return deleted_count

    def clear(self) -> None:
        """Purges all indexed chunks and vectors from active memory."""
        self._chunks = []
        self._vectors = None
        self._init_index()
