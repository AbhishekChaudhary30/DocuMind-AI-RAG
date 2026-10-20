"""Keyword-based text retrieval using BM25Okapi."""

import re
from typing import List, Optional
from rank_bm25 import BM25Okapi  # type: ignore

from app.models.schemas import DocumentChunk, RetrievalResult


def tokenize_text(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric keywords."""
    if not text:
        return []
    words = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text.lower())
    return words


class BM25KeywordSearch:
    """In-memory, session-isolated BM25Okapi retrieval engine."""

    def __init__(self):
        self._chunks: List[DocumentChunk] = []
        self._tokenized_corpus: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None

    @property
    def count(self) -> int:
        return len(self._chunks)

    def _rebuild_bm25(self) -> None:
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        else:
            self._bm25 = None

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Indexes text chunks into the BM25 model."""
        if not chunks:
            return

        for chunk in chunks:
            tokens = tokenize_text(chunk.text)
            self._tokenized_corpus.append(tokens)
            self._chunks.append(chunk)

        self._rebuild_bm25()

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        """Searches BM25 index for keyword relevance."""
        if not self._bm25 or not self._chunks:
            return []

        tokens = tokenize_text(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        max_score = float(max(scores)) if len(scores) > 0 and max(scores) > 0 else 1.0

        # Sort indices descending
        sorted_indices = scores.argsort()[::-1]

        results: List[RetrievalResult] = []
        rank = 1
        for idx in sorted_indices:
            score = float(scores[idx])
            if score <= 0.0:
                continue

            chunk = self._chunks[idx]
            if doc_ids and chunk.doc_id not in doc_ids:
                continue

            # Normalized BM25 score relative to top match
            norm_score = round(min(1.0, score / max_score), 4)

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=norm_score,
                    rank=rank,
                    retrieval_mode="keyword",
                    keyword_score=score,
                    source_label=f"[Source {rank}]",
                )
            )
            rank += 1
            if len(results) >= top_k:
                break

        return results

    def delete_document(self, doc_id: str) -> int:
        """Removes all indexed chunks for a document and rebuilds BM25 index."""
        keep_indices = [i for i, c in enumerate(self._chunks) if c.doc_id != doc_id]
        deleted_count = len(self._chunks) - len(keep_indices)

        if deleted_count == 0:
            return 0

        self._chunks = [self._chunks[i] for i in keep_indices]
        self._tokenized_corpus = [self._tokenized_corpus[i] for i in keep_indices]
        self._rebuild_bm25()
        return deleted_count

    def clear(self) -> None:
        """Purges BM25 index."""
        self._chunks = []
        self._tokenized_corpus = []
        self._bm25 = None
