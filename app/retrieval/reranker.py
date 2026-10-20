"""Passage re-ranking and context candidate deduplication."""

from typing import List
from app.models.schemas import RetrievalResult


def rerank_passages(
    results: List[RetrievalResult],
    query: str,
    top_k: int = 5,
) -> List[RetrievalResult]:
    """Applies candidate deduplication and query-term coverage boost."""
    if not results:
        return []

    query_terms = set(query.lower().split())

    scored_candidates = []
    seen_texts = set()

    for res in results:
        text_snippet = res.chunk.text.strip()[:100]
        if text_snippet in seen_texts:
            continue
        seen_texts.add(text_snippet)

        # Calculate query term coverage in chunk
        chunk_words = set(res.chunk.text.lower().split())
        matched_terms = query_terms.intersection(chunk_words)
        coverage_boost = len(matched_terms) / max(1, len(query_terms)) * 0.1

        adjusted_score = min(1.0, res.score + coverage_boost)
        scored_candidates.append((adjusted_score, res))

    # Sort descending
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    final_results: List[RetrievalResult] = []
    for rank_idx, (adj_score, res) in enumerate(scored_candidates[:top_k]):
        rank = rank_idx + 1
        res.rank = rank
        res.score = round(adj_score, 4)
        res.source_label = f"[Source {rank}]"
        final_results.append(res)

    return final_results
