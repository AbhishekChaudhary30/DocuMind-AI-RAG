"""Hybrid retrieval combining dense vector search and BM25 via Reciprocal Rank Fusion (RRF)."""

from typing import Dict, List, Optional
from app.models.schemas import DocumentChunk, RetrievalResult


def reciprocal_rank_fusion(
    dense_results: List[RetrievalResult],
    keyword_results: List[RetrievalResult],
    top_k: int = 5,
    rrf_k: int = 60,
) -> List[RetrievalResult]:
    """Combines dense semantic rankings and sparse keyword rankings using Reciprocal Rank Fusion.

    Formula:
        RRF(d) = sum(1 / (rrf_k + rank(d))) across dense and keyword rank lists.
    """
    fused_scores: Dict[str, float] = {}
    chunk_map: Dict[str, DocumentChunk] = {}
    dense_scores: Dict[str, float] = {}
    keyword_scores: Dict[str, float] = {}

    # 1. Process dense candidates
    for rank_idx, res in enumerate(dense_results):
        cid = res.chunk.chunk_id
        chunk_map[cid] = res.chunk
        dense_scores[cid] = res.score
        rank = rank_idx + 1
        fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

    # 2. Process keyword candidates
    for rank_idx, res in enumerate(keyword_results):
        cid = res.chunk.chunk_id
        chunk_map[cid] = res.chunk
        keyword_scores[cid] = res.score
        rank = rank_idx + 1
        fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

    if not fused_scores:
        return []

    # 3. Sort by fused score descending
    sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    max_fused = sorted_items[0][1] if sorted_items else 1.0

    final_results: List[RetrievalResult] = []
    for rank_idx, (cid, raw_score) in enumerate(sorted_items):
        norm_score = round(min(1.0, raw_score / max_fused), 4)
        rank = rank_idx + 1
        chunk = chunk_map[cid]

        final_results.append(
            RetrievalResult(
                chunk=chunk,
                score=norm_score,
                rank=rank,
                retrieval_mode="hybrid",
                dense_score=dense_scores.get(cid),
                keyword_score=keyword_scores.get(cid),
                source_label=f"[Source {rank}]",
            )
        )

    return final_results
