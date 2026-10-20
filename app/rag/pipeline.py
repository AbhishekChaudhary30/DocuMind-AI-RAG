"""End-to-end Retrieval-Augmented Generation pipeline."""

import time
from typing import List, Optional, Tuple

from app.embeddings.base import BaseEmbeddingProvider
from app.llm.base import BaseLLMProvider
from app.models.schemas import LLMResponse, RetrievalResult
from app.rag.answer_generator import generate_grounded_answer
from app.retrieval.hybrid_search import reciprocal_rank_fusion
from app.retrieval.keyword_search import BM25KeywordSearch
from app.retrieval.reranker import rerank_passages
from app.retrieval.vector_store import FaissVectorStore


class RAGPipeline:
    """Coordinates semantic embedding, retrieval, context assembly, and answer synthesis."""

    def __init__(
        self,
        vector_store: FaissVectorStore,
        keyword_search: BM25KeywordSearch,
        embedding_provider: BaseEmbeddingProvider,
        llm_provider: BaseLLMProvider,
    ):
        self.vector_store = vector_store
        self.keyword_search = keyword_search
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        retrieval_mode: str = "hybrid",
        doc_ids: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        """Executes retrieval according to selected mode (dense, keyword, or hybrid)."""
        mode = retrieval_mode.lower()

        if mode == "dense":
            q_vec = self.embedding_provider.embed_query(query)
            candidates = self.vector_store.search(q_vec, top_k=top_k, doc_ids=doc_ids)

        elif mode == "keyword":
            candidates = self.keyword_search.search(query, top_k=top_k, doc_ids=doc_ids)

        else:  # hybrid
            # Retrieve candidate pool from both systems
            q_vec = self.embedding_provider.embed_query(query)
            dense_cands = self.vector_store.search(q_vec, top_k=top_k * 2, doc_ids=doc_ids)
            keyword_cands = self.keyword_search.search(query, top_k=top_k * 2, doc_ids=doc_ids)

            if dense_cands and keyword_cands:
                candidates = reciprocal_rank_fusion(dense_cands, keyword_cands, top_k=top_k)
            elif dense_cands:
                candidates = dense_cands[:top_k]
            else:
                candidates = keyword_cands[:top_k]

        # Apply candidate re-ranking and deduplication
        reranked = rerank_passages(candidates, query=query, top_k=top_k)
        return reranked

    def answer_query(
        self,
        query: str,
        top_k: int = 5,
        retrieval_mode: str = "hybrid",
        doc_ids: Optional[List[str]] = None,
        conversation_history: str = "",
        context_budget_tokens: int = 3000,
        temperature: float = 0.2,
    ) -> Tuple[LLMResponse, List[RetrievalResult]]:
        """Executes full RAG workflow from question to evidence-backed response."""
        retrieved_results = self.retrieve(
            query=query,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            doc_ids=doc_ids,
        )

        response, citations = generate_grounded_answer(
            query=query,
            retrieved_results=retrieved_results,
            llm_provider=self.llm_provider,
            conversation_history=conversation_history,
            context_budget_tokens=context_budget_tokens,
            temperature=temperature,
        )

        return response, retrieved_results
