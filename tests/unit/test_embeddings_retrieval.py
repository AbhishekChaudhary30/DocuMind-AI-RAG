"""Unit tests for dense embeddings, FAISS vector store, BM25, and Hybrid RRF retrieval."""

import pytest
import numpy as np
from app.embeddings.local_embeddings import FastLocalEmbedder
from app.retrieval.vector_store import FaissVectorStore
from app.retrieval.keyword_search import BM25KeywordSearch
from app.retrieval.hybrid_search import reciprocal_rank_fusion
from app.models.schemas import DocumentChunk


@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            chunk_id="chk-1",
            doc_id="doc-A",
            filename="doc_a.pdf",
            chunk_index=0,
            start_page=1,
            end_page=1,
            page_numbers=[1],
            text="FAISS vector similarity search uses inner product on normalized embeddings.",
            token_count=12,
        ),
        DocumentChunk(
            chunk_id="chk-2",
            doc_id="doc-A",
            filename="doc_a.pdf",
            chunk_index=1,
            start_page=2,
            end_page=2,
            page_numbers=[2],
            text="BM25 is a probabilistic term-frequency saturation ranking model for keyword retrieval.",
            token_count=12,
        ),
        DocumentChunk(
            chunk_id="chk-3",
            doc_id="doc-B",
            filename="doc_b.pdf",
            chunk_index=0,
            start_page=1,
            end_page=1,
            page_numbers=[1],
            text="Reciprocal Rank Fusion fuses ranked lists from multiple search algorithms without score calibration.",
            token_count=14,
        ),
    ]


def test_fast_local_embedder_properties():
    embedder = FastLocalEmbedder(dimension=384)
    assert embedder.dimension == 384
    
    vecs = embedder.embed_texts(["First test sentence", "Second test sentence"])
    assert vecs.shape == (2, 384)
    # Check normalization: L2 norm should be 1.0
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-5)
    
    # Query embedding is identical in shape and normalized
    q_vec = embedder.embed_query("Query test")
    assert q_vec.shape == (1, 384)
    assert abs(np.linalg.norm(q_vec) - 1.0) < 1e-5


def test_faiss_vector_store_add_and_search(sample_chunks):
    embedder = FastLocalEmbedder(dimension=384)
    vector_store = FaissVectorStore(dimension=384)
    
    assert vector_store.count == 0
    
    texts = [c.text for c in sample_chunks]
    embeddings = embedder.embed_texts(texts)
    vector_store.add_chunks(sample_chunks, embeddings)
    
    assert vector_store.count == 3
    
    # Query for FAISS
    q_vec = embedder.embed_query("Tell me about FAISS vector search")
    results = vector_store.search(q_vec, top_k=2)
    
    assert len(results) == 2
    top_result = results[0]
    assert top_result.chunk.doc_id == "doc-A"
    assert "FAISS" in top_result.chunk.text
    assert top_result.score > 0.0


def test_faiss_vector_store_delete_document(sample_chunks):
    embedder = FastLocalEmbedder(dimension=384)
    vector_store = FaissVectorStore(dimension=384)
    
    embeddings = embedder.embed_texts([c.text for c in sample_chunks])
    vector_store.add_chunks(sample_chunks, embeddings)
    assert vector_store.count == 3
    
    # Delete doc-A (which has 2 chunks)
    removed_count = vector_store.delete_document("doc-A")
    assert removed_count == 2
    assert vector_store.count == 1
    
    # Search should now only return doc-B
    q_vec = embedder.embed_query("FAISS")
    results = vector_store.search(q_vec, top_k=5)
    assert len(results) == 1
    assert results[0].chunk.doc_id == "doc-B"


def test_bm25_keyword_search_and_delete(sample_chunks):
    bm25 = BM25KeywordSearch()
    bm25.add_chunks(sample_chunks)
    assert bm25.count == 3
    
    # Search for probabilistic term frequency
    results = bm25.search("probabilistic BM25 model", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk.chunk_id == "chk-2"
    assert results[0].retrieval_mode == "keyword"
    
    # Delete document doc-A
    bm25.delete_document("doc-A")
    assert bm25.count == 1
    
    # Search for BM25 now yields no doc-A results
    post_results = bm25.search("probabilistic BM25", top_k=2)
    for r in post_results:
        assert r.chunk.doc_id != "doc-A"


def test_reciprocal_rank_fusion(sample_chunks):
    embedder = FastLocalEmbedder(dimension=384)
    vector_store = FaissVectorStore(dimension=384)
    bm25 = BM25KeywordSearch()
    
    embeddings = embedder.embed_texts([c.text for c in sample_chunks])
    vector_store.add_chunks(sample_chunks, embeddings)
    bm25.add_chunks(sample_chunks)
    
    q_vec = embedder.embed_query("What is Reciprocal Rank Fusion?")
    dense_res = vector_store.search(q_vec, top_k=3)
    kw_res = bm25.search("Reciprocal Rank Fusion", top_k=3)
    
    fused = reciprocal_rank_fusion(dense_res, kw_res, top_k=3, rrf_k=60)
    assert len(fused) >= 1
    assert fused[0].chunk.chunk_id == "chk-3"
    assert fused[0].retrieval_mode == "hybrid"
    assert fused[0].score > 0.0
