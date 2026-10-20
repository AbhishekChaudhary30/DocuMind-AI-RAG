"""Integration tests for the complete end-to-end RAG pipeline."""

import pytest
from app.rag.pipeline import RAGPipeline
from app.retrieval.vector_store import FaissVectorStore
from app.retrieval.keyword_search import BM25KeywordSearch
from app.embeddings.local_embeddings import FastLocalEmbedder
from app.llm.providers import OfflineExtractiveProvider
from app.services.document_service import process_uploaded_pdf
from app.services.indexing_service import index_processed_document, remove_document_from_indices


@pytest.fixture
def rag_system():
    vector_store = FaissVectorStore(dimension=384)
    keyword_search = BM25KeywordSearch()
    embedder = FastLocalEmbedder(dimension=384)
    llm = OfflineExtractiveProvider()
    
    pipeline = RAGPipeline(
        vector_store=vector_store,
        keyword_search=keyword_search,
        embedding_provider=embedder,
        llm_provider=llm,
    )
    return pipeline, vector_store, keyword_search, embedder, llm


def test_e2e_rag_ingest_and_query(rag_system, synthetic_multipage_pdf):
    pipeline, vector_store, keyword_search, embedder, llm = rag_system
    
    # 1. Process PDF document
    doc, err, warnings = process_uploaded_pdf(
        filename="system_spec.pdf",
        file_bytes=synthetic_multipage_pdf,
    )
    assert err is None
    assert doc is not None
    doc_id = doc.doc_id
    
    # 2. Index into FAISS and BM25
    indexed_chunks = index_processed_document(doc, vector_store, keyword_search, embedder)
    assert indexed_chunks >= 1
    assert vector_store.count == indexed_chunks
    assert keyword_search.count == indexed_chunks
    
    # 3. Ask a question present in the document
    response, results = pipeline.answer_query(
        query="What does FAISS and dense semantic embeddings do?",
        top_k=3,
        retrieval_mode="hybrid",
    )
    
    assert len(response.content) > 0
    assert len(results) >= 1
    assert len(response.citations) >= 1
    
    # Verify citation provenance
    top_cit = response.citations[0]
    assert top_cit.filename == "system_spec.pdf"
    assert top_cit.page_number in [1, 2, 3]
    assert top_cit.is_valid is True
    
    # 4. Remove document from indices
    purged = remove_document_from_indices(doc_id, vector_store, keyword_search)
    assert purged == indexed_chunks
    assert vector_store.count == 0
    assert keyword_search.count == 0
    
    # 5. Query after purge returns empty retrieval results
    post_resp, post_results = pipeline.answer_query(
        query="What does FAISS do?",
        top_k=3,
        retrieval_mode="hybrid",
    )
    assert len(post_results) == 0
    assert len(post_resp.citations) == 0


def test_e2e_rag_multi_document_attribution(rag_system, create_test_pdf):
    pipeline, vector_store, keyword_search, embedder, _ = rag_system
    
    pdf_alpha = create_test_pdf(["Alpha Protocol specifies AES-256 encryption at rest."])
    pdf_beta = create_test_pdf(["Beta Protocol mandates TLS 1.3 encryption in transit."])
    
    doc_a, _, _ = process_uploaded_pdf("alpha.pdf", pdf_alpha)
    doc_b, _, _ = process_uploaded_pdf("beta.pdf", pdf_beta)
    
    index_processed_document(doc_a, vector_store, keyword_search, embedder)
    index_processed_document(doc_b, vector_store, keyword_search, embedder)
    
    assert vector_store.count == 2
    
    # Query Alpha
    resp_a, res_a = pipeline.answer_query("What does Alpha Protocol specify?", top_k=2)
    assert len(res_a) >= 1
    assert any(c.filename == "alpha.pdf" for c in resp_a.citations)
    
    # Query Beta
    resp_b, res_b = pipeline.answer_query("What does Beta Protocol mandate?", top_k=2)
    assert len(res_b) >= 1
    assert any(c.filename == "beta.pdf" for c in resp_b.citations)
