"""Vector and BM25 index management, document indexing, and removal services."""

from typing import List
from app.embeddings.base import BaseEmbeddingProvider
from app.models.schemas import ProcessedDocument
from app.retrieval.keyword_search import BM25KeywordSearch
from app.retrieval.vector_store import FaissVectorStore


def index_processed_document(
    doc: ProcessedDocument,
    vector_store: FaissVectorStore,
    keyword_search: BM25KeywordSearch,
    embedding_provider: BaseEmbeddingProvider,
) -> int:
    """Embeds document chunks and inserts them into both vector store and BM25 index."""
    if not doc.chunks:
        return 0

    chunk_texts = [c.text for c in doc.chunks]

    # Generate dense embeddings
    vectors = embedding_provider.embed_documents(chunk_texts)

    # Insert into FAISS vector store
    vector_store.add_chunks(doc.chunks, vectors)

    # Insert into BM25 index
    keyword_search.add_chunks(doc.chunks)

    return len(doc.chunks)


def remove_document_from_indices(
    doc_id: str,
    vector_store: FaissVectorStore,
    keyword_search: BM25KeywordSearch,
) -> int:
    """Completely purges a document's chunks from both vector store and BM25 index."""
    v_purged = vector_store.delete_document(doc_id)
    k_purged = keyword_search.delete_document(doc_id)
    return max(v_purged, k_purged)


def rebuild_indices_from_documents(
    documents: List[ProcessedDocument],
    vector_store: FaissVectorStore,
    keyword_search: BM25KeywordSearch,
    embedding_provider: BaseEmbeddingProvider,
) -> int:
    """Wipes active indexes and completely re-indexes all current session documents."""
    vector_store.clear()
    keyword_search.clear()

    total_indexed = 0
    for doc in documents:
        count = index_processed_document(doc, vector_store, keyword_search, embedding_provider)
        total_indexed += count

    return total_indexed
