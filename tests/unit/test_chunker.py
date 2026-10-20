"""Unit tests for the chunking subsystem."""

import pytest
from app.document_processing.chunker import chunk_document_pages, split_into_sentences
from app.models.schemas import DocumentPage


def test_split_into_sentences():
    text = "First sentence here. Second sentence starts here! Third question?"
    sentences = split_into_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "First sentence here."
    assert sentences[1] == "Second sentence starts here!"


def test_chunk_short_text():
    raw = "A brief sentence about AI architectures."
    pages = [
        DocumentPage(
            doc_id="doc1",
            page_number=1,
            raw_text=raw,
            cleaned_text=raw,
            char_count=len(raw),
            estimated_tokens=7,
        )
    ]
    chunks = chunk_document_pages(
        pages,
        filename="test.pdf",
        doc_id="doc1",
        chunk_size_tokens=100,
        chunk_overlap_tokens=20,
    )
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.doc_id == "doc1"
    assert chunk.filename == "test.pdf"
    assert chunk.start_page == 1
    assert chunk.end_page == 1
    assert chunk.chunk_index == 0
    assert "brief sentence" in chunk.text


def test_chunk_long_text_and_overlap():
    long_para = "Sentence number {} of the technical document explaining vector mathematics. ".format
    full_text = "".join(long_para(i) for i in range(1, 100))
    pages = [
        DocumentPage(
            doc_id="doc-long",
            page_number=1,
            raw_text=full_text,
            cleaned_text=full_text,
            char_count=len(full_text),
            estimated_tokens=800,
        )
    ]

    chunks = chunk_document_pages(
        pages,
        filename="long.pdf",
        doc_id="doc-long",
        chunk_size_tokens=150,
        chunk_overlap_tokens=30,
    )
    assert len(chunks) > 1
    # Check that each chunk is non-empty
    for c in chunks:
        assert len(c.text.strip()) > 0
        assert c.doc_id == "doc-long"
        assert c.chunk_index >= 0
    # Sequential indexing
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_spans_multiple_pages():
    t1 = "Page 1 introduction to machine learning pipelines."
    t2 = "Page 2 continuation discussing loss optimization functions."
    t3 = "Page 3 conclusion regarding gradient clipping."
    pages = [
        DocumentPage(doc_id="doc-multi", page_number=1, raw_text=t1, cleaned_text=t1, char_count=len(t1), estimated_tokens=8),
        DocumentPage(doc_id="doc-multi", page_number=2, raw_text=t2, cleaned_text=t2, char_count=len(t2), estimated_tokens=8),
        DocumentPage(doc_id="doc-multi", page_number=3, raw_text=t3, cleaned_text=t3, char_count=len(t3), estimated_tokens=7),
    ]
    chunks = chunk_document_pages(
        pages,
        filename="multi.pdf",
        doc_id="doc-multi",
        chunk_size_tokens=500,
        chunk_overlap_tokens=50,
    )
    assert len(chunks) == 1
    c = chunks[0]
    assert c.start_page == 1
    assert c.end_page == 3
    assert c.page_numbers == [1, 2, 3]
    assert "Page 1" in c.text and "Page 3" in c.text


def test_chunk_no_empty_chunks():
    t3 = "Real content here."
    pages = [
        DocumentPage(doc_id="doc-empty", page_number=1, raw_text="", cleaned_text="", char_count=0, estimated_tokens=0),
        DocumentPage(doc_id="doc-empty", page_number=2, raw_text="   \n   \t", cleaned_text="", char_count=0, estimated_tokens=0),
        DocumentPage(doc_id="doc-empty", page_number=3, raw_text=t3, cleaned_text=t3, char_count=len(t3), estimated_tokens=4),
    ]
    chunks = chunk_document_pages(
        pages,
        filename="empty_test.pdf",
        doc_id="doc-empty",
        chunk_size_tokens=200,
        chunk_overlap_tokens=30,
    )
    assert len(chunks) == 1
    assert chunks[0].start_page == 3
    assert "Real content" in chunks[0].text
