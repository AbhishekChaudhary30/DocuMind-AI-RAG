"""Corpus insights, factual assertions, and question recommendation engine."""

import re
from typing import Dict, List, Optional
from app.analysis.document_summary import generate_document_summary
from app.analysis.key_topics import extract_key_entities, extract_key_topics
from app.llm.base import BaseLLMProvider
from app.models.schemas import AnalysisResult, ProcessedDocument


def extract_important_facts(text: str, max_facts: int = 5) -> List[str]:
    """Extracts high-information factual statements containing numbers, metrics, or directives."""
    if not text:
        return []

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 30]

    # Score sentences by density of numbers, capitalized terms, and policy words
    scored = []
    for s in sentences:
        score = 0
        if re.search(r"\d+", s):
            score += 2
        if re.search(r"\$|%|revenue|margin|policy|must|shall|architecture", s, re.IGNORECASE):
            score += 3
        if len(s) < 250:
            score += 1
        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    facts = [item[1] for item in scored[:max_facts]]
    return facts


def generate_suggested_questions(topics: List[str], facts: List[str]) -> List[str]:
    """Derives intelligent exploratory questions from verified topics and facts."""
    questions: List[str] = []

    for topic in topics[:3]:
        questions.append(f"What are the primary specifications and requirements regarding {topic}?")
        questions.append(f"How is {topic} implemented or evaluated according to the document?")

    if facts:
        questions.append("What key metrics, SLA targets, or financial figures are reported?")

    # Deduplicate and return top 5
    unique_q = []
    for q in questions:
        if q not in unique_q:
            unique_q.append(q)
    return unique_q[:5]


def analyze_processed_documents(
    documents: List[ProcessedDocument],
    llm_provider: BaseLLMProvider,
) -> AnalysisResult:
    """Executes full automated analysis across single or multi-document corpora."""
    if not documents:
        return AnalysisResult(
            summary="No documents available to analyze.",
            key_topics=[],
            important_facts=[],
            key_entities={},
            suggested_questions=[],
            structural_stats={},
        )

    all_pages = []
    for doc in documents:
        all_pages.extend(doc.pages)

    combined_text = "\n\n".join(p.cleaned_text for p in all_pages)
    doc_titles = [d.filename for d in documents]

    # 1. Structural Statistics
    words = combined_text.split()
    total_words = len(words)
    unique_words = len(set(w.lower() for w in words))
    total_pages = sum(d.page_count for d in documents)
    total_chunks = sum(len(d.chunks) for d in documents)

    stats = {
        "Total Documents": len(documents),
        "Total Pages": total_pages,
        "Total Chunks": total_chunks,
        "Total Words": total_words,
        "Vocabulary Size": unique_words,
        "Avg Words / Page": round(total_words / max(1, total_pages), 1),
    }

    # 2. Executive Summary
    summary = generate_document_summary(all_pages, llm_provider)

    # 3. Topics and Entities
    topics = extract_key_topics(combined_text, top_n=6)
    entities = extract_key_entities(combined_text)

    # 4. Facts and Suggested Questions
    facts = extract_important_facts(combined_text, max_facts=5)
    questions = generate_suggested_questions(topics, facts)

    return AnalysisResult(
        document_titles=doc_titles,
        summary=summary,
        key_topics=topics,
        important_facts=facts,
        key_entities=entities,
        suggested_questions=questions,
        structural_stats=stats,
    )
