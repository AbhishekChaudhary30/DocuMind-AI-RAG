"""Mathematical metric formulations for quantitative RAG retrieval and answer evaluation."""

from typing import List, Tuple
from app.models.schemas import Citation, RetrievalResult


def calculate_hit_rate(
    retrieved_results: List[RetrievalResult],
    expected_doc_substring: str,
    expected_pages: List[int],
    k: int = 3,
) -> float:
    """Hit Rate @ K: 1.0 if at least one candidate in top K matches expected document and page, else 0.0.

    Formula:
        Hit@K = 1 if exists r in top_k(R) s.t. match(r) else 0
    """
    if not expected_doc_substring:
        # For unanswerable questions, hit rate is not applicable or 1.0 if empty
        return 1.0

    allowed_docs = [d.strip().lower() for d in expected_doc_substring.split("|") if d.strip()]
    candidates = retrieved_results[:k]
    for r in candidates:
        fname = r.chunk.filename.lower()
        if any(d in fname for d in allowed_docs):
            if not expected_pages or any(p in r.chunk.page_numbers for p in expected_pages):
                return 1.0
    return 0.0


def calculate_mrr(
    retrieved_results: List[RetrievalResult],
    expected_doc_substring: str,
    expected_pages: List[int],
) -> float:
    """Mean Reciprocal Rank: 1 / rank of the first relevant candidate.

    Formula:
        RR = 1 / rank_first_relevant, or 0.0 if not found.
    """
    if not expected_doc_substring:
        return 1.0

    allowed_docs = [d.strip().lower() for d in expected_doc_substring.split("|") if d.strip()]
    for idx, r in enumerate(retrieved_results):
        fname = r.chunk.filename.lower()
        if any(d in fname for d in allowed_docs):
            if not expected_pages or any(p in r.chunk.page_numbers for p in expected_pages):
                rank = idx + 1
                return round(1.0 / rank, 4)
    return 0.0


def calculate_precision_recall(
    retrieved_results: List[RetrievalResult],
    expected_doc_substring: str,
    expected_pages: List[int],
    k: int = 5,
) -> Tuple[float, float]:
    """Precision@K and Recall@K across retrieved candidates.

    Formula:
        Precision@K = (# relevant chunks in top K) / K
        Recall@K = (# relevant chunks in top K) / (total relevant chunks, min 1)
    """
    if not expected_doc_substring:
        return 1.0, 1.0

    candidates = retrieved_results[:k]
    if not candidates:
        return 0.0, 0.0

    allowed_docs = [d.strip().lower() for d in expected_doc_substring.split("|") if d.strip()]
    relevant_in_top_k = 0
    for r in candidates:
        fname = r.chunk.filename.lower()
        if any(d in fname for d in allowed_docs):
            if not expected_pages or any(p in r.chunk.page_numbers for p in expected_pages):
                relevant_in_top_k += 1

    precision = round(relevant_in_top_k / len(candidates), 4)
    # Recall approximation: normalized against top_k or at least 1 relevant
    recall = round(min(1.0, relevant_in_top_k / max(1, len(expected_pages))), 4)
    return precision, recall


def evaluate_citation_precision(
    citations: List[Citation],
    expected_doc_substring: str,
    expected_pages: List[int],
) -> float:
    """Evaluates the proportion of citations that are authentic and point to the gold source page.

    Formula:
        CitationPrecision = (# valid citations matching gold document & page) / max(1, total citations)
    """
    if not citations:
        # If no citations expected (e.g. unanswerable), precision is 1.0 if 0 citations emitted
        return 1.0 if not expected_doc_substring else 0.0

    allowed_docs = [d.strip().lower() for d in expected_doc_substring.split("|") if d.strip()]
    matched = 0
    for c in citations:
        if c.is_valid:
            if not expected_doc_substring:
                # Citations were not expected for unanswerable query
                continue
            fname_match = any(d in c.filename.lower() for d in allowed_docs)
            page_match = not expected_pages or c.page_number in expected_pages
            if fname_match and page_match:
                matched += 1

    return round(matched / len(citations), 4)


def evaluate_grounding_and_abstention(
    answer_text: str,
    expected_behavior: str,
    expected_keywords: List[str],
) -> Tuple[bool, float]:
    """Deterministic check assessing whether the answer is correctly grounded or abstains appropriately.

    Returns:
        (passed, keyword_coverage_ratio)
    """
    text_lower = answer_text.lower()

    if expected_behavior == "abstain":
        abstention_cues = ["insufficient", "not provide", "not contain", "unsupported", "cannot answer"]
        passed = any(cue in text_lower for cue in abstention_cues)
        return passed, 1.0 if passed else 0.0

    # Answer expected: check presence of required factual keywords
    if not expected_keywords:
        return True, 1.0

    found_count = sum(1 for kw in expected_keywords if kw.lower() in text_lower)
    coverage = round(found_count / len(expected_keywords), 4)
    passed = coverage >= 0.5  # At least half of key factual anchors present
    return passed, coverage
