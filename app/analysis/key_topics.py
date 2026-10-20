"""Key topics, domain concept modeling, and named entity extraction."""

import re
from collections import Counter
from typing import Dict, List

COMMON_STOPWORDS = {
    "the", "and", "is", "in", "to", "of", "for", "with", "on", "at", "by", "from",
    "up", "about", "into", "over", "after", "this", "that", "these", "those", "it",
    "as", "an", "be", "are", "was", "were", "been", "will", "would", "shall", "should",
    "can", "could", "may", "might", "must", "or", "not", "so", "if", "out", "then",
    "which", "what", "where", "when", "how", "all", "any", "each", "other", "some",
}


def extract_key_topics(text: str, top_n: int = 6) -> List[str]:
    """Extracts top thematic topics using clean frequency filtering."""
    if not text:
        return []

    words = re.findall(r"\b[a-zA-Z][a-zA-Z\-]{3,}\b", text.lower())
    filtered = [w for w in words if w not in COMMON_STOPWORDS]

    counts = Counter(filtered)
    topics = [item[0].capitalize() for item in counts.most_common(top_n)]
    return topics


def extract_key_entities(text: str) -> Dict[str, List[str]]:
    """Extracts organizations, dates, monetary metrics, and technical protocols."""
    if not text:
        return {}

    entities: Dict[str, List[str]] = {
        "Monetary Values": [],
        "Dates & Periods": [],
        "Organizations & Systems": [],
        "Technical Terms": [],
    }

    # 1. Monetary Values: e.g. $142.5M, $28.1, USD 500,000
    monetary = set(re.findall(r"(?:\$|USD\s*)\s*\d+(?:\.\d+)?\s*(?:M|B|K|million|billion)?", text, re.IGNORECASE))
    entities["Monetary Values"] = sorted(list(monetary))[:8]

    # 2. Dates & Periods: e.g. FY2025, Q1 FY25, January 1, 2025, 2024
    dates = set(re.findall(r"\b(?:FY20\d\d|Q[1-4]\s*FY\d\d|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|20\d\d)\b", text))
    entities["Dates & Periods"] = sorted(list(dates))[:8]

    # 3. Organizations and System Names (Title Case phrases with suffixes)
    orgs = set(re.findall(r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*(?:\s+(?:Group|Corp|Inc|LLC|Systems|Technology|Specification|Team|Engine))\b", text))
    entities["Organizations & Systems"] = sorted(list(orgs))[:8]

    # 4. Technical Terms & Protocols
    tech = set(re.findall(r"\b(?:RAG|PyMuPDF|LLM|FAISS|BM25|API|HTTPS|OAuth|SHA-256|Microservices|Vector|Tokens|Embeddings)\b", text, re.IGNORECASE))
    entities["Technical Terms"] = sorted(list(set(t.upper() for t in tech)))[:8]

    # Remove empty entity categories
    return {k: v for k, v in entities.items() if v}
