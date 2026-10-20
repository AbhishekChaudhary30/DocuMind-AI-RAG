"""Text normalization, whitespace sanitization, and token estimation."""

import re
import unicodedata
from typing import List


def clean_text(raw_text: str) -> str:
    """Normalizes Unicode, strips unprintable characters, and standardizes spacing."""
    if not raw_text or not isinstance(raw_text, str):
        return ""

    # 1. Unicode Normalization (NFKC standardizes ligatures, full-width chars)
    normalized = unicodedata.normalize("NFKC", raw_text)

    # 2. Remove Non-Printable and Control Characters (preserve \n and \t)
    sanitized = "".join(
        ch for ch in normalized if ch in ("\n", "\t") or (unicodedata.category(ch)[0] != "C" and ord(ch) >= 32)
    )

    # 3. Collapse multiple horizontal spaces/tabs on each line while preserving indentation
    lines = sanitized.splitlines()
    cleaned_lines = []
    for line in lines:
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(cleaned_line)

    # 4. Collapse excessive consecutive blank lines (limit to max 2)
    reconstructed = "\n".join(cleaned_lines)
    collapsed = re.sub(r"\n{3,}", "\n\n", reconstructed)

    return collapsed.strip()


def estimate_tokens(text: str) -> int:
    """Estimates token count for English/mixed technical text.

    Uses the established empirical ratio of approximately 4 characters per token
    or 0.75 words per token. Never claims to be an exact proprietary tokenizer.
    """
    if not text:
        return 0

    clean = text.strip()
    char_estimate = len(clean) / 4.0
    words = clean.split()
    word_estimate = len(words) * 1.33

    # Blend character and word estimates for robust approximation
    blended = (char_estimate + word_estimate) / 2.0
    return max(1, int(round(blended)))


def detect_scanned_text(text: str, min_chars: int = 50) -> bool:
    """Detects whether an extracted page has virtually no readable text."""
    cleaned = clean_text(text)
    return len(cleaned) < min_chars


def remove_repetitive_headers_footers(page_texts: List[str]) -> List[str]:
    """Identifies and removes exact running headers and footers that repeat across pages."""
    if len(page_texts) < 3:
        return page_texts

    first_lines = []
    last_lines = []
    for p in page_texts:
        lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
        first_lines.append(lines[0] if lines else "")
        last_lines.append(lines[-1] if lines else "")

    # Count occurrences
    header_candidates = {}
    for fl in first_lines:
        if fl and len(fl) < 100:
            header_candidates[fl] = header_candidates.get(fl, 0) + 1

    footer_candidates = {}
    for ll in last_lines:
        if ll and len(ll) < 100:
            footer_candidates[ll] = footer_candidates.get(ll, 0) + 1

    threshold = len(page_texts) * 0.6  # Repeats on at least 60% of pages

    filtered_pages = []
    for p in page_texts:
        lines = [ln for ln in p.splitlines()]
        if not lines:
            filtered_pages.append(p)
            continue

        # Check top line
        if lines and lines[0].strip() in header_candidates and header_candidates[lines[0].strip()] >= threshold:
            lines = lines[1:]

        # Check bottom line
        if lines and lines[-1].strip() in footer_candidates and footer_candidates[lines[-1].strip()] >= threshold:
            lines = lines[:-1]

        filtered_pages.append("\n".join(lines).strip())

    return filtered_pages
