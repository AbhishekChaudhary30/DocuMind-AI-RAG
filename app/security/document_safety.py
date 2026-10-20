"""Document safety, prompt injection detection, and content isolation guards."""

import re
from typing import List, Tuple

INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(?:the\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:in\s+)?developer\s+mode", re.IGNORECASE),
    re.compile(r"reveal\s+(?:your\s+)?(?:system\s+prompt|api\s+key|secret)", re.IGNORECASE),
    re.compile(r"system\s+override\s*:", re.IGNORECASE),
]


def inspect_for_prompt_injection(text: str) -> Tuple[bool, List[str]]:
    """Scans untrusted document passages for adversarial prompt injection signatures."""
    if not text:
        return False, []

    detected = []
    for pattern in INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            detected.append(f"Suspicious instruction pattern detected: '{match.group(0)}'")

    return len(detected) > 0, detected


def sanitize_for_prompt_injection(text: str) -> str:
    """Neutralizes adversarial injection phrases by prepending safety flags."""
    if not text:
        return ""

    sanitized = text
    for pattern in INJECTION_PATTERNS:
        sanitized = pattern.sub(r"[UNTRUSTED_DOC_INSTRUCTION_NEUTRALIZED: \g<0>]", sanitized)

    return sanitized
