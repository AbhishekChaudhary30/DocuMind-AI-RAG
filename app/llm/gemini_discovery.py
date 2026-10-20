"""Dynamic Google Gemini model discovery, capability inspection, and caching."""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple
import requests

from app.core.logging_config import logger
from app.security.secrets_policy import validate_key_format


@dataclass
class GeminiModelInfo:
    """Metadata describing a Gemini model compatible with text generation."""

    model_id: str  # Clean identifier without "models/" prefix, e.g. "gemini-1.5-flash"
    full_name: str = ""  # Resource name, e.g. "models/gemini-1.5-flash"
    display_name: str = ""
    description: str = ""
    supported_methods: List[str] = field(default_factory=list)
    is_flash: bool = False
    is_pro: bool = False
    input_token_limit: Optional[int] = None
    output_token_limit: Optional[int] = None


DEFAULT_FALLBACK_GEMINI_MODELS: List[str] = [
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
]


def discover_gemini_models(
    api_key: str,
    base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    timeout: int = 10,
) -> Tuple[List[GeminiModelInfo], Optional[str]]:
    """Dynamically queries the official Gemini API for models available to the supplied API key.

    Filters strictly for models that include 'generateContent' in supportedGenerationMethods,
    prioritizes Flash models for speed and cost efficiency, and handles pagination.

    Returns:
        (compatible_models, error_message_or_none)
    """
    if not api_key or not api_key.strip():
        return [], "API key is required for model discovery."

    is_valid, format_err = validate_key_format("gemini", api_key.strip())
    if not is_valid:
        return [], f"API key format invalid: {format_err}"

    discovered: List[GeminiModelInfo] = []
    next_page_token = None
    url = f"{base_url.rstrip('/')}/models"

    try:
        while True:
            params: Dict[str, Any] = {
                "key": api_key.strip(),
                "pageSize": 50,
            }
            if next_page_token:
                params["pageToken"] = next_page_token

            resp = requests.get(url, params=params, timeout=timeout)

            if resp.status_code != 200:
                err_data = {}
                try:
                    err_data = resp.json().get("error", {})
                except Exception:
                    pass
                raw_err = err_data.get("message", resp.text[:200])
                clean_err = re.sub(r"AIza[0-9A-Za-z\-_]{30,}", "[REDACTED_KEY]", raw_err)

                if resp.status_code in (401, 403):
                    return [], f"Gemini authentication/permission error ({resp.status_code}): {clean_err}."
                elif resp.status_code == 404:
                    return [], f"Gemini models endpoint not found (404) on {base_url}."
                elif resp.status_code == 429:
                    return [], f"Gemini rate limit or quota exceeded (429): {clean_err}."
                else:
                    return [], f"Gemini API returned HTTP {resp.status_code}: {clean_err}"

            data = resp.json()
            models_list = data.get("models", [])

            for item in models_list:
                methods = item.get("supportedGenerationMethods", [])
                # Require generateContent capability
                if "generateContent" not in methods:
                    continue

                full_name = item.get("name", "")
                short_id = full_name.removeprefix("models/").strip()
                if not short_id:
                    continue

                display_name = item.get("displayName") or short_id
                description = item.get("description", "")
                is_flash = "flash" in short_id.lower() or "flash" in display_name.lower()
                is_pro = "pro" in short_id.lower() or "pro" in display_name.lower()

                model_info = GeminiModelInfo(
                    model_id=short_id,
                    full_name=full_name,
                    display_name=display_name,
                    description=description,
                    supported_methods=methods,
                    is_flash=is_flash,
                    is_pro=is_pro,
                    input_token_limit=item.get("inputTokenLimit"),
                    output_token_limit=item.get("outputTokenLimit"),
                )
                discovered.append(model_info)

            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break

    except requests.exceptions.Timeout:
        return [], "Connection timeout while discovering Gemini models (exceeded 10s)."
    except requests.exceptions.RequestException as exc:
        return [], f"Network error during Gemini model discovery: {type(exc).__name__}"

    if not discovered:
        return [], "No models supporting 'generateContent' were accessible with this API key."

    # Priority sorting: Flash models first (recommended), then Pro models, then others
    def sort_key(m: GeminiModelInfo):
        type_rank = 0 if m.is_flash else (1 if m.is_pro else 2)
        version_score = 0
        if "2.5" in m.model_id:
            version_score = 4
        elif "2.0" in m.model_id:
            version_score = 3
        elif "1.5" in m.model_id:
            version_score = 2
        elif "1.0" in m.model_id:
            version_score = 1
        return (type_rank, -version_score, m.model_id)

    discovered.sort(key=sort_key)
    return discovered, None
