"""LLM provider adapters with authentication, timeouts, and error translation."""

import json
import re
import time
from typing import Optional, Tuple
import requests

from app.core.exceptions import ModelProviderError
from app.core.logging_config import logger
from app.llm.base import BaseLLMProvider
from app.models.schemas import LLMResponse


class OfflineExtractiveProvider(BaseLLMProvider):
    """Local deterministic fallback provider requiring no external API keys or network access.

    Performs extractive answer synthesis directly from the supplied retrieved context passages.
    Guarantees complete end-to-end functionality in offline and zero-cost environments.
    """

    def __init__(self, model_name: str = "offline-extractive-engine"):
        self._name = "local-offline"
        self._model = model_name

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def model_name(self) -> str:
        return self._model

    def test_connection(self) -> Tuple[bool, str]:
        return True, "Local offline extractive engine ready (Zero external credentials required)."

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        t0 = time.time()

        # Extract context block from prompt
        context_match = re.search(r"<context>(.*?)</context>", prompt, re.DOTALL)
        query_match = re.search(r"USER QUESTION:\s*(.*?)(?:\n\nPlease|\Z)", prompt, re.DOTALL)

        context_text = context_match.group(1).strip() if context_match else prompt
        query_text = query_match.group(1).strip() if query_match else ""

        # Stopwords to filter out non-informative generic terms
        stopwords = {
            "a", "an", "the", "in", "on", "of", "to", "for", "with", "at", "by", "from",
            "up", "about", "into", "over", "after", "is", "are", "was", "were", "be",
            "been", "being", "have", "has", "had", "do", "does", "did", "and", "but",
            "if", "or", "because", "as", "until", "while", "what", "which", "who", "whom",
            "this", "that", "these", "those", "am", "it", "its", "can", "could", "will",
            "would", "shall", "should", "may", "might", "must", "how", "when", "where",
            "why", "there", "their", "under", "system", "please", "based", "according"
        }

        raw_tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", query_text.lower())
        query_content_words = [w for w in raw_tokens if w not in stopwords and len(w) > 1]
        if not query_content_words:
            query_content_words = [w for w in raw_tokens if len(w) > 2]

        # Parse source blocks e.g. [Source 1] (filename - Page 1) ...
        source_pattern = re.compile(
            r"(\[Source\s+(\d+)\]\s*\(([^)]+)\))\n(.*?)(?=\n\[Source|\Z)",
            re.DOTALL,
        )
        matches = source_pattern.findall(context_text)

        candidate_sentences = []

        if matches:
            for full_tag, s_num, meta_info, body in matches:
                sentences = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", body) if len(s.strip()) > 8]
                for s in sentences:
                    s_lower = s.lower()
                    score = 0
                    for qw in query_content_words:
                        if qw in s_lower:
                            score += 2 if (any(c.isdigit() for c in qw) or len(qw) >= 4) else 1
                    if score > 0:
                        candidate_sentences.append((score, int(s_num), s))

        # Check if query had content keywords but zero were found in any retrieved document
        max_score = max((c[0] for c in candidate_sentences), default=0)
        if not candidate_sentences or (len(query_content_words) >= 3 and max_score < 2):
            content = "The provided document evidence is insufficient to answer this question. The uploaded documents do not contain reliable information matching the requested query."
            is_sufficient = False
        else:
            candidate_sentences.sort(key=lambda x: x[0], reverse=True)
            min_score_cutoff = max(2, int(max_score * 0.6))
            selected = []
            seen_previews = set()
            for score, s_num, s_text in candidate_sentences:
                if score < min_score_cutoff:
                    continue
                preview = s_text[:60].lower()
                if preview not in seen_previews:
                    seen_previews.add(preview)
                    selected.append((s_num, s_text))
                if len(selected) >= 2:
                    break

            extracted_findings = [f"{s_text} [Source {s_num}]" for s_num, s_text in selected]
            joined = " ".join(extracted_findings)
            content = f"Based on verified document passages: {joined}"
            is_sufficient = True

        latency = round(time.time() - t0, 3)
        return LLMResponse(
            content=content,
            model_name=self._model,
            provider=self._name,
            latency_seconds=latency,
            is_evidence_sufficient=is_sufficient,
        )


class OpenAICompatibleProvider(BaseLLMProvider):
    """Adapter for OpenAI, Groq, and custom OpenAI-compatible inference endpoints."""

    def __init__(
        self,
        provider_name: str,
        model_name: str,
        api_key: str,
        base_url: str,
        timeout: int = 30,
    ):
        self._provider = provider_name.lower()
        self._model = model_name
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @property
    def provider_name(self) -> str:
        return self._provider

    @property
    def model_name(self) -> str:
        return self._model

    def test_connection(self) -> Tuple[bool, str]:
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 5,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=12)
            if resp.status_code == 200:
                return True, f"Successfully connected to {self._provider.title()} API ({self._model})."
            elif resp.status_code == 401:
                return False, "Authentication failed: Invalid API key."
            elif resp.status_code == 429:
                return False, "Rate limit reached or insufficient quota on API key."
            else:
                return False, f"API error ({resp.status_code}): {resp.text[:150]}"
        except requests.exceptions.Timeout:
            return False, f"Connection timed out connecting to {self._provider} at {self._base_url}."
        except requests.exceptions.RequestException as exc:
            return False, f"Network error connecting to {self._provider}: {str(exc)}"

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        t0 = time.time()
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self._timeout)
            if resp.status_code != 200:
                raise ModelProviderError(
                    f"{self._provider.title()} generation failed ({resp.status_code}): {resp.text[:200]}",
                    provider=self._provider,
                )
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            latency = round(time.time() - t0, 3)

            is_sufficient = "insufficient" not in content.lower()[:100]

            return LLMResponse(
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model_name=self._model,
                provider=self._provider,
                latency_seconds=latency,
                is_evidence_sufficient=is_sufficient,
            )
        except requests.exceptions.RequestException as e:
            raise ModelProviderError(f"Network error from {self._provider}: {str(e)}", provider=self._provider)


class AnthropicProvider(BaseLLMProvider):
    """Adapter for Anthropic Messages API (Claude 3.5 Sonnet / Haiku)."""

    def __init__(self, model_name: str, api_key: str, timeout: int = 30):
        self._model = model_name
        self._api_key = api_key
        self._timeout = timeout
        self._url = "https://api.anthropic.com/v1/messages"

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model

    def test_connection(self) -> Tuple[bool, str]:
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 5,
        }
        try:
            resp = requests.post(self._url, headers=headers, json=payload, timeout=12)
            if resp.status_code == 200:
                return True, f"Successfully authenticated with Anthropic ({self._model})."
            elif resp.status_code == 401:
                return False, "Anthropic authentication failed: Invalid API key."
            else:
                return False, f"Anthropic error ({resp.status_code}): {resp.text[:150]}"
        except requests.exceptions.RequestException as exc:
            return False, f"Network error contacting Anthropic: {str(exc)}"

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        t0 = time.time()
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self._model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            resp = requests.post(self._url, headers=headers, json=payload, timeout=self._timeout)
            if resp.status_code != 200:
                raise ModelProviderError(f"Anthropic error ({resp.status_code}): {resp.text[:200]}", provider="anthropic")
            data = resp.json()
            content = data["content"][0]["text"]
            usage = data.get("usage", {})
            prompt_tokens = usage.get("input_tokens")
            completion_tokens = usage.get("output_tokens")
            latency = round(time.time() - t0, 3)

            return LLMResponse(
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model_name=self._model,
                provider="anthropic",
                latency_seconds=latency,
                is_evidence_sufficient="insufficient" not in content.lower()[:100],
            )
        except requests.exceptions.RequestException as e:
            raise ModelProviderError(f"Network error calling Anthropic: {str(e)}", provider="anthropic")


class GeminiProvider(BaseLLMProvider):
    """Adapter for Google Gemini API."""

    def __init__(self, model_name: str, api_key: str, timeout: int = 30):
        # Normalize model identifier removing any redundant 'models/' prefix
        self._model = model_name.removeprefix("models/").strip() if model_name else "gemini-1.5-flash"
        self._api_key = api_key.strip() if api_key else ""
        self._timeout = timeout
        self._base_url = "https://generativelanguage.googleapis.com/v1beta"

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model

    def test_connection(self) -> Tuple[bool, str]:
        """Performs a minimal generation request to verify end-to-end inference connectivity."""
        if not self._api_key:
            return False, "Gemini API key is missing. Please provide a valid key starting with 'AIza'."

        clean_model = self._model.removeprefix("models/").strip()
        url = f"{self._base_url}/models/{clean_model}:generateContent?key={self._api_key}"
        payload = {
            "contents": [{"parts": [{"text": "Hello"}]}],
            "generationConfig": {"maxOutputTokens": 5},
        }
        try:
            resp = requests.post(url, json=payload, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates and candidates[0].get("content", {}).get("parts"):
                    text_sample = candidates[0]["content"]["parts"][0].get("text", "").strip()
                    return True, f"Successfully connected to Google Gemini ({clean_model}). Response: '{text_sample[:30]}'"
                return True, f"Successfully connected to Google Gemini ({clean_model})."

            # Parse and sanitize error response without exposing the API key
            err_data = {}
            try:
                err_data = resp.json().get("error", {})
            except Exception:
                pass
            raw_err_msg = err_data.get("message", resp.text[:200])
            clean_err_msg = re.sub(r"AIza[0-9A-Za-z\-_]{30,}", "[REDACTED_KEY]", raw_err_msg)

            if resp.status_code == 400:
                return False, f"Invalid request (400): {clean_err_msg}. Verify model '{clean_model}'."
            elif resp.status_code in (401, 403):
                return False, f"Gemini authentication/permission error ({resp.status_code}): {clean_err_msg}."
            elif resp.status_code == 404:
                return (
                    False,
                    f"Model '{clean_model}' is not found for Gemini API {self._base_url.split('/')[-1]} or is not supported for generateContent (404). "
                    f"Please click '🔄 Refresh Available Models' below and select an active model (such as a Flash model)."
                )
            elif resp.status_code == 429:
                return False, f"Gemini rate limit or quota exceeded (429): {clean_err_msg}. Check your Google AI Studio quota."
            elif resp.status_code in (500, 503):
                return False, f"Google Gemini service is temporarily unavailable ({resp.status_code}). Please retry shortly."
            else:
                return False, f"Gemini API returned HTTP {resp.status_code}: {clean_err_msg}"
        except requests.exceptions.Timeout:
            return False, "Connection timeout contacting Google Gemini API (exceeded 12s)."
        except requests.exceptions.RequestException as exc:
            return False, f"Network error contacting Gemini: {type(exc).__name__}"

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        t0 = time.time()
        clean_model = self._model.removeprefix("models/").strip()
        url = f"{self._base_url}/models/{clean_model}:generateContent?key={self._api_key}"
        full_text = f"{system_prompt}\n\n{prompt}"
        payload = {
            "contents": [{"parts": [{"text": full_text}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        try:
            resp = requests.post(url, json=payload, timeout=self._timeout)
            if resp.status_code != 200:
                err_data = {}
                try:
                    err_data = resp.json().get("error", {})
                except Exception:
                    pass
                raw_err = err_data.get("message", resp.text[:200])
                clean_err = re.sub(r"AIza[0-9A-Za-z\-_]{30,}", "[REDACTED_KEY]", raw_err)
                raise ModelProviderError(f"Gemini error ({resp.status_code}): {clean_err}", provider="gemini")
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise ModelProviderError("Gemini returned zero response candidates.", provider="gemini")
            content_parts = candidates[0].get("content", {}).get("parts", [])
            if not content_parts or not content_parts[0].get("text"):
                raise ModelProviderError("Gemini returned empty content in response.", provider="gemini")
            content = content_parts[0]["text"]
            latency = round(time.time() - t0, 3)

            return LLMResponse(
                content=content,
                model_name=clean_model,
                provider="gemini",
                latency_seconds=latency,
                is_evidence_sufficient="insufficient" not in content.lower()[:100],
            )
        except requests.exceptions.RequestException as e:
            raise ModelProviderError(f"Network error calling Gemini: {type(e).__name__}", provider="gemini")


class OllamaProvider(BaseLLMProvider):
    """Adapter for local Ollama server."""

    def __init__(self, model_name: str, endpoint: Optional[str] = None, timeout: int = 45):
        self._model = model_name
        self._endpoint = (endpoint or "http://localhost:11434").rstrip("/")
        self._timeout = timeout

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    def test_connection(self) -> Tuple[bool, str]:
        url = f"{self._endpoint}/api/tags"
        try:
            resp = requests.get(url, timeout=6)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                return True, f"Ollama online at {self._endpoint}. Local models: {', '.join(models[:4]) or 'None downloaded yet'}"
            return False, f"Ollama returned HTTP status {resp.status_code}."
        except requests.exceptions.RequestException as exc:
            return False, f"Could not connect to Ollama at {self._endpoint}. Ensure Ollama is running."

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        t0 = time.time()
        url = f"{self._endpoint}/api/chat"
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": temperature},
            "stream": False,
        }
        try:
            resp = requests.post(url, json=payload, timeout=self._timeout)
            if resp.status_code != 200:
                raise ModelProviderError(f"Ollama error ({resp.status_code}): {resp.text[:200]}", provider="ollama")
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            latency = round(time.time() - t0, 3)

            return LLMResponse(
                content=content,
                prompt_tokens=data.get("prompt_eval_count"),
                completion_tokens=data.get("eval_count"),
                model_name=self._model,
                provider="ollama",
                latency_seconds=latency,
                is_evidence_sufficient="insufficient" not in content.lower()[:100],
            )
        except requests.exceptions.RequestException as e:
            raise ModelProviderError(f"Failed to communicate with Ollama: {str(e)}", provider="ollama")
