"""Cloud and self-hosted provider embedding adapters."""

from typing import List, Optional
import numpy as np
import requests

from app.core.exceptions import ModelProviderError
from app.embeddings.base import BaseEmbeddingProvider, normalize_vectors


class ProviderEmbeddingAdapter(BaseEmbeddingProvider):
    """Adapter for generating embeddings via OpenAI, Gemini, or Ollama APIs."""

    def __init__(
        self,
        provider: str,
        model_name: str,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: int = 20,
    ):
        self._provider = provider.lower()
        self._model_name = model_name
        self._api_key = api_key
        self._endpoint = endpoint
        self._timeout = timeout

        if self._provider == "openai":
            self._dim = 1536 if "large" not in model_name else 3072
            if "small" in model_name:
                self._dim = 1536
        elif self._provider == "gemini":
            self._dim = 768
        elif self._provider == "ollama":
            self._dim = 768 if "nomic" in model_name else 384
        else:
            self._dim = 384

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return f"{self._provider}:{self._model_name}"

    def _call_openai(self, texts: List[str]) -> np.ndarray:
        if not self._api_key:
            raise ModelProviderError("OpenAI embedding requires an active API key.", provider="openai")

        url = f"{self._endpoint or 'https://api.openai.com/v1'}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model_name,
            "input": texts,
            "encoding_format": "float",
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self._timeout)
            if resp.status_code != 200:
                raise ModelProviderError(f"OpenAI embedding request failed ({resp.status_code}): {resp.text[:200]}")
            data = resp.json()
            rows = [item["embedding"] for item in data["data"]]
            arr = np.array(rows, dtype=np.float32)
            self._dim = arr.shape[1]
            return normalize_vectors(arr)
        except requests.RequestException as e:
            raise ModelProviderError(f"Network error contacting OpenAI embeddings: {str(e)}")

    def _call_ollama(self, texts: List[str]) -> np.ndarray:
        base_url = self._endpoint or "http://localhost:11434"
        url = f"{base_url.rstrip('/')}/api/embed"
        payload = {
            "model": self._model_name,
            "input": texts,
        }
        try:
            resp = requests.post(url, json=payload, timeout=self._timeout)
            if resp.status_code != 200:
                raise ModelProviderError(f"Ollama embed failed ({resp.status_code}): {resp.text[:200]}")
            data = resp.json()
            embeddings = data.get("embeddings") or []
            if not embeddings:
                raise ModelProviderError("Ollama returned empty embeddings response.")
            arr = np.array(embeddings, dtype=np.float32)
            self._dim = arr.shape[1]
            return normalize_vectors(arr)
        except requests.RequestException as e:
            raise ModelProviderError(f"Could not connect to Ollama at {base_url}: {str(e)}")

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if self._provider == "openai":
            vecs = self._call_openai(texts)
        elif self._provider == "ollama":
            vecs = self._call_ollama(texts)
        else:
            raise ModelProviderError(f"Remote embeddings not supported for provider '{self._provider}'.")

        self.validate_vectors(vecs, len(texts))
        return vecs

    def embed_query(self, text: str) -> np.ndarray:
        vecs = self.embed_documents([text])
        return vecs
