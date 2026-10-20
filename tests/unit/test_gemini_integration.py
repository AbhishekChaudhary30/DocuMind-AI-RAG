"""Unit tests for Google Gemini dynamic discovery, provider adapter, connection test, error taxonomy, and RAG integration."""

import json
from unittest.mock import MagicMock, patch
import pytest
import requests

from app.core.exceptions import ModelProviderError
from app.llm import get_llm_provider
from app.llm.gemini_discovery import (
    DEFAULT_FALLBACK_GEMINI_MODELS,
    GeminiModelInfo,
    discover_gemini_models,
)
from app.llm.providers import GeminiProvider, OfflineExtractiveProvider
from app.models.schemas import DocumentChunk, RetrievalResult
from app.rag.answer_generator import generate_grounded_answer
from app.security.secrets_policy import purge_session_credentials


SAMPLE_GEMINI_KEY = "AIzaSyD-mock-testing-gemini-key-123456"


@pytest.fixture
def mock_retrieval_results():
    c1 = DocumentChunk(
        chunk_id="chk-101",
        doc_id="doc-guide",
        filename="system_architecture.pdf",
        chunk_index=0,
        start_page=3,
        end_page=3,
        page_numbers=[3],
        text="The vector database performs cosine similarity on normalized unit vectors.",
        estimated_tokens=14,
    )
    c2 = DocumentChunk(
        chunk_id="chk-102",
        doc_id="doc-guide",
        filename="system_architecture.pdf",
        chunk_index=1,
        start_page=4,
        end_page=5,
        page_numbers=[4, 5],
        text="Hybrid search uses Reciprocal Rank Fusion with constant k=60 to merge dense and keyword search.",
        estimated_tokens=16,
    )
    return [
        RetrievalResult(chunk=c1, score=0.88, retrieval_mode="hybrid", rank=1),
        RetrievalResult(chunk=c2, score=0.75, retrieval_mode="hybrid", rank=2),
    ]


# =========================================================================
# 1. Dynamic Model Discovery Tests
# =========================================================================

def test_discover_gemini_models_success():
    """Verifies model discovery filters for generateContent, prioritizes Flash, and strips prefix."""
    mock_payload = {
        "models": [
            {
                "name": "models/text-embedding-004",
                "displayName": "Text Embedding 004",
                "description": "Embedding model",
                "supportedGenerationMethods": ["embedContent"],
            },
            {
                "name": "models/gemini-1.5-pro",
                "displayName": "Gemini 1.5 Pro",
                "description": "Complex reasoning model",
                "supportedGenerationMethods": ["generateContent", "countTokens"],
            },
            {
                "name": "models/gemini-1.5-flash",
                "displayName": "Gemini 1.5 Flash",
                "description": "Fast multimodal model",
                "supportedGenerationMethods": ["generateContent", "countTokens"],
            },
            {
                "name": "models/gemini-2.0-flash",
                "displayName": "Gemini 2.0 Flash",
                "description": "Next-gen fast model",
                "supportedGenerationMethods": ["generateContent"],
            },
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_payload

    with patch("requests.get", return_value=mock_resp) as mock_get:
        models, err = discover_gemini_models(SAMPLE_GEMINI_KEY)

    assert err is None
    assert len(models) == 3

    # Embedding model must NOT be included
    model_ids = [m.model_id for m in models]
    assert "text-embedding-004" not in model_ids
    assert "models/text-embedding-004" not in model_ids

    # Flash models MUST be prioritized first
    assert models[0].is_flash is True
    assert "flash" in models[0].model_id.lower()
    assert models[1].is_flash is True

    # Pro model comes after Flash models
    pro_model = next(m for m in models if "pro" in m.model_id.lower())
    assert pro_model.is_pro is True
    assert models.index(pro_model) > 0

    # Ensure URL used the key safely
    call_args = mock_get.call_args
    assert "key" in call_args[1]["params"]
    assert call_args[1]["params"]["key"] == SAMPLE_GEMINI_KEY


def test_discover_gemini_models_pagination():
    """Verifies that discover_gemini_models follows nextPageToken."""
    page_1 = {
        "models": [
            {
                "name": "models/gemini-1.5-flash",
                "displayName": "Gemini 1.5 Flash",
                "supportedGenerationMethods": ["generateContent"],
            }
        ],
        "nextPageToken": "token_for_page_2",
    }
    page_2 = {
        "models": [
            {
                "name": "models/gemini-1.5-pro",
                "displayName": "Gemini 1.5 Pro",
                "supportedGenerationMethods": ["generateContent"],
            }
        ]
    }

    mock_resp_1 = MagicMock(status_code=200, json=lambda: page_1)
    mock_resp_2 = MagicMock(status_code=200, json=lambda: page_2)

    with patch("requests.get", side_effect=[mock_resp_1, mock_resp_2]) as mock_get:
        models, err = discover_gemini_models(SAMPLE_GEMINI_KEY)

    assert err is None
    assert len(models) == 2
    assert mock_get.call_count == 2
    model_ids = [m.model_id for m in models]
    assert "gemini-1.5-flash" in model_ids
    assert "gemini-1.5-pro" in model_ids


def test_discover_gemini_models_no_compatible_models():
    """Verifies fallback when API returns no models with generateContent capability."""
    mock_payload = {
        "models": [
            {
                "name": "models/text-embedding-004",
                "displayName": "Text Embedding 004",
                "supportedGenerationMethods": ["embedContent"],
            }
        ]
    }
    mock_resp = MagicMock(status_code=200, json=lambda: mock_payload)

    with patch("requests.get", return_value=mock_resp):
        models, err = discover_gemini_models(SAMPLE_GEMINI_KEY)

    assert err is not None
    assert "No models supporting 'generateContent'" in err
    assert len(models) == 0


def test_discover_gemini_models_auth_error_401_403():
    """Verifies handling of invalid credentials without leaking the secret."""
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.text = f"API_KEY_INVALID for {SAMPLE_GEMINI_KEY}"
    mock_resp.json.return_value = {
        "error": {
            "code": 403,
            "message": f"API key not valid. Please pass a valid API key: {SAMPLE_GEMINI_KEY}",
            "status": "PERMISSION_DENIED",
        }
    }

    with patch("requests.get", return_value=mock_resp):
        models, err = discover_gemini_models(SAMPLE_GEMINI_KEY)

    assert models == []
    assert err is not None
    assert "403" in err
    # Critical security assertion: API key must NOT be leaked in error message
    assert SAMPLE_GEMINI_KEY not in err
    assert "[REDACTED_KEY]" in err


def test_discover_gemini_models_quota_error_429():
    """Verifies handling of rate limit or quota exhaustion."""
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.json.return_value = {
        "error": {
            "code": 429,
            "message": "Resource has been exhausted (e.g. check quota).",
            "status": "RESOURCE_EXHAUSTED",
        }
    }

    with patch("requests.get", return_value=mock_resp):
        models, err = discover_gemini_models(SAMPLE_GEMINI_KEY)

    assert models == []
    assert err is not None
    assert "429" in err
    assert "quota" in err.lower() or "rate limit" in err.lower()


def test_discover_gemini_models_not_found_404():
    """Verifies handling of 404 endpoint/model not found."""
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.json.return_value = {
        "error": {
            "code": 404,
            "message": "Method not found.",
        }
    }

    with patch("requests.get", return_value=mock_resp):
        models, err = discover_gemini_models(SAMPLE_GEMINI_KEY)

    assert models == []
    assert err is not None
    assert "404" in err


def test_discover_gemini_models_timeout():
    """Verifies timeout handling returns clean error message without raising exception."""
    with patch("requests.get", side_effect=requests.exceptions.Timeout("Connection timed out")):
        models, err = discover_gemini_models(SAMPLE_GEMINI_KEY)

    assert models == []
    assert err is not None
    assert "timeout" in err.lower()


def test_discover_gemini_models_empty_key():
    """Verifies empty or malformed key is rejected prior to network call."""
    with patch("requests.get") as mock_get:
        models, err = discover_gemini_models("")
        assert models == []
        assert err is not None
        assert "API key is required" in err
        mock_get.assert_not_called()


# =========================================================================
# 2. GeminiProvider Connection & Generation Tests
# =========================================================================

def test_gemini_provider_test_connection_success():
    """Verifies real minimal generation request during connection test."""
    provider = GeminiProvider(model_name="gemini-1.5-flash", api_key=SAMPLE_GEMINI_KEY)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello! How can I help you today?"}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ]
    }

    with patch("requests.post", return_value=mock_resp) as mock_post:
        ok, msg = provider.test_connection()

    assert ok is True
    assert "Successfully connected to Google Gemini (gemini-1.5-flash)" in msg
    assert "Hello!" in msg
    mock_post.assert_called_once()
    # Confirm endpoint used stripped model name and maxOutputTokens=5
    sent_url = mock_post.call_args[0][0]
    assert "models/gemini-1.5-flash:generateContent" in sent_url
    assert mock_post.call_args[1]["json"]["generationConfig"]["maxOutputTokens"] == 5


def test_gemini_provider_test_connection_404_model_unavailable():
    """Verifies reproduction and actionable diagnosis of the user's confirmed 404 error."""
    provider = GeminiProvider(model_name="gemini-1.5-pro", api_key=SAMPLE_GEMINI_KEY)

    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.json.return_value = {
        "error": {
            "code": 404,
            "message": "models/gemini-1.5-pro is not found for API version v1beta, or is not supported for generateContent.",
        }
    }

    with patch("requests.post", return_value=mock_resp):
        ok, msg = provider.test_connection()

    assert ok is False
    assert "404" in msg
    assert "gemini-1.5-pro" in msg
    assert "Refresh Available Models" in msg or "Flash" in msg


def test_gemini_provider_test_connection_429_quota():
    """Verifies 429 quota exhaustion is accurately described."""
    provider = GeminiProvider(model_name="gemini-1.5-flash", api_key=SAMPLE_GEMINI_KEY)

    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.json.return_value = {
        "error": {
            "code": 429,
            "message": "Resource has been exhausted (check quota).",
        }
    }

    with patch("requests.post", return_value=mock_resp):
        ok, msg = provider.test_connection()

    assert ok is False
    assert "quota exceeded (429)" in msg


def test_gemini_provider_redacts_api_key_in_errors():
    """Verifies that API keys in error payloads are always redacted."""
    provider = GeminiProvider(model_name="gemini-1.5-flash", api_key=SAMPLE_GEMINI_KEY)

    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.text = f"Bad request with key {SAMPLE_GEMINI_KEY}"
    mock_resp.json.return_value = {
        "error": {
            "code": 400,
            "message": f"Invalid argument for {SAMPLE_GEMINI_KEY}",
        }
    }

    with patch("requests.post", return_value=mock_resp):
        ok, msg = provider.test_connection()

    assert ok is False
    assert SAMPLE_GEMINI_KEY not in msg
    assert "[REDACTED_KEY]" in msg


def test_gemini_provider_generate_success():
    """Verifies generate() produces a valid LLMResponse."""
    provider = GeminiProvider(model_name="models/gemini-1.5-flash", api_key=SAMPLE_GEMINI_KEY)
    assert provider.model_name == "gemini-1.5-flash"  # Prefix stripped

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "DocuMind AI utilizes hybrid search [Source 1]."}],
                }
            }
        ]
    }

    with patch("requests.post", return_value=mock_resp):
        response = provider.generate(
            prompt="What search does DocuMind use?",
            system_prompt="Answer based only on retrieved context.",
        )

    assert response.provider == "gemini"
    assert response.model_name == "gemini-1.5-flash"
    assert "hybrid search [Source 1]" in response.content
    assert response.is_evidence_sufficient is True


def test_gemini_provider_generate_empty_candidates():
    """Verifies error handling when Gemini returns no response candidates."""
    provider = GeminiProvider(model_name="gemini-1.5-flash", api_key=SAMPLE_GEMINI_KEY)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"candidates": []}

    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(ModelProviderError) as exc_info:
            provider.generate(prompt="Hello", system_prompt="Answer.")

    assert "zero response candidates" in str(exc_info.value)


# =========================================================================
# 3. RAG Integration & Offline Fallback Tests
# =========================================================================

def test_rag_generation_with_configured_gemini_model(mock_retrieval_results):
    """Verifies full grounded answer generation using the Gemini provider."""
    provider = GeminiProvider(model_name="gemini-1.5-flash", api_key=SAMPLE_GEMINI_KEY)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "The vector database performs cosine similarity on normalized unit vectors [Source 1]."
                        }
                    ],
                }
            }
        ]
    }

    with patch("requests.post", return_value=mock_resp):
        response, citations = generate_grounded_answer(
            query="How does the vector database work?",
            retrieved_results=mock_retrieval_results,
            llm_provider=provider,
        )

    assert response.provider == "gemini"
    assert response.model_name == "gemini-1.5-flash"
    assert len(citations) >= 1
    assert citations[0].source_label == "[Source 1]"
    assert citations[0].chunk_id == "chk-101"


def test_gemini_offline_fallback_without_api_key(mock_retrieval_results):
    """Verifies that selecting Gemini without an API key safely falls back to OfflineExtractiveProvider."""
    llm = get_llm_provider("gemini", model_id="gemini-1.5-flash", api_key=None)
    assert isinstance(llm, OfflineExtractiveProvider)

    # Must generate grounded answers locally without any network access
    response, citations = generate_grounded_answer(
        query="vector database",
        retrieved_results=mock_retrieval_results,
        llm_provider=llm,
    )

    assert response.provider == "local-offline"
    assert len(response.content) > 0


# =========================================================================
# 4. Credential Security & Invalidation Tests
# =========================================================================

def test_purge_session_credentials_clears_gemini_discovery():
    """Verifies that purging credentials clears both keys and discovered Gemini model metadata."""
    session = {
        "api_key": SAMPLE_GEMINI_KEY,
        "gemini_discovered_models": [
            GeminiModelInfo(model_id="gemini-1.5-flash", display_name="Flash", is_flash=True)
        ],
        "discovered_models": ["gemini-1.5-flash"],
        "model_config": {"provider": "gemini", "model_id": "gemini-1.5-flash", "is_connected": True},
    }

    purged = purge_session_credentials(session)
    assert purged >= 3
    assert "api_key" not in session
    assert "gemini_discovered_models" not in session
    assert "discovered_models" not in session
