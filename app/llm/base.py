"""Abstract base class for Large Language Model provider adapters."""

from abc import ABC, abstractmethod
from typing import Tuple
from app.models.schemas import LLMResponse


class BaseLLMProvider(ABC):
    """Abstract interface defining LLM inference and connection testing."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Executes model inference and returns structured response."""
        pass

    @abstractmethod
    def test_connection(self) -> Tuple[bool, str]:
        """Performs a lightweight diagnostic check to verify credentials and connectivity."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Identifier of the provider (e.g. 'openai', 'anthropic', 'ollama')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the active model (e.g. 'gpt-4o-mini', 'llama3.2')."""
        pass
