"""Abstract LLM provider interface."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, AsyncIterator
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: float = 30.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": self.model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            **self.extra
        }


@dataclass
class LLMResponse:
    """Standardized LLM response across providers."""
    content: str
    model: str
    usage: Dict[str, int]  # prompt_tokens, completion_tokens, total_tokens
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_response: Any = None

    @property
    def prompt_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        return self.usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)


class LLMProvider(ABC):
    """
    Abstract LLM provider interface.

    All provider implementations must inherit from this class
    and implement the required methods.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._validate()

    @abstractmethod
    def _validate(self) -> None:
        """Validate provider configuration. Raises ConfigurationError if invalid."""
        pass

    @abstractmethod
    async def complete(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """
        Generate a completion for the given messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional provider-specific options

        Returns:
            LLMResponse with completion
        """
        pass

    @abstractmethod
    async def complete_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream a completion for the given messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional provider-specific options

        Yields:
            Content chunks as they arrive
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using provider's tokenizer.

        Args:
            text: Text to count tokens for

        Returns:
            Token count
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name for identification."""
        pass

    @property
    def model_name(self) -> str:
        """Return the current model name."""
        return self.config.model

    async def complete_text(self, prompt: str, **kwargs) -> str:
        """
        Simple text completion helper.

        Args:
            prompt: Text prompt
            **kwargs: Additional options

        Returns:
            Completion text
        """
        messages = [{"role": "user", "content": prompt}]
        response = await self.complete(messages, **kwargs)
        return response.content

    def complete_text_sync(self, prompt: str, **kwargs) -> str:
        """Synchronous version of complete_text."""
        import asyncio
        return asyncio.run(self.complete_text(prompt, **kwargs))
