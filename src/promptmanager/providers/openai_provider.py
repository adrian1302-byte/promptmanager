"""OpenAI provider implementation."""

from typing import List, Dict, Any, AsyncIterator, Optional
import httpx

from .base import LLMProvider, LLMConfig, LLMResponse
from ..core.registry import provider_registry
from ..core.exceptions import ProviderError, ConfigurationError


@provider_registry.register("openai", aliases=["gpt", "gpt-4", "gpt-3.5"])
class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation."""

    # Model to encoding mapping for token counting
    MODEL_ENCODINGS = {
        "gpt-4": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4-turbo-preview": "cl100k_base",
        "gpt-4o": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "gpt-3.5-turbo-16k": "cl100k_base",
    }

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client: Optional[httpx.AsyncClient] = None
        self._tokenizer = None

    def _validate(self) -> None:
        """Validate OpenAI configuration."""
        if not self.config.api_key:
            raise ConfigurationError(
                "OpenAI API key is required",
                config_key="openai_api_key"
            )

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url or "https://api.openai.com/v1",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.config.timeout
            )
        return self._client

    async def complete(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Generate a completion using OpenAI API."""
        client = self._get_client()

        request_data = {
            "model": kwargs.get("model", self.config.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

        # Add any extra parameters
        for key in ["top_p", "frequency_penalty", "presence_penalty", "stop"]:
            if key in kwargs:
                request_data[key] = kwargs[key]

        try:
            response = await client.post("/chat/completions", json=request_data)
            response.raise_for_status()
            data = response.json()

            return LLMResponse(
                content=data["choices"][0]["message"]["content"],
                model=data["model"],
                usage={
                    "prompt_tokens": data["usage"]["prompt_tokens"],
                    "completion_tokens": data["usage"]["completion_tokens"],
                    "total_tokens": data["usage"]["total_tokens"],
                },
                finish_reason=data["choices"][0].get("finish_reason"),
                raw_response=data
            )
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                f"OpenAI API error: {e.response.text}",
                provider="openai",
                status_code=e.response.status_code,
                cause=e
            )
        except Exception as e:
            raise ProviderError(
                f"OpenAI request failed: {str(e)}",
                provider="openai",
                cause=e
            )

    async def complete_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream a completion using OpenAI API."""
        client = self._get_client()

        request_data = {
            "model": kwargs.get("model", self.config.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": True,
        }

        try:
            async with client.stream(
                "POST",
                "/chat/completions",
                json=request_data
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        import json
                        try:
                            chunk = json.loads(line[6:])
                            delta = chunk["choices"][0]["delta"]
                            if "content" in delta and delta["content"]:
                                yield delta["content"]
                        except (json.JSONDecodeError, KeyError):
                            continue
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                f"OpenAI streaming error: {e.response.text}",
                provider="openai",
                status_code=e.response.status_code,
                cause=e
            )

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        if self._tokenizer is None:
            try:
                import tiktoken
                # Get encoding for model
                model = self.config.model
                if model in self.MODEL_ENCODINGS:
                    self._tokenizer = tiktoken.get_encoding(self.MODEL_ENCODINGS[model])
                else:
                    try:
                        self._tokenizer = tiktoken.encoding_for_model(model)
                    except KeyError:
                        # Fallback to cl100k_base (GPT-4 encoding)
                        self._tokenizer = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                # Fallback: rough estimate (1 token ~= 4 chars)
                return len(text) // 4

        return len(self._tokenizer.encode(text))

    @property
    def provider_name(self) -> str:
        return "openai"

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
