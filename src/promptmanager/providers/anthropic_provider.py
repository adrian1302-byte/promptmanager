"""Anthropic (Claude) provider implementation."""

from typing import List, Dict, Any, AsyncIterator, Optional
import httpx

from .base import LLMProvider, LLMConfig, LLMResponse
from ..core.registry import provider_registry
from ..core.exceptions import ProviderError, ConfigurationError


@provider_registry.register("anthropic", aliases=["claude", "claude-3"])
class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider implementation."""

    # Anthropic model versions
    MODELS = {
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-3-haiku": "claude-3-haiku-20240307",
        "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku": "claude-3-5-haiku-20241022",
    }

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client: Optional[httpx.AsyncClient] = None

    def _validate(self) -> None:
        """Validate Anthropic configuration."""
        if not self.config.api_key:
            raise ConfigurationError(
                "Anthropic API key is required",
                config_key="anthropic_api_key"
            )

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url or "https://api.anthropic.com/v1",
                headers={
                    "x-api-key": self.config.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                timeout=self.config.timeout
            )
        return self._client

    def _resolve_model(self, model: str) -> str:
        """Resolve model alias to full model name."""
        return self.MODELS.get(model, model)

    def _convert_messages(
        self,
        messages: List[Dict[str, str]]
    ) -> tuple[Optional[str], List[Dict[str, str]]]:
        """
        Convert OpenAI-style messages to Anthropic format.

        Returns (system_message, messages)
        """
        system = None
        converted = []

        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                converted.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        return system, converted

    async def complete(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Generate a completion using Anthropic API."""
        client = self._get_client()

        system, converted_messages = self._convert_messages(messages)

        request_data = {
            "model": self._resolve_model(kwargs.get("model", self.config.model)),
            "messages": converted_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

        if system:
            request_data["system"] = system

        # Temperature is optional in Anthropic API
        temperature = kwargs.get("temperature", self.config.temperature)
        if temperature is not None:
            request_data["temperature"] = temperature

        # Add optional parameters
        for key in ["top_p", "top_k", "stop_sequences"]:
            if key in kwargs:
                request_data[key] = kwargs[key]

        try:
            response = await client.post("/messages", json=request_data)
            response.raise_for_status()
            data = response.json()

            # Extract content from response
            content = ""
            for block in data.get("content", []):
                if block["type"] == "text":
                    content += block["text"]

            return LLMResponse(
                content=content,
                model=data["model"],
                usage={
                    "prompt_tokens": data["usage"]["input_tokens"],
                    "completion_tokens": data["usage"]["output_tokens"],
                    "total_tokens": (
                        data["usage"]["input_tokens"] +
                        data["usage"]["output_tokens"]
                    ),
                },
                finish_reason=data.get("stop_reason"),
                raw_response=data
            )
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                f"Anthropic API error: {e.response.text}",
                provider="anthropic",
                status_code=e.response.status_code,
                cause=e
            )
        except Exception as e:
            raise ProviderError(
                f"Anthropic request failed: {str(e)}",
                provider="anthropic",
                cause=e
            )

    async def complete_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream a completion using Anthropic API."""
        client = self._get_client()

        system, converted_messages = self._convert_messages(messages)

        request_data = {
            "model": self._resolve_model(kwargs.get("model", self.config.model)),
            "messages": converted_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": True,
        }

        if system:
            request_data["system"] = system

        try:
            async with client.stream(
                "POST",
                "/messages",
                json=request_data
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        import json
                        try:
                            event = json.loads(line[6:])
                            if event["type"] == "content_block_delta":
                                delta = event.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    yield delta.get("text", "")
                        except (json.JSONDecodeError, KeyError):
                            continue
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                f"Anthropic streaming error: {e.response.text}",
                provider="anthropic",
                status_code=e.response.status_code,
                cause=e
            )

    def count_tokens(self, text: str) -> int:
        """
        Count tokens for Anthropic models.

        Note: Anthropic uses a custom tokenizer. For now, we use
        a rough approximation. For accurate counts, use the API.
        """
        try:
            # Try to use anthropic library if available
            import anthropic
            client = anthropic.Anthropic(api_key=self.config.api_key)
            return client.count_tokens(text)
        except ImportError:
            # Fallback: rough estimate (Claude is ~1 token per 3.5 chars)
            return len(text) // 3

    @property
    def provider_name(self) -> str:
        return "anthropic"

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
