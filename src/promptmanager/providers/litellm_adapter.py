"""LiteLLM adapter for universal provider support."""

from typing import List, Dict, Any, AsyncIterator, Optional

from .base import LLMProvider, LLMConfig, LLMResponse
from ..core.registry import provider_registry
from ..core.exceptions import ProviderError, ConfigurationError


@provider_registry.register("litellm", aliases=["universal", "any"])
class LiteLLMProvider(LLMProvider):
    """
    Universal provider using LiteLLM for 100+ model support.

    LiteLLM provides a unified interface to multiple LLM providers:
    - OpenAI, Anthropic, Google, Cohere, Replicate
    - Azure OpenAI, AWS Bedrock, Vertex AI
    - Ollama, vLLM, and other local models

    Model format: "provider/model" (e.g., "anthropic/claude-3-opus")
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._litellm = None

    def _validate(self) -> None:
        """Validate LiteLLM is available."""
        try:
            import litellm
            self._litellm = litellm
        except ImportError:
            raise ConfigurationError(
                "LiteLLM is required for universal provider support. "
                "Install with: pip install litellm",
                config_key="litellm"
            )

    async def complete(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Generate a completion using LiteLLM."""
        try:
            response = await self._litellm.acompletion(
                model=kwargs.get("model", self.config.model),
                messages=messages,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                **{k: v for k, v in kwargs.items()
                   if k not in ["model", "temperature", "max_tokens"]}
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                finish_reason=response.choices[0].finish_reason,
                raw_response=response
            )
        except Exception as e:
            raise ProviderError(
                f"LiteLLM request failed: {str(e)}",
                provider="litellm",
                cause=e
            )

    async def complete_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream a completion using LiteLLM."""
        try:
            response = await self._litellm.acompletion(
                model=kwargs.get("model", self.config.model),
                messages=messages,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                stream=True,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )

            async for chunk in response:
                if hasattr(chunk.choices[0].delta, 'content'):
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
        except Exception as e:
            raise ProviderError(
                f"LiteLLM streaming failed: {str(e)}",
                provider="litellm",
                cause=e
            )

    def count_tokens(self, text: str) -> int:
        """Count tokens using LiteLLM's token counter."""
        try:
            return self._litellm.token_counter(
                model=self.config.model,
                text=text
            )
        except Exception:
            # Fallback: rough estimate
            return len(text) // 4

    @property
    def provider_name(self) -> str:
        return "litellm"

    def list_models(self) -> List[str]:
        """List available models through LiteLLM."""
        # Common models supported by LiteLLM
        return [
            # OpenAI
            "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo",
            # Anthropic
            "anthropic/claude-3-opus", "anthropic/claude-3-sonnet",
            "anthropic/claude-3-haiku", "anthropic/claude-3-5-sonnet",
            # Google
            "gemini/gemini-pro", "gemini/gemini-1.5-pro",
            # Cohere
            "cohere/command", "cohere/command-light",
            # Local
            "ollama/llama2", "ollama/mistral", "ollama/codellama",
        ]


# Factory function for creating provider based on model string
def create_provider_from_model(model: str, **kwargs) -> LLMProvider:
    """
    Create a provider instance based on model string.

    Automatically detects the provider from model naming:
    - "gpt-*" -> OpenAI
    - "claude-*" -> Anthropic
    - "provider/model" -> LiteLLM

    Args:
        model: Model name or "provider/model" format
        **kwargs: Additional configuration

    Returns:
        Configured LLMProvider instance
    """
    from ..core.config import get_settings
    settings = get_settings()

    # Detect provider from model name
    if "/" in model:
        # LiteLLM format: provider/model
        config = LLMConfig(
            model=model,
            api_key=kwargs.get("api_key"),
            **kwargs
        )
        return LiteLLMProvider(config)

    elif model.startswith("gpt-") or model.startswith("text-"):
        # OpenAI model
        from .openai_provider import OpenAIProvider
        config = LLMConfig(
            model=model,
            api_key=kwargs.get("api_key", settings.provider.openai_api_key),
            **kwargs
        )
        return OpenAIProvider(config)

    elif model.startswith("claude"):
        # Anthropic model
        from .anthropic_provider import AnthropicProvider
        config = LLMConfig(
            model=model,
            api_key=kwargs.get("api_key", settings.provider.anthropic_api_key),
            **kwargs
        )
        return AnthropicProvider(config)

    else:
        # Default to LiteLLM for universal support
        config = LLMConfig(
            model=model,
            api_key=kwargs.get("api_key"),
            **kwargs
        )
        return LiteLLMProvider(config)
