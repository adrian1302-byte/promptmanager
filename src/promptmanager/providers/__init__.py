"""LLM Provider abstractions for provider-agnostic design."""

from .base import LLMProvider, LLMConfig, LLMResponse
from ..core.registry import provider_registry

__all__ = [
    "LLMProvider",
    "LLMConfig",
    "LLMResponse",
    "get_provider",
    "list_providers",
]


def get_provider(name: str = "openai", **kwargs) -> LLMProvider:
    """
    Get a configured LLM provider instance.

    Args:
        name: Provider name ("openai", "anthropic", "litellm")
        **kwargs: Provider-specific configuration

    Returns:
        Configured LLMProvider instance
    """
    from ..core.config import get_settings
    settings = get_settings()

    # Build config from settings and kwargs
    config_kwargs = {
        "model": kwargs.get("model", settings.provider.default_model),
        "timeout": kwargs.get("timeout", settings.provider.timeout),
    }

    if name == "openai":
        config_kwargs["api_key"] = kwargs.get("api_key", settings.provider.openai_api_key)
    elif name == "anthropic":
        config_kwargs["api_key"] = kwargs.get("api_key", settings.provider.anthropic_api_key)

    config_kwargs.update(kwargs)
    config = LLMConfig(**config_kwargs)

    return provider_registry.get_provider(name, config)


def list_providers() -> list:
    """List all available providers."""
    return provider_registry.list_registered()


# Import providers to register them
from . import openai_provider, anthropic_provider, litellm_adapter
