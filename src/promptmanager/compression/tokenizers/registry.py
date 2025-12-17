"""Tokenizer registry for managing token counters."""

from typing import Dict, List, Optional
from .base import TokenCounter, SimpleTokenCounter


class TokenizerRegistry:
    """
    Registry for managing tokenizer instances.

    Provides caching and lazy loading of tokenizers.
    """

    def __init__(self):
        self._counters: Dict[str, TokenCounter] = {}
        self._fallback = SimpleTokenCounter()

    def get(self, name: str) -> TokenCounter:
        """
        Get or create a token counter.

        Args:
            name: Tokenizer/model name

        Returns:
            TokenCounter instance
        """
        if name not in self._counters:
            counter = self._create_counter(name)
            self._counters[name] = counter
        return self._counters[name]

    def _create_counter(self, name: str) -> TokenCounter:
        """Create a token counter for the given name."""
        # OpenAI models -> tiktoken
        openai_models = [
            "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini",
            "gpt-3.5-turbo", "gpt-3.5-turbo-16k",
            "text-embedding-ada-002", "text-embedding-3-small",
            "text-davinci-003", "davinci", "curie", "babbage", "ada"
        ]

        # Check for tiktoken models
        if name in openai_models or name.startswith("gpt-"):
            try:
                from .tiktoken_counter import TiktokenCounter
                return TiktokenCounter(name)
            except ImportError:
                pass

        # Check for HuggingFace models
        if "/" in name or name in ["gpt2", "llama", "mistral"]:
            try:
                from .hf_counter import HuggingFaceCounter
                return HuggingFaceCounter(name)
            except ImportError:
                pass

        # Try tiktoken with default encoding
        try:
            from .tiktoken_counter import TiktokenCounter
            return TiktokenCounter("gpt-4")  # Default to GPT-4 encoding
        except ImportError:
            pass

        # Fallback to simple counter
        return self._fallback

    def register(self, name: str, counter: TokenCounter) -> None:
        """
        Register a custom token counter.

        Args:
            name: Name to register under
            counter: TokenCounter instance
        """
        self._counters[name] = counter

    def list_available(self) -> List[str]:
        """List available tokenizer names."""
        available = list(self._counters.keys())

        # Add known tokenizers that can be loaded
        known = [
            "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo",
            "gpt2", "simple"
        ]

        for name in known:
            if name not in available:
                available.append(name)

        return sorted(available)

    def clear_cache(self) -> None:
        """Clear cached tokenizer instances."""
        self._counters.clear()

    def is_loaded(self, name: str) -> bool:
        """Check if a tokenizer is loaded."""
        return name in self._counters


# Global registry instance
tokenizer_registry = TokenizerRegistry()


def get_tokenizer(name: str = "gpt-4") -> TokenCounter:
    """
    Get a tokenizer by name.

    Convenience function for accessing the global registry.

    Args:
        name: Tokenizer/model name

    Returns:
        TokenCounter instance
    """
    return tokenizer_registry.get(name)


def count_tokens(text: str, tokenizer: str = "gpt-4") -> int:
    """
    Count tokens in text.

    Convenience function for quick token counting.

    Args:
        text: Text to count tokens for
        tokenizer: Tokenizer/model name

    Returns:
        Token count
    """
    counter = tokenizer_registry.get(tokenizer)
    return counter.count(text)
