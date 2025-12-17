"""Abstract base class for token counters."""

from abc import ABC, abstractmethod
from typing import List, Optional


class TokenCounter(ABC):
    """
    Abstract base class for token counters.

    Provides a unified interface for counting tokens across different
    tokenizers (tiktoken, HuggingFace, etc.).
    """

    name: str = "base"

    @abstractmethod
    def count(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Token count
        """
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Text to encode

        Returns:
            List of token IDs
        """
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text
        """
        pass

    def truncate(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to maximum tokens.

        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens

        Returns:
            Truncated text
        """
        tokens = self.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.decode(tokens[:max_tokens])

    def truncate_to_fit(
        self,
        text: str,
        max_tokens: int,
        suffix: str = "..."
    ) -> str:
        """
        Truncate text to fit within max_tokens, adding suffix if truncated.

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens including suffix
            suffix: Suffix to add if truncated

        Returns:
            Truncated text with suffix if needed
        """
        tokens = self.encode(text)
        if len(tokens) <= max_tokens:
            return text

        # Reserve tokens for suffix
        suffix_tokens = self.count(suffix)
        available = max_tokens - suffix_tokens

        if available <= 0:
            return suffix

        truncated = self.decode(tokens[:available])
        return truncated + suffix

    def split_by_tokens(
        self,
        text: str,
        chunk_size: int,
        overlap: int = 0
    ) -> List[str]:
        """
        Split text into chunks of approximately chunk_size tokens.

        Args:
            text: Text to split
            chunk_size: Target tokens per chunk
            overlap: Number of overlapping tokens between chunks

        Returns:
            List of text chunks
        """
        tokens = self.encode(text)

        if len(tokens) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunks.append(self.decode(chunk_tokens))

            start = end - overlap
            if start >= len(tokens):
                break

        return chunks


class SimpleTokenCounter(TokenCounter):
    """
    Simple token counter using character/word approximation.

    Used as a fallback when no tokenizer library is available.
    """

    name = "simple"

    def __init__(self, chars_per_token: float = 4.0):
        """
        Initialize with characters per token ratio.

        Args:
            chars_per_token: Average characters per token (default: 4.0)
        """
        self.chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        """Count tokens using character approximation."""
        return max(1, int(len(text) / self.chars_per_token))

    def encode(self, text: str) -> List[int]:
        """Encode by splitting into pseudo-tokens."""
        # Split by characters to simulate tokens
        chunk_size = int(self.chars_per_token)
        return list(range(0, len(text), chunk_size))

    def decode(self, tokens: List[int]) -> str:
        """
        Cannot decode with simple counter.

        This is a limitation of the approximation approach.
        """
        raise NotImplementedError(
            "SimpleTokenCounter cannot decode tokens. "
            "Use a proper tokenizer for decode operations."
        )

    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate based on character approximation."""
        max_chars = int(max_tokens * self.chars_per_token)
        if len(text) <= max_chars:
            return text
        return text[:max_chars]
