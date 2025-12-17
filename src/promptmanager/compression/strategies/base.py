"""Abstract base class for compression strategies."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class CompressionConfig:
    """Configuration for compression operations."""
    target_ratio: float = 0.5  # Target size as ratio of original (0.5 = 50%)
    quality_threshold: float = 0.85  # Minimum quality to preserve
    preserve_patterns: List[str] = field(default_factory=list)  # Regex patterns to preserve
    preserve_first_n_tokens: int = 0  # Preserve first N tokens (e.g., instructions)
    preserve_last_n_tokens: int = 0  # Preserve last N tokens
    aggressive_mode: bool = False  # Allow more aggressive compression
    max_iterations: int = 5  # Max compression iterations for iterative strategies

    def __post_init__(self):
        # Validate ratio
        if not 0.1 <= self.target_ratio <= 1.0:
            raise ValueError("target_ratio must be between 0.1 and 1.0")
        if not 0.0 <= self.quality_threshold <= 1.0:
            raise ValueError("quality_threshold must be between 0.0 and 1.0")


class CompressionStrategy(ABC):
    """
    Abstract base class for compression strategies.

    All compression implementations must inherit from this class
    and implement the required methods.
    """

    name: str = "base"
    description: str = "Base compression strategy"
    supports_streaming: bool = False
    requires_external_model: bool = False

    @abstractmethod
    def compress(
        self,
        text: str,
        config: CompressionConfig,
        content_type: Optional[str] = None
    ) -> str:
        """
        Compress the input text.

        Args:
            text: Input text to compress
            config: Compression configuration
            content_type: Optional hint about content type

        Returns:
            Compressed text
        """
        pass

    @abstractmethod
    def estimate_compression_ratio(self, text: str) -> float:
        """
        Estimate achievable compression ratio without actually compressing.

        Args:
            text: Text to analyze

        Returns:
            Estimated ratio (1.0 = no compression, 0.5 = 50% size reduction)
        """
        pass

    def get_capabilities(self) -> Dict[str, Any]:
        """Return strategy capabilities and metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "supports_streaming": self.supports_streaming,
            "requires_external_model": self.requires_external_model,
        }

    def _should_preserve(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any preservation patterns."""
        import re
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _split_preserved_sections(
        self,
        text: str,
        config: CompressionConfig
    ) -> tuple:
        """
        Split text into preserved and compressible sections.

        Returns (prefix_to_preserve, middle_to_compress, suffix_to_preserve)
        """
        words = text.split()
        total_words = len(words)

        # Estimate tokens (rough: 1 word ~= 1.3 tokens)
        tokens_per_word = 1.3

        prefix_words = int(config.preserve_first_n_tokens / tokens_per_word)
        suffix_words = int(config.preserve_last_n_tokens / tokens_per_word)

        # Ensure we don't exceed total
        prefix_words = min(prefix_words, total_words)
        suffix_words = min(suffix_words, total_words - prefix_words)

        prefix = " ".join(words[:prefix_words]) if prefix_words > 0 else ""
        suffix = " ".join(words[-suffix_words:]) if suffix_words > 0 else ""
        middle = " ".join(words[prefix_words:total_words - suffix_words if suffix_words > 0 else total_words])

        return prefix, middle, suffix
