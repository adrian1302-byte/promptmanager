"""Main compression orchestrator."""

from typing import Optional, Union, List, Dict, Any
from enum import Enum
import time

from ..core.types import Prompt, CompressionResult, ContentType
from ..core.base import Compressor
from ..core.exceptions import CompressionError
from .strategies.base import CompressionStrategy, CompressionConfig
from .strategies.lexical import LexicalCompressor
from .strategies.statistical import StatisticalCompressor
from .strategies.code import CodeCompressor
from .strategies.hybrid import HybridCompressor
from .tokenizers.registry import tokenizer_registry


class StrategyType(Enum):
    """Available compression strategies."""
    LEXICAL = "lexical"
    STATISTICAL = "statistical"
    SEMANTIC = "semantic"
    CODE = "code"
    HYBRID = "hybrid"
    AUTO = "auto"


class PromptCompressor(Compressor):
    """
    Main orchestrator for prompt compression.

    Provides a unified interface to multiple compression strategies
    with automatic strategy selection and quality preservation.

    Example:
        >>> compressor = PromptCompressor(tokenizer="gpt-4")
        >>> result = await compressor.compress(
        ...     prompt,
        ...     target_ratio=0.5,
        ...     strategy="hybrid"
        ... )
        >>> print(f"Compressed from {result.original_tokens} to {result.compressed_tokens}")
    """

    name = "prompt_compressor"
    description = "Main compression orchestrator with multi-strategy support"

    def __init__(
        self,
        tokenizer: str = "gpt-4",
        default_strategy: Union[str, StrategyType] = StrategyType.HYBRID,
        quality_threshold: float = 0.85,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the PromptCompressor.

        Args:
            tokenizer: Tokenizer to use for counting (e.g., "gpt-4", "gpt-3.5-turbo")
            default_strategy: Default compression strategy
            quality_threshold: Minimum quality score to preserve (0-1)
            config: Additional configuration options
        """
        super().__init__(config)

        self.tokenizer_name = tokenizer
        self.tokenizer = tokenizer_registry.get(tokenizer)
        self.default_strategy = self._resolve_strategy(default_strategy)
        self.quality_threshold = quality_threshold

        # Initialize strategies
        self._strategies: Dict[StrategyType, CompressionStrategy] = {}
        self._init_strategies()

    def _init_strategies(self) -> None:
        """Initialize available compression strategies."""
        self._strategies = {
            StrategyType.LEXICAL: LexicalCompressor(),
            StrategyType.STATISTICAL: StatisticalCompressor(),
            StrategyType.CODE: CodeCompressor(),
            StrategyType.HYBRID: HybridCompressor(),
        }

        # Try to load semantic compressor (optional dependency)
        try:
            from .strategies.semantic import SemanticCompressor
            self._strategies[StrategyType.SEMANTIC] = SemanticCompressor()
        except ImportError:
            pass

    async def process(
        self,
        prompt: Prompt,
        target_ratio: float = 0.5,
        preserve_instruction: bool = True,
        **kwargs
    ) -> CompressionResult:
        """
        Compress a prompt (async interface).

        Args:
            prompt: The prompt to compress
            target_ratio: Target size as ratio of original (0.5 = 50%)
            preserve_instruction: Whether to preserve instruction portions
            **kwargs: Additional options (strategy, quality_threshold, etc.)

        Returns:
            CompressionResult with compressed prompt and metrics
        """
        return self.compress(prompt, target_ratio, preserve_instruction, **kwargs)

    def compress(
        self,
        prompt: Union[str, Prompt],
        target_ratio: float = 0.5,
        strategy: Optional[Union[str, StrategyType]] = None,
        quality_threshold: Optional[float] = None,
        preserve_instruction: bool = True,
        content_type: Optional[str] = None,
        **kwargs
    ) -> CompressionResult:
        """
        Compress a prompt.

        Args:
            prompt: The prompt to compress (string or Prompt object)
            target_ratio: Target size as ratio of original (0.5 = 50% of tokens)
            strategy: Compression strategy ("lexical", "statistical", "hybrid", "auto")
            quality_threshold: Minimum quality score to preserve
            preserve_instruction: Whether to preserve instruction portions
            content_type: Content type hint ("code", "natural_language", etc.)
            **kwargs: Additional strategy-specific options

        Returns:
            CompressionResult with compressed prompt and metrics
        """
        start_time = time.perf_counter()

        # Normalize input
        if isinstance(prompt, str):
            prompt = Prompt.from_text(prompt)

        # Get original text and token count
        original_text = prompt.text
        original_tokens = self.tokenizer.count(original_text)

        # Resolve parameters
        strategy_type = self._resolve_strategy(strategy or self.default_strategy)
        quality_threshold = quality_threshold or self.quality_threshold

        # Detect content type
        detected_content_type = self._detect_content_type(original_text, content_type)

        # Auto-select strategy if needed
        if strategy_type == StrategyType.AUTO:
            strategy_type = self._auto_select_strategy(original_text, detected_content_type)

        # Validate strategy is available
        if strategy_type not in self._strategies:
            raise CompressionError(
                f"Strategy '{strategy_type.value}' is not available",
                strategy=strategy_type.value
            )

        # Build compression config
        config = CompressionConfig(
            target_ratio=target_ratio,
            quality_threshold=quality_threshold,
            preserve_first_n_tokens=100 if preserve_instruction else 0,
            aggressive_mode=target_ratio < 0.4,
            **{k: v for k, v in kwargs.items() if k in CompressionConfig.__dataclass_fields__}
        )

        try:
            # Execute compression
            compressor = self._strategies[strategy_type]
            compressed_text = compressor.compress(
                original_text,
                config,
                detected_content_type.value if isinstance(detected_content_type, ContentType) else detected_content_type
            )

            # Count compressed tokens
            compressed_tokens = self.tokenizer.count(compressed_text)

            # Create compressed prompt
            compressed_prompt = prompt.copy()
            if compressed_prompt.messages:
                # Update the last user message (typically the main content)
                for i in range(len(compressed_prompt.messages) - 1, -1, -1):
                    if compressed_prompt.messages[i].role.value == "user":
                        compressed_prompt.messages[i].content = compressed_text
                        break
                else:
                    compressed_prompt.messages[-1].content = compressed_text

            # Calculate metrics
            compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
            processing_time = self._measure_time(start_time)

            return CompressionResult(
                original=prompt,
                processed=compressed_prompt,
                success=True,
                processing_time_ms=processing_time,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                compression_ratio=compression_ratio,
                strategy_used=strategy_type.value,
                quality_preserved=self._estimate_quality(original_text, compressed_text),
                metrics={
                    "tokens_saved": original_tokens - compressed_tokens,
                    "reduction_percentage": (1 - compression_ratio) * 100,
                },
                metadata={
                    "tokenizer": self.tokenizer_name,
                    "content_type": detected_content_type.value if isinstance(detected_content_type, ContentType) else detected_content_type,
                }
            )

        except Exception as e:
            raise CompressionError(
                f"Compression failed: {str(e)}",
                strategy=strategy_type.value,
                cause=e
            )

    def estimate_compression(self, prompt: Union[str, Prompt]) -> float:
        """
        Estimate achievable compression ratio.

        Args:
            prompt: The prompt to analyze

        Returns:
            Estimated compression ratio (0-1)
        """
        if isinstance(prompt, str):
            text = prompt
        else:
            text = prompt.text

        # Get estimates from all strategies
        estimates = {}
        for strategy_type, compressor in self._strategies.items():
            estimates[strategy_type.value] = compressor.estimate_compression_ratio(text)

        # Return the best (lowest) estimate
        return min(estimates.values())

    def estimate_all_strategies(
        self,
        prompt: Union[str, Prompt]
    ) -> Dict[str, float]:
        """
        Get compression estimates from all strategies.

        Args:
            prompt: The prompt to analyze

        Returns:
            Dictionary of strategy -> estimated ratio
        """
        if isinstance(prompt, str):
            text = prompt
        else:
            text = prompt.text

        return {
            strategy_type.value: compressor.estimate_compression_ratio(text)
            for strategy_type, compressor in self._strategies.items()
        }

    def count_tokens(self, text: str, tokenizer: Optional[str] = None) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count
            tokenizer: Optional tokenizer override

        Returns:
            Token count
        """
        if tokenizer:
            counter = tokenizer_registry.get(tokenizer)
            return counter.count(text)
        return self.tokenizer.count(text)

    def list_strategies(self) -> List[Dict[str, Any]]:
        """List all available strategies with their capabilities."""
        return [
            {
                "name": strategy_type.value,
                **compressor.get_capabilities()
            }
            for strategy_type, compressor in self._strategies.items()
        ]

    def list_tokenizers(self) -> List[str]:
        """List available tokenizers."""
        return tokenizer_registry.list_available()

    def _resolve_strategy(
        self,
        strategy: Union[str, StrategyType]
    ) -> StrategyType:
        """Convert string to StrategyType enum."""
        if isinstance(strategy, StrategyType):
            return strategy
        try:
            return StrategyType(strategy.lower())
        except ValueError:
            raise CompressionError(
                f"Unknown strategy: {strategy}. Available: {[s.value for s in StrategyType]}"
            )

    def _detect_content_type(
        self,
        text: str,
        hint: Optional[str]
    ) -> ContentType:
        """Detect content type from text or use hint."""
        if hint:
            try:
                return ContentType(hint.lower())
            except ValueError:
                pass

        # Use hybrid compressor's detection
        hybrid = self._strategies.get(StrategyType.HYBRID)
        if hybrid and hasattr(hybrid, '_detect_content_type'):
            content_type_str = hybrid._detect_content_type(text)
            try:
                return ContentType(content_type_str)
            except ValueError:
                pass

        return ContentType.NATURAL_LANGUAGE

    def _auto_select_strategy(
        self,
        text: str,
        content_type: ContentType
    ) -> StrategyType:
        """Auto-select best strategy based on content."""
        # Code content -> code strategy
        if content_type == ContentType.CODE:
            return StrategyType.CODE

        token_count = self.tokenizer.count(text)

        # Short texts -> fast lexical
        if token_count < 500:
            return StrategyType.LEXICAL

        # Medium texts -> statistical
        if token_count < 2000:
            return StrategyType.STATISTICAL

        # Long texts -> hybrid
        return StrategyType.HYBRID

    def _estimate_quality(self, original: str, compressed: str) -> float:
        """
        Estimate quality preservation (simple heuristic).

        Returns a score between 0 and 1.
        """
        # Word overlap as a simple quality proxy
        original_words = set(original.lower().split())
        compressed_words = set(compressed.lower().split())

        if not original_words:
            return 1.0

        overlap = len(original_words & compressed_words)
        return overlap / len(original_words)


# Convenience functions
def compress(
    text: str,
    target_ratio: float = 0.5,
    tokenizer: str = "gpt-4",
    strategy: str = "hybrid"
) -> CompressionResult:
    """
    Quick compression function.

    Args:
        text: Input text to compress
        target_ratio: Target size as ratio of original
        tokenizer: Tokenizer to use
        strategy: Compression strategy

    Returns:
        CompressionResult
    """
    compressor = PromptCompressor(tokenizer=tokenizer)
    return compressor.compress(text, target_ratio=target_ratio, strategy=strategy)


def count_tokens(text: str, tokenizer: str = "gpt-4") -> int:
    """
    Quick token counting function.

    Args:
        text: Text to count
        tokenizer: Tokenizer to use

    Returns:
        Token count
    """
    counter = tokenizer_registry.get(tokenizer)
    return counter.count(text)
