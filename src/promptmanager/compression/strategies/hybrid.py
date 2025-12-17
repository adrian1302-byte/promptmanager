"""Hybrid compression strategy combining multiple approaches."""

from typing import Optional, List, Tuple
from .base import CompressionStrategy, CompressionConfig
from .lexical import LexicalCompressor
from .statistical import StatisticalCompressor
from .code import CodeCompressor


class HybridCompressor(CompressionStrategy):
    """
    Hybrid compression combining multiple strategies adaptively.

    Automatically selects and combines strategies based on:
    - Content type detection
    - Target compression ratio
    - Text characteristics

    This is the recommended default strategy for production use.
    """

    name = "hybrid"
    description = "Adaptive hybrid compression combining multiple strategies"
    supports_streaming = False
    requires_external_model = False

    def __init__(self):
        self._lexical = LexicalCompressor()
        self._statistical = StatisticalCompressor()
        self._code = CodeCompressor()

        # Try to load semantic compressor (optional dependency)
        self._semantic = None
        self._has_semantic = False
        try:
            from .semantic import SemanticCompressor
            self._semantic = SemanticCompressor()
            self._has_semantic = True
        except ImportError:
            pass

    def compress(
        self,
        text: str,
        config: CompressionConfig,
        content_type: Optional[str] = None
    ) -> str:
        """Apply hybrid compression using multiple strategies."""
        # Detect content type if not provided
        if content_type is None:
            content_type = self._detect_content_type(text)

        # Build compression pipeline
        pipeline = self._build_pipeline(text, config, content_type)

        # Apply strategies in sequence
        result = text
        original_len = len(text)

        for stage_name, compressor, stage_config in pipeline:
            result = compressor.compress(result, stage_config, content_type)

            # Check if we've reached target ratio
            current_ratio = len(result) / original_len
            if current_ratio <= config.target_ratio:
                break

        return result

    def estimate_compression_ratio(self, text: str) -> float:
        """Estimate achievable compression ratio."""
        content_type = self._detect_content_type(text)

        # Get estimates from applicable strategies
        estimates = []

        estimates.append(self._lexical.estimate_compression_ratio(text))
        estimates.append(self._statistical.estimate_compression_ratio(text))

        if content_type == "code":
            estimates.append(self._code.estimate_compression_ratio(text))

        if self._has_semantic:
            estimates.append(self._semantic.estimate_compression_ratio(text))

        # Hybrid can often achieve better than individual strategies
        return min(estimates) * 0.9

    def _detect_content_type(self, text: str) -> str:
        """Detect content type from text."""
        import re

        # Code detection
        code_indicators = [
            r'\bdef\s+\w+\s*\(', r'\bfunction\s+\w+', r'\bclass\s+\w+',
            r'\bimport\s+\w+', r'#include', r'\bpublic\s+class',
            r'\bconst\s+\w+\s*=', r'\blet\s+\w+\s*=', r'\bvar\s+\w+\s*=',
            r'^\s*\{[\s\S]*\}\s*$',  # JSON-like
        ]

        code_matches = sum(
            1 for pattern in code_indicators
            if re.search(pattern, text, re.MULTILINE)
        )

        if code_matches >= 2:
            return "code"

        # Structured data detection
        if text.strip().startswith('{') or text.strip().startswith('['):
            return "structured_data"

        # Conversation detection
        conversation_patterns = [
            r'^(User|Human|Assistant|AI|Bot|System):\s*',
            r'^>\s*',
            r'^(Q|A):\s*',
        ]
        conv_matches = sum(
            1 for pattern in conversation_patterns
            if re.search(pattern, text, re.MULTILINE)
        )

        if conv_matches >= 2:
            return "conversation"

        return "natural_language"

    def _build_pipeline(
        self,
        text: str,
        config: CompressionConfig,
        content_type: str
    ) -> List[Tuple[str, CompressionStrategy, CompressionConfig]]:
        """Build compression pipeline based on content and target."""
        pipeline = []

        # Calculate intermediate targets
        # Start with gentle compression, increase if needed
        initial_target = max(0.8, config.target_ratio + 0.2)

        # Stage 1: Always start with lexical (fast, safe)
        lexical_config = CompressionConfig(
            target_ratio=initial_target,
            quality_threshold=config.quality_threshold,
            preserve_patterns=config.preserve_patterns,
            preserve_first_n_tokens=config.preserve_first_n_tokens,
            preserve_last_n_tokens=config.preserve_last_n_tokens,
            aggressive_mode=False
        )
        pipeline.append(("lexical", self._lexical, lexical_config))

        # Stage 2: Content-specific compression
        if content_type == "code":
            code_config = CompressionConfig(
                target_ratio=max(0.7, config.target_ratio + 0.1),
                quality_threshold=config.quality_threshold,
                preserve_patterns=config.preserve_patterns,
                aggressive_mode=config.target_ratio < 0.5
            )
            pipeline.append(("code", self._code, code_config))

        # Stage 3: Statistical compression for further reduction
        if config.target_ratio < 0.7:
            stat_config = CompressionConfig(
                target_ratio=config.target_ratio,
                quality_threshold=config.quality_threshold,
                preserve_patterns=config.preserve_patterns,
                aggressive_mode=config.target_ratio < 0.5
            )
            pipeline.append(("statistical", self._statistical, stat_config))

        # Stage 4: Semantic compression if available and needed
        if self._has_semantic and config.target_ratio < 0.5:
            sem_config = CompressionConfig(
                target_ratio=config.target_ratio,
                quality_threshold=config.quality_threshold,
                aggressive_mode=True
            )
            pipeline.append(("semantic", self._semantic, sem_config))

        return pipeline

    def get_capabilities(self) -> dict:
        """Return capabilities including available sub-strategies."""
        return {
            "name": self.name,
            "description": self.description,
            "supports_streaming": self.supports_streaming,
            "requires_external_model": self.requires_external_model,
            "sub_strategies": ["lexical", "statistical", "code"] +
                            (["semantic"] if self._has_semantic else []),
            "has_semantic": self._has_semantic,
        }
