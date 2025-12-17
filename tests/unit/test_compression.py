"""Tests for compression module."""

import pytest
from promptmanager.compression import (
    PromptCompressor,
    StrategyType,
    compress,
    count_tokens,
    TokenCounter,
    CompressionStrategy,
    CompressionConfig,
)
from promptmanager.compression.strategies.lexical import LexicalCompressor as LexicalStrategy
from promptmanager.compression.strategies.statistical import StatisticalCompressor as StatisticalStrategy
from promptmanager.compression.strategies.hybrid import HybridCompressor as HybridStrategy
from promptmanager.core.exceptions import CompressionError


class TestPromptCompressor:
    """Tests for PromptCompressor class."""

    @pytest.fixture
    def compressor(self):
        """Create a compressor instance."""
        return PromptCompressor()

    def test_create_compressor(self, compressor):
        """Test creating a compressor."""
        assert compressor is not None

    def test_compress_short_prompt(self, compressor, short_prompt):
        """Test compressing a short prompt."""
        result = compressor.compress(short_prompt, target_ratio=0.8)
        assert result is not None
        assert result.original_tokens > 0
        assert result.compressed_tokens > 0

    def test_compress_long_prompt(self, compressor, long_prompt):
        """Test compressing a long prompt."""
        result = compressor.compress(long_prompt, target_ratio=0.5)
        assert result.compression_ratio <= 1.0
        assert result.compressed_tokens <= result.original_tokens

    def test_compress_with_lexical_strategy(self, compressor, long_prompt):
        """Test compression with lexical strategy."""
        result = compressor.compress(
            long_prompt,
            target_ratio=0.7,
            strategy=StrategyType.LEXICAL
        )
        assert result.strategy_used == "lexical"

    def test_compress_with_statistical_strategy(self, compressor, long_prompt):
        """Test compression with statistical strategy."""
        result = compressor.compress(
            long_prompt,
            target_ratio=0.7,
            strategy=StrategyType.STATISTICAL
        )
        assert result.strategy_used == "statistical"

    def test_compress_with_hybrid_strategy(self, compressor, long_prompt):
        """Test compression with hybrid strategy."""
        result = compressor.compress(
            long_prompt,
            target_ratio=0.5,
            strategy=StrategyType.HYBRID
        )
        assert result.strategy_used == "hybrid"

    def test_compress_preserves_meaning(self, compressor, short_prompt):
        """Test that compression preserves key content."""
        result = compressor.compress(short_prompt, target_ratio=0.8)
        # Get compressed text
        if hasattr(result.processed, 'text'):
            compressed_text = result.processed.text
        else:
            compressed_text = str(result.processed)
        # Should still contain key words
        assert "python" in compressed_text.lower() or "function" in compressed_text.lower() or "sort" in compressed_text.lower()

    def test_compress_empty_string(self, compressor):
        """Test compressing empty string."""
        # Empty string may raise an error or return empty result
        try:
            result = compressor.compress("", target_ratio=0.5)
            assert result.original_tokens == 0
        except CompressionError:
            # Empty string handling may vary
            pass

    def test_count_tokens(self, compressor, short_prompt):
        """Test token counting."""
        count = compressor.count_tokens(short_prompt)
        assert count > 0
        assert isinstance(count, int)

    def test_different_ratios(self, compressor, long_prompt):
        """Test different compression ratios."""
        result_80 = compressor.compress(long_prompt, target_ratio=0.8)
        result_50 = compressor.compress(long_prompt, target_ratio=0.5)

        # More aggressive compression should result in fewer tokens
        assert result_50.compressed_tokens <= result_80.compressed_tokens


class TestStrategyType:
    """Tests for StrategyType enum."""

    def test_strategy_values(self):
        """Test strategy enum values."""
        assert StrategyType.LEXICAL.value == "lexical"
        assert StrategyType.STATISTICAL.value == "statistical"
        assert StrategyType.HYBRID.value == "hybrid"

    def test_strategy_from_string(self):
        """Test creating strategy from string."""
        strategy = StrategyType("lexical")
        assert strategy == StrategyType.LEXICAL


class TestLexicalStrategy:
    """Tests for LexicalStrategy."""

    @pytest.fixture
    def strategy(self):
        """Create a lexical strategy."""
        return LexicalStrategy()

    def test_compress(self, strategy, long_prompt):
        """Test lexical compression."""
        config = CompressionConfig(target_ratio=0.7)
        result = strategy.compress(long_prompt, config)
        assert len(result) <= len(long_prompt)

    def test_removes_stopwords(self, strategy):
        """Test that common stopwords are handled."""
        text = "I would like you to please help me understand"
        config = CompressionConfig(target_ratio=0.5)
        result = strategy.compress(text, config)
        # Result should be shorter
        assert len(result) < len(text)

    def test_preserves_key_content(self, strategy):
        """Test that key content is preserved."""
        text = "Write a Python function to implement binary search"
        config = CompressionConfig(target_ratio=0.8)
        result = strategy.compress(text, config)
        # Key technical terms should be preserved
        assert "python" in result.lower() or "function" in result.lower() or "binary" in result.lower()


class TestStatisticalStrategy:
    """Tests for StatisticalStrategy."""

    @pytest.fixture
    def strategy(self):
        """Create a statistical strategy."""
        return StatisticalStrategy()

    def test_compress(self, strategy, long_prompt):
        """Test statistical compression."""
        config = CompressionConfig(target_ratio=0.7)
        result = strategy.compress(long_prompt, config)
        assert len(result) <= len(long_prompt)

    def test_handles_short_text(self, strategy):
        """Test handling short text."""
        text = "Hello world"
        config = CompressionConfig(target_ratio=0.5)
        result = strategy.compress(text, config)
        assert result is not None


class TestHybridStrategy:
    """Tests for HybridStrategy."""

    @pytest.fixture
    def strategy(self):
        """Create a hybrid strategy."""
        return HybridStrategy()

    def test_compress(self, strategy, long_prompt):
        """Test hybrid compression."""
        config = CompressionConfig(target_ratio=0.5)
        result = strategy.compress(long_prompt, config)
        assert len(result) < len(long_prompt)

    def test_aggressive_compression(self, strategy, long_prompt):
        """Test aggressive compression ratio."""
        config = CompressionConfig(target_ratio=0.3)
        result = strategy.compress(long_prompt, config)
        # Should achieve significant compression
        assert len(result) < len(long_prompt) * 0.7


class TestCompressionConfig:
    """Tests for CompressionConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = CompressionConfig()
        assert config.target_ratio == 0.5  # Default is 0.5
        assert config.quality_threshold == 0.85  # Uses quality_threshold

    def test_custom_config(self):
        """Test custom configuration."""
        config = CompressionConfig(
            target_ratio=0.5,
            quality_threshold=0.9,
            aggressive_mode=True
        )
        assert config.target_ratio == 0.5
        assert config.quality_threshold == 0.9
        assert config.aggressive_mode == True


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_compress_function(self, long_prompt):
        """Test compress convenience function."""
        result = compress(long_prompt, target_ratio=0.7)
        assert result is not None
        assert result.compressed_tokens <= result.original_tokens

    def test_count_tokens_function(self, short_prompt):
        """Test count_tokens convenience function."""
        count = count_tokens(short_prompt)
        assert count > 0


class TestTokenCounting:
    """Tests for token counting functionality."""

    @pytest.fixture
    def compressor(self):
        """Create a compressor."""
        return PromptCompressor()

    def test_count_english_text(self, compressor):
        """Test counting tokens in English text."""
        text = "The quick brown fox jumps over the lazy dog"
        count = compressor.count_tokens(text)
        assert count > 0
        assert count < 20  # Reasonable token count for this sentence

    def test_count_code(self, compressor):
        """Test counting tokens in code."""
        code = "def hello(): print('Hello, World!')"
        count = compressor.count_tokens(code)
        assert count > 0

    def test_count_empty_string(self, compressor):
        """Test counting tokens in empty string."""
        count = compressor.count_tokens("")
        assert count == 0

    def test_count_unicode(self, compressor):
        """Test counting tokens with unicode."""
        text = "Hello! "
        count = compressor.count_tokens(text)
        assert count > 0


class TestEdgeCases:
    """Tests for edge cases in compression."""

    @pytest.fixture
    def compressor(self):
        """Create a compressor."""
        return PromptCompressor()

    def test_very_short_prompt(self, compressor):
        """Test compressing very short prompt."""
        result = compressor.compress("Hi", target_ratio=0.5)
        assert result is not None

    def test_single_word(self, compressor):
        """Test compressing single word."""
        result = compressor.compress("Python", target_ratio=0.5)
        assert result is not None

    def test_special_characters(self, compressor):
        """Test prompt with special characters."""
        text = "Test @#$%^& special !!! characters ???"
        result = compressor.compress(text, target_ratio=0.8)
        assert result is not None

    def test_multiline_prompt(self, compressor):
        """Test multiline prompt."""
        text = """Line 1
        Line 2
        Line 3"""
        result = compressor.compress(text, target_ratio=0.8)
        assert result is not None

    def test_ratio_bounds(self, compressor, long_prompt):
        """Test compression ratio bounds."""
        # Low ratio (valid range is 0.1-1.0)
        result_low = compressor.compress(long_prompt, target_ratio=0.2)
        assert result_low.compression_ratio <= 1.0

        # High ratio (minimal compression)
        result_high = compressor.compress(long_prompt, target_ratio=0.8)
        assert result_high.compression_ratio <= 1.0
