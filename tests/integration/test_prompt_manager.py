"""Integration tests for PromptManager class."""

import pytest
import asyncio
from promptmanager import (
    PromptManager,
    StrategyType,
    EnhancementMode,
    EnhancementLevel,
    PromptStyle,
)


class TestPromptManagerIntegration:
    """Integration tests for the main PromptManager class."""

    @pytest.fixture
    def pm(self):
        """Create a PromptManager instance."""
        return PromptManager()

    # Compression Tests
    @pytest.mark.asyncio
    async def test_compress_basic(self, pm, long_prompt):
        """Test basic compression through PromptManager."""
        result = await pm.compress(long_prompt, ratio=0.7)
        assert result is not None
        assert result.compression_ratio <= 1.0
        assert result.compressed_tokens <= result.original_tokens

    @pytest.mark.asyncio
    async def test_compress_all_strategies(self, pm, long_prompt):
        """Test all compression strategies."""
        strategies = ["lexical", "statistical", "hybrid"]
        for strategy in strategies:
            result = await pm.compress(long_prompt, ratio=0.7, strategy=strategy)
            assert result is not None
            assert result.strategy_used == strategy

    def test_compress_sync(self, pm, long_prompt):
        """Test synchronous compression."""
        result = pm.compress_sync(long_prompt, ratio=0.7)
        assert result is not None

    # Enhancement Tests
    @pytest.mark.asyncio
    async def test_enhance_basic(self, pm, messy_prompt):
        """Test basic enhancement through PromptManager."""
        result = await pm.enhance(messy_prompt)
        assert result is not None
        assert len(result.improvements) > 0

    @pytest.mark.asyncio
    async def test_enhance_all_modes(self, pm, messy_prompt):
        """Test all enhancement modes (without LLM)."""
        result = await pm.enhance(messy_prompt, mode=EnhancementMode.RULES_ONLY)
        assert result is not None

    @pytest.mark.asyncio
    async def test_enhance_all_levels(self, pm, messy_prompt):
        """Test all enhancement levels."""
        levels = [EnhancementLevel.MINIMAL, EnhancementLevel.LIGHT,
                  EnhancementLevel.MODERATE, EnhancementLevel.AGGRESSIVE]
        for level in levels:
            result = await pm.enhance(messy_prompt, mode=EnhancementMode.RULES_ONLY, level=level)
            assert result is not None

    def test_enhance_sync(self, pm, messy_prompt):
        """Test synchronous enhancement."""
        result = pm.enhance_sync(messy_prompt, mode=EnhancementMode.RULES_ONLY)
        assert result is not None

    # Generation Tests
    @pytest.mark.asyncio
    async def test_generate_basic(self, pm):
        """Test basic generation through PromptManager."""
        result = await pm.generate(task="Write a hello world program")
        assert result is not None
        assert result.processed != ""

    @pytest.mark.asyncio
    async def test_generate_all_styles(self, pm):
        """Test all generation styles."""
        styles = [PromptStyle.ZERO_SHOT, PromptStyle.CHAIN_OF_THOUGHT,
                  PromptStyle.CODE_GENERATION]
        for style in styles:
            result = await pm.generate(task="Write code", style=style)
            assert result is not None

    @pytest.mark.asyncio
    async def test_generate_with_examples(self, pm):
        """Test generation with examples."""
        examples = [
            {"input": "Hello", "output": "Hi there!"},
            {"input": "Goodbye", "output": "See you later!"},
        ]
        result = await pm.generate(
            task="Respond to greetings",
            style=PromptStyle.FEW_SHOT,
            examples=examples
        )
        assert result is not None

    def test_generate_sync(self, pm):
        """Test synchronous generation."""
        result = pm.generate_sync(task="Test task")
        assert result is not None

    # Validation Tests
    def test_validate_valid_prompt(self, pm, short_prompt):
        """Test validating a valid prompt."""
        result = pm.validate(short_prompt)
        assert result.is_valid == True
        assert result.score > 0.5

    def test_validate_invalid_prompt(self, pm, empty_prompt):
        """Test validating invalid prompt."""
        result = pm.validate(empty_prompt)
        assert result.is_valid == False

    def test_validate_injection_prompt(self, pm, injection_prompt):
        """Test validating injection prompt."""
        result = pm.validate(injection_prompt)
        assert result.is_valid == False

    # Analysis Tests
    @pytest.mark.asyncio
    async def test_analyze_basic(self, pm, code_prompt):
        """Test basic analysis through PromptManager."""
        result = await pm.analyze(code_prompt)
        assert "intent" in result
        assert "quality" in result

    @pytest.mark.asyncio
    async def test_analyze_detects_intent(self, pm):
        """Test that analysis detects correct intent."""
        # Code prompt
        code_result = await pm.analyze("Write a Python function")
        assert code_result["intent"]["primary"] == "code_generation"

        # Question prompt
        qa_result = await pm.analyze("What is machine learning?")
        assert qa_result["intent"]["primary"] == "question_answering"

    def test_analyze_sync(self, pm, code_prompt):
        """Test synchronous analysis."""
        result = pm.analyze_sync(code_prompt)
        assert "intent" in result

    # Pipeline Tests
    @pytest.mark.asyncio
    async def test_process_basic(self, pm, messy_prompt):
        """Test basic pipeline processing."""
        result = await pm.process(
            messy_prompt,
            compress=True,
            enhance=True,
            validate=True
        )
        assert result.success == True
        assert len(result.step_results) == 3

    @pytest.mark.asyncio
    async def test_process_enhance_only(self, pm, messy_prompt):
        """Test pipeline with enhancement only."""
        result = await pm.process(
            messy_prompt,
            compress=False,
            enhance=True,
            validate=False
        )
        assert len(result.step_results) == 1

    @pytest.mark.asyncio
    async def test_process_compress_only(self, pm, long_prompt):
        """Test pipeline with compression only."""
        result = await pm.process(
            long_prompt,
            compress=True,
            enhance=False,
            validate=False
        )
        assert len(result.step_results) == 1

    def test_process_sync(self, pm, messy_prompt):
        """Test synchronous pipeline processing."""
        result = pm.process_sync(messy_prompt)
        assert result is not None

    def test_create_pipeline(self, pm):
        """Test creating custom pipeline."""
        pipeline = pm.pipeline()
        assert pipeline is not None
        pipeline.enhance().compress().validate()
        assert len(pipeline.steps) == 3

    # Token Counting Tests
    def test_count_tokens(self, pm, short_prompt):
        """Test token counting."""
        count = pm.count_tokens(short_prompt)
        assert count > 0
        assert isinstance(count, int)

    def test_count_tokens_empty(self, pm):
        """Test token counting for empty string."""
        count = pm.count_tokens("")
        assert count == 0

    # Version Control Tests
    def test_save_and_get_prompt(self, pm):
        """Test saving and retrieving prompt."""
        pm.save_prompt(
            prompt_id="test_save",
            name="Test Save",
            content="Test content"
        )
        prompt = pm.get_prompt("test_save")
        assert prompt is not None
        assert prompt.content == "Test content"

    def test_list_prompts(self, pm):
        """Test listing saved prompts."""
        pm.save_prompt("list_test1", "List Test 1", "Content 1")
        pm.save_prompt("list_test2", "List Test 2", "Content 2")
        prompts = pm.list_prompts()
        assert len(prompts) >= 2


class TestCrossComponentIntegration:
    """Tests for cross-component interactions."""

    @pytest.fixture
    def pm(self):
        """Create a PromptManager instance."""
        return PromptManager()

    @pytest.mark.asyncio
    async def test_enhance_then_compress(self, pm, messy_prompt):
        """Test enhancing then compressing a prompt."""
        # Enhance first
        enhanced = await pm.enhance(messy_prompt, mode=EnhancementMode.RULES_ONLY)
        enhanced_text = enhanced.processed if isinstance(enhanced.processed, str) else str(enhanced.processed)

        # Then compress
        compressed = await pm.compress(enhanced_text, ratio=0.7)
        assert compressed is not None

    @pytest.mark.asyncio
    async def test_generate_then_validate(self, pm):
        """Test generating then validating a prompt."""
        # Generate a prompt
        generated = await pm.generate(task="Write a sorting algorithm")
        generated_text = generated.processed if isinstance(generated.processed, str) else str(generated.processed)

        # Validate it
        validation = pm.validate(generated_text)
        # Generated prompts should be valid
        assert validation.score > 0.5

    @pytest.mark.asyncio
    async def test_full_workflow(self, pm, messy_prompt):
        """Test full workflow: analyze -> enhance -> compress -> validate."""
        # 1. Analyze the prompt
        analysis = await pm.analyze(messy_prompt)
        assert analysis["quality"]["overall_score"] < 0.8  # Messy prompt should have low score

        # 2. Enhance it
        enhanced = await pm.enhance(messy_prompt, mode=EnhancementMode.RULES_ONLY)
        enhanced_text = enhanced.processed if isinstance(enhanced.processed, str) else str(enhanced.processed)

        # 3. Analyze enhanced version
        enhanced_analysis = await pm.analyze(enhanced_text)
        # Should have improved
        assert enhanced_analysis["quality"]["overall_score"] >= analysis["quality"]["overall_score"]

        # 4. Compress
        compressed = await pm.compress(enhanced_text, ratio=0.8)
        assert compressed.compressed_tokens <= compressed.original_tokens

        # 5. Validate final result
        compressed_text = compressed.processed.text if hasattr(compressed.processed, 'text') else str(compressed.processed)
        validation = pm.validate(compressed_text)
        assert validation.score > 0


class TestPromptManagerConfiguration:
    """Tests for PromptManager configuration."""

    def test_default_model(self):
        """Test default model configuration."""
        pm = PromptManager()
        assert pm.model == "gpt-4"

    def test_custom_model(self):
        """Test custom model configuration."""
        pm = PromptManager(model="gpt-3.5-turbo")
        assert pm.model == "gpt-3.5-turbo"

    def test_lazy_initialization(self):
        """Test lazy initialization of components."""
        pm = PromptManager()
        # Components should be None initially
        assert pm._compressor is None
        assert pm._enhancer is None
        assert pm._generator is None

        # Access should initialize them
        _ = pm.compressor
        assert pm._compressor is not None


class TestEdgeCases:
    """Integration tests for edge cases."""

    @pytest.fixture
    def pm(self):
        """Create a PromptManager instance."""
        return PromptManager()

    @pytest.mark.asyncio
    async def test_empty_prompt_workflow(self, pm):
        """Test handling empty prompt through workflow."""
        # Validation should fail
        validation = pm.validate("")
        assert validation.is_valid == False

    @pytest.mark.asyncio
    async def test_very_long_prompt_workflow(self, pm, long_prompt):
        """Test handling very long prompt through workflow."""
        very_long = long_prompt * 10

        # Should handle compression
        compressed = await pm.compress(very_long, ratio=0.3)
        assert compressed.compression_ratio < 1.0

    @pytest.mark.asyncio
    async def test_special_characters_workflow(self, pm):
        """Test handling special characters through workflow."""
        prompt = "Test @#$%^& special !!! characters ??? with $ymb0ls"

        # Should handle all operations
        enhanced = await pm.enhance(prompt, mode=EnhancementMode.RULES_ONLY)
        assert enhanced is not None

        validation = pm.validate(prompt)
        assert validation is not None

    @pytest.mark.asyncio
    async def test_unicode_workflow(self, pm):
        """Test handling unicode through workflow."""
        prompt = "Process these emojis:  and characters"

        enhanced = await pm.enhance(prompt, mode=EnhancementMode.RULES_ONLY)
        assert enhanced is not None

    @pytest.mark.asyncio
    async def test_multiline_workflow(self, pm):
        """Test handling multiline prompts through workflow."""
        prompt = """First line
        Second line
        Third line with code:
        ```python
        print("hello")
        ```
        End"""

        result = await pm.process(prompt)
        assert result is not None

    @pytest.mark.asyncio
    async def test_rapid_sequential_calls(self, pm, short_prompt):
        """Test rapid sequential calls."""
        for _ in range(10):
            result = await pm.compress(short_prompt, ratio=0.8)
            assert result is not None

    @pytest.mark.asyncio
    async def test_concurrent_calls(self, pm, short_prompt, messy_prompt, long_prompt):
        """Test concurrent calls."""
        tasks = [
            pm.compress(long_prompt, ratio=0.7),
            pm.enhance(messy_prompt, mode=EnhancementMode.RULES_ONLY),
            pm.analyze(short_prompt),
        ]
        results = await asyncio.gather(*tasks)
        assert len(results) == 3
        assert all(r is not None for r in results)
