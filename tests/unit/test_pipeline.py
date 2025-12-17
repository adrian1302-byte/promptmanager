"""Tests for pipeline module."""

import pytest
import asyncio
from promptmanager.pipeline import (
    Pipeline,
    PipelineResult,
    process_prompt,
)
from promptmanager.pipeline.pipeline import (
    PipelineStep,
    PipelineStepType,
    StepResult,
    create_compression_pipeline,
    create_enhancement_pipeline,
    create_full_pipeline,
)
from promptmanager.compression import StrategyType
from promptmanager.enhancement import EnhancementMode, EnhancementLevel
from promptmanager.core.exceptions import PipelineError


class TestPipeline:
    """Tests for Pipeline class."""

    @pytest.fixture
    def pipeline(self):
        """Create a basic pipeline."""
        return Pipeline()

    def test_create_pipeline(self, pipeline):
        """Test creating a pipeline."""
        assert pipeline is not None
        assert len(pipeline.steps) == 0

    def test_add_compress_step(self, pipeline):
        """Test adding compression step."""
        pipeline.compress(ratio=0.7, strategy=StrategyType.HYBRID)
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].step_type == PipelineStepType.COMPRESS

    def test_add_enhance_step(self, pipeline):
        """Test adding enhancement step."""
        pipeline.enhance(level=EnhancementLevel.MODERATE)
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].step_type == PipelineStepType.ENHANCE

    def test_add_validate_step(self, pipeline):
        """Test adding validation step."""
        pipeline.validate()
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].step_type == PipelineStepType.VALIDATE

    def test_fluent_interface(self, pipeline):
        """Test fluent interface chaining."""
        result = pipeline.enhance().compress().validate()
        assert result is pipeline
        assert len(pipeline.steps) == 3

    def test_add_custom_step(self, pipeline):
        """Test adding custom step."""
        def custom_handler(text, config):
            return text.upper()

        pipeline.custom("uppercase", custom_handler)
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].step_type == PipelineStepType.CUSTOM

    @pytest.mark.asyncio
    async def test_run_empty_pipeline(self, pipeline, short_prompt):
        """Test running empty pipeline."""
        result = await pipeline.run(short_prompt)
        assert result.success == True
        assert result.output_text == short_prompt

    @pytest.mark.asyncio
    async def test_run_compress_pipeline(self, long_prompt):
        """Test running compression pipeline."""
        pipeline = Pipeline().compress(ratio=0.7)
        result = await pipeline.run(long_prompt)
        assert result.success == True
        assert len(result.step_results) == 1

    @pytest.mark.asyncio
    async def test_run_enhance_pipeline(self, messy_prompt):
        """Test running enhancement pipeline."""
        pipeline = Pipeline().enhance(mode=EnhancementMode.RULES_ONLY)
        result = await pipeline.run(messy_prompt)
        assert result.success == True
        assert result.output_text != messy_prompt

    @pytest.mark.asyncio
    async def test_run_validate_pipeline(self, short_prompt):
        """Test running validation pipeline."""
        pipeline = Pipeline().validate(fail_on_error=False)
        result = await pipeline.run(short_prompt)
        assert result.success == True

    @pytest.mark.asyncio
    async def test_run_full_pipeline(self, messy_prompt):
        """Test running full pipeline."""
        pipeline = Pipeline().enhance().compress(ratio=0.8).validate(fail_on_error=False)
        result = await pipeline.run(messy_prompt)
        assert len(result.step_results) == 3

    @pytest.mark.asyncio
    async def test_run_custom_pipeline(self, short_prompt):
        """Test running pipeline with custom step."""
        def add_prefix(text, config):
            prefix = config.get("prefix", "PREFIX: ")
            return prefix + text

        pipeline = Pipeline().custom("add_prefix", add_prefix, prefix="CUSTOM: ")
        result = await pipeline.run(short_prompt)
        assert result.output_text.startswith("CUSTOM: ")

    @pytest.mark.asyncio
    async def test_async_custom_step(self, short_prompt):
        """Test async custom step handler."""
        async def async_handler(text, config):
            await asyncio.sleep(0.01)
            return text.upper()

        pipeline = Pipeline().custom("async_upper", async_handler)
        result = await pipeline.run(short_prompt)
        assert result.output_text == short_prompt.upper()

    def test_run_sync(self, short_prompt):
        """Test synchronous pipeline run."""
        pipeline = Pipeline().validate(fail_on_error=False)
        result = pipeline.run_sync(short_prompt)
        assert result.success == True

    def test_on_error_setting(self, pipeline):
        """Test setting error handling."""
        pipeline.compress().on_error("skip")
        assert pipeline.steps[0].on_error == "skip"

    def test_disable_step(self, pipeline):
        """Test disabling a step."""
        pipeline.compress()
        pipeline.disable_step("compress_0")
        assert pipeline.steps[0].enabled == False

    def test_enable_step(self, pipeline):
        """Test enabling a step."""
        pipeline.compress()
        pipeline.disable_step("compress_0")
        pipeline.enable_step("compress_0")
        assert pipeline.steps[0].enabled == True

    def test_list_steps(self, pipeline):
        """Test listing steps."""
        pipeline.enhance().compress().validate()
        steps = pipeline.list_steps()
        assert len(steps) == 3
        assert steps[0]["type"] == "enhance"

    def test_clear_steps(self, pipeline):
        """Test clearing steps."""
        pipeline.enhance().compress()
        pipeline.clear()
        assert len(pipeline.steps) == 0

    def test_clone_pipeline(self, pipeline):
        """Test cloning pipeline."""
        pipeline.enhance().compress()
        clone = pipeline.clone()
        assert len(clone.steps) == 2
        assert clone is not pipeline


class TestPipelineResult:
    """Tests for PipelineResult class."""

    def test_create_result(self):
        """Test creating pipeline result."""
        result = PipelineResult(
            success=True,
            input_text="input",
            output_text="output"
        )
        assert result.success == True
        assert result.input_text == "input"
        assert result.output_text == "output"

    def test_result_with_steps(self):
        """Test result with step results."""
        steps = [
            StepResult(
                step_name="test",
                step_type=PipelineStepType.ENHANCE,
                success=True,
                input_text="in",
                output_text="out"
            )
        ]
        result = PipelineResult(
            success=True,
            input_text="input",
            output_text="output",
            step_results=steps
        )
        assert len(result.step_results) == 1

    def test_to_processing_result(self):
        """Test converting to ProcessingResult."""
        result = PipelineResult(
            success=True,
            input_text="input",
            output_text="output",
            total_duration_ms=10.0
        )
        processing_result = result.to_processing_result()
        assert processing_result.metadata["pipeline_success"] == True


class TestStepResult:
    """Tests for StepResult class."""

    def test_create_step_result(self):
        """Test creating step result."""
        step = StepResult(
            step_name="compress_0",
            step_type=PipelineStepType.COMPRESS,
            success=True,
            input_text="input",
            output_text="output",
            duration_ms=5.0
        )
        assert step.step_name == "compress_0"
        assert step.success == True
        assert step.duration_ms == 5.0

    def test_step_result_with_error(self):
        """Test step result with error."""
        step = StepResult(
            step_name="compress_0",
            step_type=PipelineStepType.COMPRESS,
            success=False,
            input_text="input",
            output_text="input",
            error="Compression failed"
        )
        assert step.success == False
        assert step.error == "Compression failed"


class TestPipelineStep:
    """Tests for PipelineStep class."""

    def test_create_step(self):
        """Test creating pipeline step."""
        step = PipelineStep(
            step_type=PipelineStepType.COMPRESS,
            name="compress_0"
        )
        assert step.step_type == PipelineStepType.COMPRESS
        assert step.enabled == True

    def test_step_with_config(self):
        """Test step with configuration."""
        step = PipelineStep(
            step_type=PipelineStepType.COMPRESS,
            name="compress_0",
            config={"ratio": 0.5, "strategy": "hybrid"}
        )
        assert step.config["ratio"] == 0.5


class TestPipelineStepType:
    """Tests for PipelineStepType enum."""

    def test_step_type_values(self):
        """Test step type enum values."""
        assert PipelineStepType.COMPRESS.value == "compress"
        assert PipelineStepType.ENHANCE.value == "enhance"
        assert PipelineStepType.VALIDATE.value == "validate"
        assert PipelineStepType.CUSTOM.value == "custom"


class TestPipelineFactories:
    """Tests for pipeline factory functions."""

    def test_create_compression_pipeline(self):
        """Test creating compression pipeline."""
        pipeline = create_compression_pipeline(ratio=0.6)
        assert len(pipeline.steps) == 2  # compress + validate

    def test_create_enhancement_pipeline(self):
        """Test creating enhancement pipeline."""
        pipeline = create_enhancement_pipeline(level=EnhancementLevel.MODERATE)
        assert len(pipeline.steps) == 2  # enhance + validate

    def test_create_full_pipeline(self):
        """Test creating full pipeline."""
        pipeline = create_full_pipeline(
            compression_ratio=0.7,
            enhancement_level=EnhancementLevel.MODERATE
        )
        assert len(pipeline.steps) == 3  # enhance + compress + validate

    def test_create_full_pipeline_no_validate(self):
        """Test creating full pipeline without validation."""
        pipeline = create_full_pipeline(validate=False)
        assert len(pipeline.steps) == 2  # enhance + compress


class TestProcessPromptFunction:
    """Tests for process_prompt convenience function."""

    @pytest.mark.asyncio
    async def test_process_prompt_default(self, messy_prompt):
        """Test process_prompt with defaults."""
        result = await process_prompt(messy_prompt)
        assert result.success == True

    @pytest.mark.asyncio
    async def test_process_prompt_compress_only(self, long_prompt):
        """Test process_prompt with compression only."""
        result = await process_prompt(
            long_prompt,
            compress=True,
            enhance=False,
            validate=False
        )
        assert len(result.step_results) == 1

    @pytest.mark.asyncio
    async def test_process_prompt_enhance_only(self, messy_prompt):
        """Test process_prompt with enhancement only."""
        result = await process_prompt(
            messy_prompt,
            compress=False,
            enhance=True,
            validate=False
        )
        assert len(result.step_results) == 1

    @pytest.mark.asyncio
    async def test_process_prompt_custom_ratio(self, long_prompt):
        """Test process_prompt with custom compression ratio."""
        result = await process_prompt(
            long_prompt,
            compression_ratio=0.5
        )
        assert result.success == True


class TestErrorHandling:
    """Tests for pipeline error handling."""

    @pytest.mark.asyncio
    async def test_error_stop_behavior(self):
        """Test pipeline stops on error with 'stop' behavior."""
        def failing_handler(text, config):
            raise ValueError("Test error")

        pipeline = Pipeline()
        pipeline.custom("fail", failing_handler)
        pipeline.steps[0].on_error = "stop"
        pipeline.validate()

        result = await pipeline.run("test")
        assert result.success == False
        assert len(result.step_results) == 1  # Stopped after first step

    @pytest.mark.asyncio
    async def test_error_skip_behavior(self):
        """Test pipeline skips on error with 'skip' behavior."""
        def failing_handler(text, config):
            raise ValueError("Test error")

        pipeline = Pipeline()
        pipeline.custom("fail", failing_handler).on_error("skip")
        pipeline.validate(fail_on_error=False)

        result = await pipeline.run("Test prompt for validation")
        # Should continue after error
        assert len(result.step_results) == 2

    @pytest.mark.asyncio
    async def test_error_continue_behavior(self):
        """Test pipeline continues on error with 'continue' behavior."""
        def failing_handler(text, config):
            raise ValueError("Test error")

        pipeline = Pipeline()
        pipeline.custom("fail", failing_handler).on_error("continue")
        pipeline.validate(fail_on_error=False)

        result = await pipeline.run("Test prompt")
        assert len(result.step_results) == 2

    @pytest.mark.asyncio
    async def test_validation_failure(self):
        """Test pipeline with validation failure."""
        pipeline = Pipeline().validate(fail_on_error=True)
        result = await pipeline.run("")  # Empty prompt should fail validation
        assert result.success == False


class TestEdgeCases:
    """Tests for edge cases in pipeline."""

    @pytest.mark.asyncio
    async def test_empty_prompt(self):
        """Test pipeline with empty prompt."""
        pipeline = Pipeline().enhance().compress()
        result = await pipeline.run("")
        assert result is not None

    @pytest.mark.asyncio
    async def test_very_long_prompt(self, long_prompt):
        """Test pipeline with very long prompt."""
        very_long = long_prompt * 10
        pipeline = Pipeline().compress(ratio=0.3)
        result = await pipeline.run(very_long)
        assert result.success == True

    @pytest.mark.asyncio
    async def test_special_characters(self):
        """Test pipeline with special characters."""
        pipeline = Pipeline().enhance().validate(fail_on_error=False)
        result = await pipeline.run("Test @#$%^& special chars!")
        assert result is not None

    @pytest.mark.asyncio
    async def test_unicode_prompt(self):
        """Test pipeline with unicode."""
        pipeline = Pipeline().validate(fail_on_error=False)
        result = await pipeline.run("Test unicode  characters")
        assert result is not None

    @pytest.mark.asyncio
    async def test_multiline_prompt(self):
        """Test pipeline with multiline prompt."""
        prompt = """Line 1
        Line 2
        Line 3"""
        pipeline = Pipeline().enhance().compress()
        result = await pipeline.run(prompt)
        assert result is not None

    @pytest.mark.asyncio
    async def test_disabled_all_steps(self):
        """Test pipeline with all steps disabled."""
        pipeline = Pipeline()
        pipeline.enhance()
        pipeline.compress()
        pipeline.disable_step("enhance_0")
        pipeline.disable_step("compress_1")
        result = await pipeline.run("test")
        assert result.output_text == "test"
        assert len(result.step_results) == 0
