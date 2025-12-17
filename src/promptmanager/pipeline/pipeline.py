"""Composable pipeline orchestration for prompt processing."""

import asyncio
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union, Callable, Awaitable
from enum import Enum

from ..core.types import ProcessingResult
from ..core.exceptions import PipelineError
from ..providers.base import LLMProvider

from ..compression import PromptCompressor, StrategyType
from ..enhancement import PromptEnhancer, EnhancementMode, EnhancementLevel
from ..generation import PromptGenerator, PromptStyle
from ..control import PromptValidator, ValidationResult


class PipelineStepType(Enum):
    """Types of pipeline steps."""
    COMPRESS = "compress"
    ENHANCE = "enhance"
    GENERATE = "generate"
    VALIDATE = "validate"
    CUSTOM = "custom"


@dataclass
class PipelineStep:
    """A single step in the pipeline."""
    step_type: PipelineStepType
    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    on_error: str = "stop"  # "stop", "skip", "continue"


@dataclass
class StepResult:
    """Result from a single pipeline step."""
    step_name: str
    step_type: PipelineStepType
    success: bool
    input_text: str
    output_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class PipelineResult:
    """Result from running a complete pipeline."""
    success: bool
    input_text: str
    output_text: str
    step_results: List[StepResult] = field(default_factory=list)
    total_duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_processing_result(self) -> ProcessingResult:
        """Convert to standard ProcessingResult."""
        return ProcessingResult(
            original=self.input_text,
            processed=self.output_text,
            metadata={
                "pipeline_success": self.success,
                "steps_completed": len([s for s in self.step_results if s.success]),
                "total_steps": len(self.step_results),
                "total_duration_ms": self.total_duration_ms,
                **self.metadata
            }
        )


class Pipeline:
    """
    Composable pipeline for prompt processing.

    Allows chaining compress, enhance, validate, and custom
    operations in a fluent interface.
    """

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """
        Initialize the pipeline.

        Args:
            llm_provider: Optional LLM provider for operations that need it
        """
        self.llm_provider = llm_provider
        self.steps: List[PipelineStep] = []

        # Lazy-initialized processors
        self._compressor: Optional[PromptCompressor] = None
        self._enhancer: Optional[PromptEnhancer] = None
        self._generator: Optional[PromptGenerator] = None
        self._validator: Optional[PromptValidator] = None

        # Custom step handlers
        self._custom_handlers: Dict[str, Callable] = {}

    # Fluent interface for building pipelines
    def compress(
        self,
        ratio: float = 0.7,
        strategy: Union[str, StrategyType] = StrategyType.HYBRID,
        **kwargs
    ) -> "Pipeline":
        """
        Add a compression step.

        Args:
            ratio: Target compression ratio
            strategy: Compression strategy
            **kwargs: Additional compression options

        Returns:
            Self for chaining
        """
        if isinstance(strategy, str):
            strategy = StrategyType(strategy)

        self.steps.append(PipelineStep(
            step_type=PipelineStepType.COMPRESS,
            name=f"compress_{len(self.steps)}",
            config={
                "ratio": ratio,
                "strategy": strategy.value,
                **kwargs
            }
        ))
        return self

    def enhance(
        self,
        mode: Union[str, EnhancementMode] = EnhancementMode.HYBRID,
        level: Union[str, EnhancementLevel] = EnhancementLevel.MODERATE,
        **kwargs
    ) -> "Pipeline":
        """
        Add an enhancement step.

        Args:
            mode: Enhancement mode
            level: Enhancement level
            **kwargs: Additional enhancement options

        Returns:
            Self for chaining
        """
        if isinstance(mode, str):
            mode = EnhancementMode(mode)
        if isinstance(level, str):
            level = EnhancementLevel(level)

        self.steps.append(PipelineStep(
            step_type=PipelineStepType.ENHANCE,
            name=f"enhance_{len(self.steps)}",
            config={
                "mode": mode.value,
                "level": level.value,
                **kwargs
            }
        ))
        return self

    def validate(
        self,
        fail_on_error: bool = True,
        **kwargs
    ) -> "Pipeline":
        """
        Add a validation step.

        Args:
            fail_on_error: Whether to fail pipeline on validation errors
            **kwargs: Additional validation options

        Returns:
            Self for chaining
        """
        self.steps.append(PipelineStep(
            step_type=PipelineStepType.VALIDATE,
            name=f"validate_{len(self.steps)}",
            config={
                "fail_on_error": fail_on_error,
                **kwargs
            },
            on_error="stop" if fail_on_error else "continue"
        ))
        return self

    def custom(
        self,
        name: str,
        handler: Callable[[str, Dict[str, Any]], Union[str, Awaitable[str]]],
        **kwargs
    ) -> "Pipeline":
        """
        Add a custom processing step.

        Args:
            name: Step name
            handler: Function that takes (text, config) and returns processed text
            **kwargs: Configuration for the handler

        Returns:
            Self for chaining
        """
        self._custom_handlers[name] = handler
        self.steps.append(PipelineStep(
            step_type=PipelineStepType.CUSTOM,
            name=name,
            config=kwargs
        ))
        return self

    def on_error(self, action: str) -> "Pipeline":
        """
        Set error handling for the last added step.

        Args:
            action: "stop", "skip", or "continue"

        Returns:
            Self for chaining
        """
        if self.steps:
            self.steps[-1].on_error = action
        return self

    def disable_step(self, step_name: str) -> "Pipeline":
        """Disable a step by name."""
        for step in self.steps:
            if step.name == step_name:
                step.enabled = False
                break
        return self

    def enable_step(self, step_name: str) -> "Pipeline":
        """Enable a step by name."""
        for step in self.steps:
            if step.name == step_name:
                step.enabled = True
                break
        return self

    # Execution
    async def run(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """
        Run the pipeline on a prompt.

        Args:
            prompt: Input prompt
            context: Additional context for processing

        Returns:
            PipelineResult with all step results
        """
        import time
        start_time = time.time()

        context = context or {}
        current_text = prompt
        step_results: List[StepResult] = []
        success = True

        for step in self.steps:
            if not step.enabled:
                continue

            step_start = time.time()

            try:
                result = await self._execute_step(step, current_text, context)

                step_results.append(StepResult(
                    step_name=step.name,
                    step_type=step.step_type,
                    success=True,
                    input_text=current_text,
                    output_text=result["text"],
                    metadata=result.get("metadata", {}),
                    duration_ms=(time.time() - step_start) * 1000
                ))

                current_text = result["text"]

            except Exception as e:
                error_msg = str(e)

                step_results.append(StepResult(
                    step_name=step.name,
                    step_type=step.step_type,
                    success=False,
                    input_text=current_text,
                    output_text=current_text,  # Keep original on error
                    error=error_msg,
                    duration_ms=(time.time() - step_start) * 1000
                ))

                if step.on_error == "stop":
                    success = False
                    break
                elif step.on_error == "skip":
                    continue
                # "continue" - proceed with unchanged text

        total_duration = (time.time() - start_time) * 1000

        return PipelineResult(
            success=success,
            input_text=prompt,
            output_text=current_text,
            step_results=step_results,
            total_duration_ms=total_duration,
            metadata={
                "steps_executed": len(step_results),
                "context": context
            }
        )

    def run_sync(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """Synchronous version of run."""
        return asyncio.run(self.run(prompt, context))

    async def _execute_step(
        self,
        step: PipelineStep,
        text: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single pipeline step."""

        if step.step_type == PipelineStepType.COMPRESS:
            return await self._run_compress(text, step.config)

        elif step.step_type == PipelineStepType.ENHANCE:
            return await self._run_enhance(text, step.config)

        elif step.step_type == PipelineStepType.VALIDATE:
            return await self._run_validate(text, step.config)

        elif step.step_type == PipelineStepType.CUSTOM:
            return await self._run_custom(step.name, text, step.config)

        else:
            raise PipelineError(f"Unknown step type: {step.step_type}")

    async def _run_compress(
        self,
        text: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run compression step."""
        if not self._compressor:
            self._compressor = PromptCompressor()

        strategy = StrategyType(config.get("strategy", "hybrid"))
        ratio = config.get("ratio", 0.7)

        # compress() is synchronous
        result = self._compressor.compress(
            text,
            target_ratio=ratio,
            strategy=strategy
        )

        # Get text from processed (Prompt object)
        if hasattr(result.processed, 'text'):
            compressed_text = result.processed.text
        elif hasattr(result.processed, 'content'):
            compressed_text = result.processed.content
        else:
            compressed_text = str(result.processed)

        return {
            "text": compressed_text,
            "metadata": {
                "original_tokens": result.original_tokens,
                "compressed_tokens": result.compressed_tokens,
                "compression_ratio": result.compression_ratio,
                "strategy": result.strategy_used
            }
        }

    async def _run_enhance(
        self,
        text: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run enhancement step."""
        if not self._enhancer:
            self._enhancer = PromptEnhancer(llm_provider=self.llm_provider)

        mode = EnhancementMode(config.get("mode", "hybrid"))
        level = EnhancementLevel(config.get("level", "moderate"))

        result = await self._enhancer.enhance(text, mode=mode, level=level)

        return {
            "text": result.enhanced_prompt,
            "metadata": {
                "applied_rules": result.applied_rules,
                "llm_enhanced": result.llm_enhanced,
                "quality_improvement": result.quality_improvement
            }
        }

    async def _run_validate(
        self,
        text: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run validation step."""
        if not self._validator:
            self._validator = PromptValidator()

        result = self._validator.validate(text)

        if config.get("fail_on_error", True) and not result.is_valid:
            errors = [i.message for i in result.errors]
            raise PipelineError(f"Validation failed: {'; '.join(errors)}")

        return {
            "text": text,  # Validation doesn't modify text
            "metadata": {
                "is_valid": result.is_valid,
                "score": result.score,
                "issues": [
                    {"message": i.message, "severity": i.severity.value}
                    for i in result.issues
                ]
            }
        }

    async def _run_custom(
        self,
        name: str,
        text: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run custom step."""
        handler = self._custom_handlers.get(name)
        if not handler:
            raise PipelineError(f"Custom handler '{name}' not found")

        result = handler(text, config)

        # Handle async handlers
        if asyncio.iscoroutine(result):
            result = await result

        if isinstance(result, str):
            return {"text": result, "metadata": {}}
        elif isinstance(result, dict):
            return result
        else:
            raise PipelineError(f"Custom handler must return str or dict, got {type(result)}")

    # Pipeline info
    def list_steps(self) -> List[Dict[str, Any]]:
        """List all pipeline steps."""
        return [
            {
                "name": step.name,
                "type": step.step_type.value,
                "enabled": step.enabled,
                "config": step.config,
                "on_error": step.on_error
            }
            for step in self.steps
        ]

    def clear(self) -> "Pipeline":
        """Clear all steps."""
        self.steps = []
        return self

    def clone(self) -> "Pipeline":
        """Create a copy of this pipeline."""
        new_pipeline = Pipeline(llm_provider=self.llm_provider)
        new_pipeline.steps = [
            PipelineStep(
                step_type=s.step_type,
                name=s.name,
                config=s.config.copy(),
                enabled=s.enabled,
                on_error=s.on_error
            )
            for s in self.steps
        ]
        new_pipeline._custom_handlers = self._custom_handlers.copy()
        return new_pipeline


# Pre-built pipeline templates
def create_compression_pipeline(ratio: float = 0.6) -> Pipeline:
    """Create a simple compression pipeline."""
    return Pipeline().compress(ratio=ratio).validate(fail_on_error=False)


def create_enhancement_pipeline(
    level: EnhancementLevel = EnhancementLevel.MODERATE
) -> Pipeline:
    """Create an enhancement pipeline."""
    return Pipeline().enhance(level=level).validate(fail_on_error=False)


def create_full_pipeline(
    compression_ratio: float = 0.7,
    enhancement_level: EnhancementLevel = EnhancementLevel.MODERATE,
    validate: bool = True
) -> Pipeline:
    """Create a full processing pipeline."""
    pipeline = Pipeline()
    pipeline.enhance(level=enhancement_level)
    pipeline.compress(ratio=compression_ratio)

    if validate:
        pipeline.validate(fail_on_error=False)

    return pipeline


# Convenience function
async def process_prompt(
    prompt: str,
    compress: bool = True,
    enhance: bool = True,
    validate: bool = True,
    compression_ratio: float = 0.7,
    enhancement_level: EnhancementLevel = EnhancementLevel.MODERATE,
    llm_provider: Optional[LLMProvider] = None
) -> PipelineResult:
    """
    Process a prompt through a configurable pipeline.

    Args:
        prompt: Input prompt
        compress: Whether to compress
        enhance: Whether to enhance
        validate: Whether to validate
        compression_ratio: Target compression ratio
        enhancement_level: Enhancement level
        llm_provider: Optional LLM provider

    Returns:
        PipelineResult
    """
    pipeline = Pipeline(llm_provider=llm_provider)

    if enhance:
        pipeline.enhance(level=enhancement_level)

    if compress:
        pipeline.compress(ratio=compression_ratio)

    if validate:
        pipeline.validate(fail_on_error=False)

    return await pipeline.run(prompt)


def process_prompt_sync(
    prompt: str,
    compress: bool = True,
    enhance: bool = True,
    validate: bool = True,
    compression_ratio: float = 0.7,
    enhancement_level: EnhancementLevel = EnhancementLevel.MODERATE,
    llm_provider: Optional[LLMProvider] = None
) -> PipelineResult:
    """Synchronous version of process_prompt."""
    return asyncio.run(process_prompt(
        prompt, compress, enhance, validate,
        compression_ratio, enhancement_level, llm_provider
    ))
