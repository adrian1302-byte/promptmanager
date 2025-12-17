"""
PromptManager - Production-ready LLM Prompt Management

A comprehensive system for prompt Control, Enhancement, Compression, and Generation.

Basic Usage:
    >>> from promptmanager import PromptManager
    >>> pm = PromptManager()
    >>>
    >>> # Compress a prompt
    >>> result = await pm.compress("Your long prompt...", ratio=0.5)
    >>> print(result.compressed_text)
    >>>
    >>> # Enhance a prompt
    >>> result = await pm.enhance("messy prompt")
    >>> print(result.enhanced_prompt)
    >>>
    >>> # Generate a prompt
    >>> result = await pm.generate(task="Write a Python sorting function")
    >>> print(result.prompt)
    >>>
    >>> # Run a pipeline
    >>> result = await pm.process("prompt", compress=True, enhance=True)
    >>> print(result.output_text)

For more control, use the individual modules:
    - promptmanager.compression: Prompt compression strategies
    - promptmanager.enhancement: Prompt enhancement and analysis
    - promptmanager.generation: Prompt generation from tasks
    - promptmanager.control: Versioning and validation
    - promptmanager.pipeline: Composable processing pipelines
    - promptmanager.api: REST API server
    - promptmanager.cli: Command-line interface
"""

import asyncio
from typing import Optional, Dict, Any, List, Union

from .core.types import (
    Prompt,
    Message,
    ProcessingResult,
    CompressionResult,
    EnhancementResult,
    GenerationResult,
)
from .core.exceptions import (
    PromptManagerError,
    CompressionError,
    EnhancementError,
    GenerationError,
    ValidationError,
    PipelineError,
)
from .providers.base import LLMProvider
from .compression import PromptCompressor, StrategyType
from .enhancement import (
    PromptEnhancer,
    EnhancementMode,
    EnhancementLevel,
    PromptIntent,
)
from .generation import PromptGenerator, PromptStyle
from .control import PromptControlManager, PromptValidator, ValidationResult
from .pipeline import Pipeline, PipelineResult


__version__ = "1.0.0"
__all__ = [
    # Main class
    "PromptManager",
    # Core types
    "Prompt",
    "Message",
    "ProcessingResult",
    "CompressionResult",
    "EnhancementResult",
    "GenerationResult",
    # Exceptions
    "PromptManagerError",
    "CompressionError",
    "EnhancementError",
    "GenerationError",
    "ValidationError",
    "PipelineError",
    # Enums
    "StrategyType",
    "EnhancementMode",
    "EnhancementLevel",
    "PromptIntent",
    "PromptStyle",
    # Individual components (for advanced use)
    "PromptCompressor",
    "PromptEnhancer",
    "PromptGenerator",
    "PromptControlManager",
    "PromptValidator",
    "Pipeline",
    # Result types
    "ValidationResult",
    "PipelineResult",
]


class PromptManager:
    """
    Main interface for prompt management operations.

    Provides a unified API for compression, enhancement, generation,
    validation, and pipeline processing.

    Example:
        >>> pm = PromptManager()
        >>> result = await pm.compress("long prompt...", ratio=0.5)
        >>> result = await pm.enhance("messy prompt")
        >>> result = await pm.generate(task="Write a function")
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        model: str = "gpt-4",
        storage_path: Optional[str] = None,
    ):
        """
        Initialize the PromptManager.

        Args:
            llm_provider: Optional LLM provider for LLM-based enhancements
            model: Default model for tokenization
            storage_path: Path for version storage (None for in-memory)
        """
        self.llm_provider = llm_provider
        self.model = model

        # Initialize components lazily
        self._compressor: Optional[PromptCompressor] = None
        self._enhancer: Optional[PromptEnhancer] = None
        self._generator: Optional[PromptGenerator] = None
        self._controller: Optional[PromptControlManager] = None
        self._validator: Optional[PromptValidator] = None
        self._storage_path = storage_path

    @property
    def compressor(self) -> PromptCompressor:
        """Get or create the compressor instance."""
        if self._compressor is None:
            self._compressor = PromptCompressor(tokenizer=self.model)
        return self._compressor

    @property
    def enhancer(self) -> PromptEnhancer:
        """Get or create the enhancer instance."""
        if self._enhancer is None:
            self._enhancer = PromptEnhancer(llm_provider=self.llm_provider)
        return self._enhancer

    @property
    def generator(self) -> PromptGenerator:
        """Get or create the generator instance."""
        if self._generator is None:
            self._generator = PromptGenerator(llm_provider=self.llm_provider)
        return self._generator

    @property
    def controller(self) -> PromptControlManager:
        """Get or create the control manager instance."""
        if self._controller is None:
            from .control import ControlConfig
            config = ControlConfig(storage_path=self._storage_path)
            self._controller = PromptControlManager(config)
        return self._controller

    @property
    def validator(self) -> PromptValidator:
        """Get or create the validator instance."""
        if self._validator is None:
            self._validator = PromptValidator()
        return self._validator

    # Compression methods
    async def compress(
        self,
        prompt: str,
        ratio: float = 0.7,
        strategy: Union[str, StrategyType] = StrategyType.HYBRID,
        **kwargs
    ) -> CompressionResult:
        """
        Compress a prompt to reduce token count.

        Args:
            prompt: The prompt to compress
            ratio: Target compression ratio (0.1-1.0)
            strategy: Compression strategy (lexical, statistical, code, hybrid)
            **kwargs: Additional options

        Returns:
            CompressionResult with compressed text and metrics
        """
        if isinstance(strategy, str):
            strategy = StrategyType(strategy)

        # Compressor.compress is synchronous
        return self.compressor.compress(
            prompt,
            target_ratio=ratio,
            strategy=strategy,
            **kwargs
        )

    def compress_sync(
        self,
        prompt: str,
        ratio: float = 0.7,
        strategy: Union[str, StrategyType] = StrategyType.HYBRID,
        **kwargs
    ) -> CompressionResult:
        """Synchronous version of compress."""
        if isinstance(strategy, str):
            strategy = StrategyType(strategy)
        return self.compressor.compress(prompt, target_ratio=ratio, strategy=strategy, **kwargs)

    # Enhancement methods
    async def enhance(
        self,
        prompt: str,
        mode: Union[str, EnhancementMode] = EnhancementMode.RULES_ONLY,
        level: Union[str, EnhancementLevel] = EnhancementLevel.MODERATE,
        **kwargs
    ) -> EnhancementResult:
        """
        Enhance a prompt for clarity and effectiveness.

        Args:
            prompt: The prompt to enhance
            mode: Enhancement mode (rules_only, llm_only, hybrid, adaptive)
            level: Enhancement level (minimal, light, moderate, aggressive)
            **kwargs: Additional options

        Returns:
            EnhancementResult with enhanced prompt and improvements
        """
        if isinstance(mode, str):
            mode = EnhancementMode(mode)
        if isinstance(level, str):
            level = EnhancementLevel(level)

        result = await self.enhancer.enhance(prompt, mode=mode, level=level, **kwargs)
        return result.to_enhancement_result()

    def enhance_sync(
        self,
        prompt: str,
        mode: Union[str, EnhancementMode] = EnhancementMode.RULES_ONLY,
        level: Union[str, EnhancementLevel] = EnhancementLevel.MODERATE,
        **kwargs
    ) -> EnhancementResult:
        """Synchronous version of enhance."""
        return asyncio.run(self.enhance(prompt, mode, level, **kwargs))

    # Generation methods
    async def generate(
        self,
        task: str,
        style: Optional[Union[str, PromptStyle]] = None,
        examples: Optional[List[Dict[str, str]]] = None,
        context: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate an optimized prompt from a task description.

        Args:
            task: The task description
            style: Prompt style (zero_shot, few_shot, chain_of_thought, code_generation)
            examples: Examples for few-shot learning
            context: Additional context
            **kwargs: Additional template variables

        Returns:
            GenerationResult with generated prompt
        """
        if isinstance(style, str):
            style = PromptStyle(style)

        result = await self.generator.generate(
            task=task,
            style=style,
            examples=examples,
            context=context,
            **kwargs
        )
        return result.to_generation_result()

    def generate_sync(
        self,
        task: str,
        style: Optional[Union[str, PromptStyle]] = None,
        examples: Optional[List[Dict[str, str]]] = None,
        context: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """Synchronous version of generate."""
        return asyncio.run(self.generate(task, style, examples, context, **kwargs))

    # Validation methods
    def validate(self, prompt: str) -> ValidationResult:
        """
        Validate a prompt for security and quality issues.

        Args:
            prompt: The prompt to validate

        Returns:
            ValidationResult with validation status and issues
        """
        return self.validator.validate(prompt)

    # Analysis methods
    async def analyze(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze a prompt without modifying it.

        Args:
            prompt: The prompt to analyze

        Returns:
            Analysis results including intent and quality scores
        """
        return await self.enhancer.analyze(prompt)

    def analyze_sync(self, prompt: str) -> Dict[str, Any]:
        """Synchronous version of analyze."""
        return asyncio.run(self.analyze(prompt))

    # Pipeline methods
    def pipeline(self) -> Pipeline:
        """
        Create a new processing pipeline.

        Returns:
            Pipeline instance for chaining operations

        Example:
            >>> result = await pm.pipeline()\\
            ...     .enhance(level="moderate")\\
            ...     .compress(ratio=0.6)\\
            ...     .validate()\\
            ...     .run("Your prompt")
        """
        return Pipeline(llm_provider=self.llm_provider)

    async def process(
        self,
        prompt: str,
        compress: bool = True,
        enhance: bool = True,
        validate: bool = True,
        compression_ratio: float = 0.7,
        enhancement_level: Union[str, EnhancementLevel] = EnhancementLevel.MODERATE,
    ) -> PipelineResult:
        """
        Process a prompt through a configurable pipeline.

        Args:
            prompt: The prompt to process
            compress: Whether to compress
            enhance: Whether to enhance
            validate: Whether to validate
            compression_ratio: Target compression ratio
            enhancement_level: Enhancement level

        Returns:
            PipelineResult with processed prompt
        """
        from .pipeline import process_prompt

        if isinstance(enhancement_level, str):
            enhancement_level = EnhancementLevel(enhancement_level)

        return await process_prompt(
            prompt=prompt,
            compress=compress,
            enhance=enhance,
            validate=validate,
            compression_ratio=compression_ratio,
            enhancement_level=enhancement_level,
            llm_provider=self.llm_provider
        )

    def process_sync(
        self,
        prompt: str,
        compress: bool = True,
        enhance: bool = True,
        validate: bool = True,
        compression_ratio: float = 0.7,
        enhancement_level: Union[str, EnhancementLevel] = EnhancementLevel.MODERATE,
    ) -> PipelineResult:
        """Synchronous version of process."""
        return asyncio.run(self.process(
            prompt, compress, enhance, validate,
            compression_ratio, enhancement_level
        ))

    # Token counting
    def count_tokens(self, prompt: str) -> int:
        """
        Count tokens in a prompt.

        Args:
            prompt: The prompt to count

        Returns:
            Number of tokens
        """
        return self.compressor.count_tokens(prompt)

    # Version control methods
    def save_prompt(
        self,
        prompt_id: str,
        name: str,
        content: str,
        **kwargs
    ):
        """Save a prompt to the version control system."""
        return self.controller.create_prompt(prompt_id, name, content, **kwargs)

    def get_prompt(self, prompt_id: str, version: Optional[int] = None):
        """Get a saved prompt."""
        return self.controller.get_prompt(prompt_id, version)

    def list_prompts(self) -> List[Dict[str, Any]]:
        """List all saved prompts."""
        return self.controller.list_prompts()
