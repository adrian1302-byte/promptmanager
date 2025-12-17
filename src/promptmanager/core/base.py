"""Abstract base classes for prompt processors."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, TypeVar, Generic, Union
import time

from .types import (
    Prompt,
    ProcessingResult,
    CompressionResult,
    EnhancementResult,
    GenerationResult
)

T = TypeVar('T', bound=ProcessingResult)


class BaseProcessor(ABC, Generic[T]):
    """
    Abstract base class for all prompt processors.

    Provides common functionality for compression, enhancement, and generation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._validate_config()

    def _validate_config(self) -> None:
        """Override to validate configuration. Raises ConfigurationError if invalid."""
        pass

    @abstractmethod
    async def process(self, prompt: Prompt, **kwargs) -> T:
        """
        Process a prompt asynchronously.

        Args:
            prompt: The prompt to process
            **kwargs: Additional processing options

        Returns:
            Processing result of type T
        """
        pass

    def process_sync(self, prompt: Prompt, **kwargs) -> T:
        """
        Synchronous wrapper for process.

        Uses asyncio to run the async process method.
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.process(prompt, **kwargs)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(self.process(prompt, **kwargs))
        except RuntimeError:
            # No event loop exists, create one
            return asyncio.run(self.process(prompt, **kwargs))

    def _normalize_prompt(self, prompt: Union[str, Prompt]) -> Prompt:
        """Convert string to Prompt if needed."""
        if isinstance(prompt, str):
            return Prompt.from_text(prompt)
        return prompt

    def _measure_time(self, start_time: float) -> float:
        """Calculate elapsed time in milliseconds."""
        return (time.perf_counter() - start_time) * 1000


class Compressor(BaseProcessor[CompressionResult]):
    """Abstract base class for compression strategies."""

    name: str = "base_compressor"
    description: str = "Base compression strategy"

    @abstractmethod
    async def process(
        self,
        prompt: Prompt,
        target_ratio: float = 0.5,
        preserve_instruction: bool = True,
        **kwargs
    ) -> CompressionResult:
        """
        Compress a prompt.

        Args:
            prompt: The prompt to compress
            target_ratio: Target size as ratio of original (0.5 = 50%)
            preserve_instruction: Whether to preserve instruction portions
            **kwargs: Additional options

        Returns:
            CompressionResult with compressed prompt and metrics
        """
        pass

    @abstractmethod
    def estimate_compression(self, prompt: Prompt) -> float:
        """
        Estimate achievable compression ratio without actually compressing.

        Args:
            prompt: The prompt to analyze

        Returns:
            Estimated compression ratio (0-1)
        """
        pass

    def get_capabilities(self) -> Dict[str, Any]:
        """Return compressor capabilities and metadata."""
        return {
            "name": self.name,
            "description": self.description,
        }


class Enhancer(BaseProcessor[EnhancementResult]):
    """Abstract base class for enhancement strategies."""

    name: str = "base_enhancer"
    description: str = "Base enhancement strategy"

    @abstractmethod
    async def process(
        self,
        prompt: Prompt,
        level: str = "moderate",
        focus: Optional[list] = None,
        **kwargs
    ) -> EnhancementResult:
        """
        Enhance a prompt.

        Args:
            prompt: The prompt to enhance
            level: Enhancement intensity ("light", "moderate", "aggressive")
            focus: Specific areas to focus on
            **kwargs: Additional options

        Returns:
            EnhancementResult with enhanced prompt and improvements
        """
        pass

    @abstractmethod
    async def analyze_quality(self, prompt: Prompt) -> Dict[str, float]:
        """
        Analyze prompt quality without enhancement.

        Args:
            prompt: The prompt to analyze

        Returns:
            Dictionary of quality scores
        """
        pass


class Generator(BaseProcessor[GenerationResult]):
    """Abstract base class for prompt generation."""

    name: str = "base_generator"
    description: str = "Base prompt generator"

    @abstractmethod
    async def process(
        self,
        prompt: Prompt,
        task_type: Optional[str] = None,
        style: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate or transform a prompt.

        For generation, the input prompt contains the task description.

        Args:
            prompt: Prompt containing task description
            task_type: Type of task (code_generation, qa, etc.)
            style: Prompt style (zero_shot, few_shot, chain_of_thought)
            **kwargs: Additional options

        Returns:
            GenerationResult with generated prompt
        """
        pass

    @abstractmethod
    def list_templates(self) -> list:
        """List available templates."""
        pass

    @abstractmethod
    def list_styles(self) -> list:
        """List available prompt styles."""
        pass
