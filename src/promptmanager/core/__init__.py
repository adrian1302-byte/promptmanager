"""Core module - foundational types, base classes, and configuration."""

from .types import (
    Message,
    Prompt,
    PromptRole,
    ProcessingResult,
    CompressionResult,
    EnhancementResult,
    GenerationResult,
    ContentType,
)
from .base import BaseProcessor, Compressor, Enhancer, Generator
from .config import Settings, get_settings
from .exceptions import (
    PromptManagerError,
    CompressionError,
    EnhancementError,
    GenerationError,
    ProviderError,
    ValidationError,
    ConfigurationError,
    PipelineError,
)
from .registry import Registry, ProviderRegistry

__all__ = [
    # Types
    "Message",
    "Prompt",
    "PromptRole",
    "ProcessingResult",
    "CompressionResult",
    "EnhancementResult",
    "GenerationResult",
    "ContentType",
    # Base classes
    "BaseProcessor",
    "Compressor",
    "Enhancer",
    "Generator",
    # Configuration
    "Settings",
    "get_settings",
    # Exceptions
    "PromptManagerError",
    "CompressionError",
    "EnhancementError",
    "GenerationError",
    "ProviderError",
    "ValidationError",
    "ConfigurationError",
    "PipelineError",
    # Registry
    "Registry",
    "ProviderRegistry",
]
