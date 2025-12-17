"""Prompt generation module."""

from .generator import (
    PromptGenerator,
    GenerationConfig,
    DetailedGenerationResult,
    generate,
    generate_sync,
    generate_code_prompt,
    generate_code_prompt_sync,
    generate_cot_prompt,
    generate_cot_prompt_sync,
)
from .templates import (
    TemplateEngine,
    TemplateMetadata,
    SlotFiller,
    SlotDefinition,
    SlotType,
    SlotFillingResult,
    FilledSlot,
    get_template_slots,
    COMMON_SLOTS,
)
from .styles import (
    StyleRegistry,
    StyleDefinition,
    StyleRecommendation,
    PromptStyle,
    TASK_STYLE_MAP,
    get_recommended_style,
)

__all__ = [
    # Main orchestrator
    "PromptGenerator",
    "GenerationConfig",
    "DetailedGenerationResult",
    # Convenience functions
    "generate",
    "generate_sync",
    "generate_code_prompt",
    "generate_code_prompt_sync",
    "generate_cot_prompt",
    "generate_cot_prompt_sync",
    # Templates
    "TemplateEngine",
    "TemplateMetadata",
    "SlotFiller",
    "SlotDefinition",
    "SlotType",
    "SlotFillingResult",
    "FilledSlot",
    "get_template_slots",
    "COMMON_SLOTS",
    # Styles
    "StyleRegistry",
    "StyleDefinition",
    "StyleRecommendation",
    "PromptStyle",
    "TASK_STYLE_MAP",
    "get_recommended_style",
]
