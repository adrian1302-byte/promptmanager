"""Prompt style management."""

from .style_registry import (
    StyleRegistry,
    StyleDefinition,
    StyleRecommendation,
    PromptStyle,
    TASK_STYLE_MAP,
    get_recommended_style,
)

__all__ = [
    "StyleRegistry",
    "StyleDefinition",
    "StyleRecommendation",
    "PromptStyle",
    "TASK_STYLE_MAP",
    "get_recommended_style",
]
