"""CLI commands."""

from .compress import compress, tokens
from .enhance import enhance, analyze
from .generate import generate, recommend, list_templates, list_styles

__all__ = [
    "compress",
    "tokens",
    "enhance",
    "analyze",
    "generate",
    "recommend",
    "list_templates",
    "list_styles",
]
