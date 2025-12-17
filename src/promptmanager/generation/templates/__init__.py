"""Template system for prompt generation."""

from .engine import TemplateEngine, TemplateMetadata
from .slot_filler import (
    SlotFiller,
    SlotDefinition,
    SlotType,
    SlotFillingResult,
    FilledSlot,
    get_template_slots,
    COMMON_SLOTS,
)

__all__ = [
    "TemplateEngine",
    "TemplateMetadata",
    "SlotFiller",
    "SlotDefinition",
    "SlotType",
    "SlotFillingResult",
    "FilledSlot",
    "get_template_slots",
    "COMMON_SLOTS",
]
