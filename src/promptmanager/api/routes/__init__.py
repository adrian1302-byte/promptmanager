"""API routes."""

from .compression import router as compression_router
from .enhancement import router as enhancement_router
from .generation import router as generation_router
from .validation import router as validation_router
from .pipeline import router as pipeline_router
from .health import router as health_router

__all__ = [
    "compression_router",
    "enhancement_router",
    "generation_router",
    "validation_router",
    "pipeline_router",
    "health_router",
]
