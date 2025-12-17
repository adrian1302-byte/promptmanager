"""Health check routes."""

from fastapi import APIRouter

from ..schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns the status of the API and its components.
    """
    # Check component availability
    components = {}

    # Check compression
    try:
        from ...compression import PromptCompressor
        PromptCompressor()
        components["compression"] = "healthy"
    except Exception:
        components["compression"] = "unavailable"

    # Check enhancement
    try:
        from ...enhancement import PromptEnhancer
        PromptEnhancer()
        components["enhancement"] = "healthy"
    except Exception:
        components["enhancement"] = "unavailable"

    # Check generation
    try:
        from ...generation import PromptGenerator
        PromptGenerator()
        components["generation"] = "healthy"
    except Exception:
        components["generation"] = "unavailable"

    # Check validation
    try:
        from ...control import PromptValidator
        PromptValidator()
        components["validation"] = "healthy"
    except Exception:
        components["validation"] = "unavailable"

    # Determine overall status
    all_healthy = all(v == "healthy" for v in components.values())
    status = "healthy" if all_healthy else "degraded"

    return HealthResponse(
        status=status,
        version="1.0.0",
        components=components
    )


@router.get("/")
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "name": "PromptManager API",
        "version": "1.0.0",
        "description": "Production-ready prompt management: Control, Enhance, Compress, Generate",
        "docs": "/docs",
        "endpoints": {
            "compress": "/api/v1/compress",
            "enhance": "/api/v1/enhance",
            "generate": "/api/v1/generate",
            "validate": "/api/v1/validate",
            "pipeline": "/api/v1/pipeline",
            "health": "/health"
        }
    }
