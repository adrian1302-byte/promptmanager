"""FastAPI application factory."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import (
    compression_router,
    enhancement_router,
    generation_router,
    validation_router,
    pipeline_router,
    health_router,
)


def create_app(
    title: str = "PromptManager API",
    description: str = "Production-ready prompt management: Control, Enhance, Compress, Generate",
    version: str = "1.0.0",
    enable_cors: bool = True,
    cors_origins: list = None
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        title: API title
        description: API description
        version: API version
        enable_cors: Whether to enable CORS
        cors_origins: Allowed CORS origins (default: ["*"])

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Configure CORS
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins or ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Include routers
    app.include_router(health_router)
    app.include_router(compression_router, prefix="/api/v1")
    app.include_router(enhancement_router, prefix="/api/v1")
    app.include_router(generation_router, prefix="/api/v1")
    app.include_router(validation_router, prefix="/api/v1")
    app.include_router(pipeline_router, prefix="/api/v1")

    return app


# Default app instance
app = create_app()


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1
) -> None:
    """
    Run the API server.

    Args:
        host: Host to bind to
        port: Port to listen on
        reload: Enable auto-reload
        workers: Number of worker processes
    """
    import uvicorn

    uvicorn.run(
        "promptmanager.api.app:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers
    )


if __name__ == "__main__":
    run_server()
