"""Compression API routes."""

from fastapi import APIRouter, HTTPException
import time

from ..schemas import CompressRequest, CompressResponse, ErrorResponse
from ...compression import PromptCompressor, StrategyType

router = APIRouter(prefix="/compress", tags=["compression"])


@router.post(
    "",
    response_model=CompressResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def compress_prompt(request: CompressRequest) -> CompressResponse:
    """
    Compress a prompt to reduce token count.

    Supports multiple compression strategies:
    - **lexical**: Fast rule-based compression (stopwords, phrases)
    - **statistical**: TF-IDF based sentence importance ranking
    - **code**: Optimized for code-heavy prompts
    - **hybrid**: Adaptive multi-strategy (recommended)
    """
    start_time = time.time()

    try:
        # Validate strategy
        try:
            strategy = StrategyType(request.strategy)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid strategy: {request.strategy}. "
                       f"Valid options: lexical, statistical, code, hybrid"
            )

        # Create compressor
        compressor = PromptCompressor(model=request.model or "gpt-4")

        # Compress
        result = await compressor.compress(
            request.prompt,
            target_ratio=request.target_ratio,
            strategy=strategy,
            preserve_code=request.preserve_code
        )

        processing_time = (time.time() - start_time) * 1000

        return CompressResponse(
            success=True,
            original_prompt=request.prompt,
            compressed_prompt=result.compressed_text,
            original_tokens=result.original_tokens,
            compressed_tokens=result.compressed_tokens,
            compression_ratio=result.compression_ratio,
            strategy_used=result.strategy_used,
            processing_time_ms=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/estimate")
async def estimate_compression(request: CompressRequest) -> dict:
    """
    Estimate compression results without actually compressing.

    Returns expected token counts and compression ratio.
    """
    try:
        compressor = PromptCompressor(model=request.model or "gpt-4")

        estimate = await compressor.estimate_compression(
            request.prompt,
            target_ratio=request.target_ratio
        )

        return {
            "original_tokens": estimate["original_tokens"],
            "estimated_tokens": estimate["estimated_tokens"],
            "estimated_ratio": estimate["estimated_ratio"],
            "recommended_strategy": estimate["recommended_strategy"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tokens")
async def count_tokens(prompt: str, model: str = "gpt-4") -> dict:
    """Count tokens in a prompt."""
    try:
        compressor = PromptCompressor(model=model)
        count = compressor.count_tokens(prompt)

        return {
            "prompt_length": len(prompt),
            "token_count": count,
            "model": model
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
