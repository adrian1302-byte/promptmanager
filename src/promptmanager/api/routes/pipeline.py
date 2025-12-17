"""Pipeline API routes."""

from fastapi import APIRouter, HTTPException

from ..schemas import PipelineRequest, PipelineResponse, PipelineStepResponse, ErrorResponse
from ...pipeline import process_prompt
from ...enhancement import EnhancementLevel

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.post(
    "",
    response_model=PipelineResponse,
    responses={500: {"model": ErrorResponse}}
)
async def run_pipeline(request: PipelineRequest) -> PipelineResponse:
    """
    Process a prompt through a configurable pipeline.

    The pipeline can include:
    - **Compression**: Reduce token count while preserving meaning
    - **Enhancement**: Improve clarity and effectiveness
    - **Validation**: Check for security and quality issues

    Steps are executed in order: enhance -> compress -> validate
    """
    try:
        # Parse enhancement level
        try:
            level = EnhancementLevel(request.enhancement_level)
        except ValueError:
            level = EnhancementLevel.MODERATE

        result = await process_prompt(
            prompt=request.prompt,
            compress=request.compress,
            enhance=request.enhance,
            validate=request.validate,
            compression_ratio=request.compression_ratio,
            enhancement_level=level
        )

        return PipelineResponse(
            success=result.success,
            original_prompt=result.input_text,
            processed_prompt=result.output_text,
            steps=[
                PipelineStepResponse(
                    step_name=step.step_name,
                    step_type=step.step_type.value,
                    success=step.success,
                    duration_ms=step.duration_ms,
                    error=step.error,
                    metadata=step.metadata
                )
                for step in result.step_results
            ],
            total_duration_ms=result.total_duration_ms
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compress-enhance")
async def compress_and_enhance(
    prompt: str,
    compression_ratio: float = 0.7,
    enhancement_level: str = "moderate"
) -> dict:
    """
    Quick endpoint to compress and enhance a prompt.

    Runs enhancement first, then compression.
    """
    try:
        level = EnhancementLevel(enhancement_level)
    except ValueError:
        level = EnhancementLevel.MODERATE

    try:
        result = await process_prompt(
            prompt=prompt,
            compress=True,
            enhance=True,
            validate=False,
            compression_ratio=compression_ratio,
            enhancement_level=level
        )

        return {
            "success": result.success,
            "original": result.input_text,
            "processed": result.output_text,
            "duration_ms": result.total_duration_ms
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
