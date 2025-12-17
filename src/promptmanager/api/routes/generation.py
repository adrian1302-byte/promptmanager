"""Generation API routes."""

from fastapi import APIRouter, HTTPException
import time

from ..schemas import GenerateRequest, GenerateResponse, ErrorResponse
from ...generation import PromptGenerator, PromptStyle

router = APIRouter(prefix="/generate", tags=["generation"])


@router.post(
    "",
    response_model=GenerateResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def generate_prompt(request: GenerateRequest) -> GenerateResponse:
    """
    Generate an optimized prompt from a task description.

    Styles:
    - **zero_shot**: Direct instruction without examples
    - **few_shot**: Task with examples for in-context learning
    - **chain_of_thought**: Step-by-step reasoning for complex tasks
    - **code_generation**: Structured prompt for coding tasks

    The system can automatically select the best style based on the task.
    """
    start_time = time.time()

    try:
        # Validate style if provided
        style = None
        if request.style:
            try:
                style = PromptStyle(request.style)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid style: {request.style}. "
                           f"Valid options: zero_shot, few_shot, chain_of_thought, code_generation"
                )

        # Create generator
        generator = PromptGenerator()

        # Build variables
        variables = request.variables or {}
        if request.language:
            variables["language"] = request.language

        # Generate
        result = await generator.generate(
            task=request.task,
            style=style,
            template=request.template,
            variables=variables,
            examples=request.examples,
            context=request.context,
            constraints=request.constraints
        )

        processing_time = (time.time() - start_time) * 1000

        return GenerateResponse(
            success=True,
            generated_prompt=result.prompt,
            template_used=result.template_used,
            style_used=result.style_used.value,
            style_confidence=result.style_recommendation.confidence if result.style_recommendation else 1.0,
            variables_used=result.variables_used,
            warnings=result.warnings,
            processing_time_ms=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quick")
async def quick_generate(task: str, style: str = None) -> dict:
    """
    Quick prompt generation with minimal configuration.

    Just provide a task description and optionally a style hint.
    """
    try:
        generator = PromptGenerator()

        prompt_style = None
        if style:
            try:
                prompt_style = PromptStyle(style)
            except ValueError:
                pass  # Let auto-select handle it

        prompt = await generator.quick_generate(task, style=prompt_style)

        return {
            "task": task,
            "generated_prompt": prompt
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates")
async def list_templates() -> dict:
    """List all available prompt templates."""
    try:
        generator = PromptGenerator()
        templates = generator.list_templates()

        return {
            "templates": templates,
            "total": len(templates)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/styles")
async def list_styles() -> dict:
    """List all available prompt styles."""
    try:
        generator = PromptGenerator()
        styles = generator.list_styles()

        return {
            "styles": styles,
            "total": len(styles)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommend-style")
async def recommend_style(task: str, has_examples: bool = False, complexity: float = 0.5) -> dict:
    """
    Get a style recommendation for a task.

    Parameters:
    - task: The task description
    - has_examples: Whether you have examples available
    - complexity: Estimated task complexity (0-1)
    """
    try:
        generator = PromptGenerator()
        recommendation = generator.recommend_style(
            task=task,
            has_examples=has_examples,
            complexity=complexity
        )

        return {
            "recommended_style": recommendation.style.value,
            "template": recommendation.template_name,
            "confidence": recommendation.confidence,
            "reasoning": recommendation.reasoning,
            "alternatives": [s.value for s in recommendation.alternative_styles]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
