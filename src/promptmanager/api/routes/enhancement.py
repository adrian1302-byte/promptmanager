"""Enhancement API routes."""

from fastapi import APIRouter, HTTPException
import time

from ..schemas import (
    EnhanceRequest,
    EnhanceResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    IntentScores,
    QualityScores,
    ErrorResponse,
)
from ...enhancement import PromptEnhancer, EnhancementMode, EnhancementLevel

router = APIRouter(prefix="/enhance", tags=["enhancement"])


@router.post(
    "",
    response_model=EnhanceResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def enhance_prompt(request: EnhanceRequest) -> EnhanceResponse:
    """
    Enhance a prompt for clarity and effectiveness.

    Enhancement modes:
    - **rules_only**: Fast deterministic rule-based improvements
    - **llm_only**: LLM-based refinement (requires provider)
    - **hybrid**: Rules first, then LLM if needed (recommended)
    - **adaptive**: Automatically choose based on quality score

    Enhancement levels:
    - **minimal**: Only critical fixes
    - **light**: Minor improvements
    - **moderate**: Balanced improvements (recommended)
    - **aggressive**: Comprehensive rewriting
    """
    start_time = time.time()

    try:
        # Validate mode
        try:
            mode = EnhancementMode(request.mode)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode: {request.mode}. "
                       f"Valid options: rules_only, llm_only, hybrid, adaptive"
            )

        # Validate level
        try:
            level = EnhancementLevel(request.level)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid level: {request.level}. "
                       f"Valid options: minimal, light, moderate, aggressive"
            )

        # Create enhancer (without LLM provider for API - rules only by default)
        enhancer = PromptEnhancer()

        # Enhance
        result = await enhancer.enhance(
            request.prompt,
            mode=mode,
            level=level
        )

        processing_time = (time.time() - start_time) * 1000

        return EnhanceResponse(
            success=True,
            original_prompt=request.prompt,
            enhanced_prompt=result.enhanced_prompt,
            detected_intent=result.detected_intent.value if result.detected_intent else None,
            intent_confidence=result.intent_confidence,
            quality_score=result.final_quality.overall_score if result.final_quality else 0.0,
            quality_improvement=result.quality_improvement,
            applied_rules=result.applied_rules,
            llm_enhanced=result.llm_enhanced,
            suggestions=result.suggestions,
            processing_time_ms=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={500: {"model": ErrorResponse}}
)
async def analyze_prompt(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Analyze a prompt without modifying it.

    Returns:
    - Intent detection (code_generation, qa, summarization, etc.)
    - Quality scores (clarity, structure, completeness, specificity, grammar)
    - Statistics (length, word count, sentence count)
    """
    try:
        enhancer = PromptEnhancer()
        analysis = await enhancer.analyze(request.prompt)

        return AnalyzeResponse(
            success=True,
            intent=IntentScores(
                primary=analysis["intent"]["primary"],
                confidence=analysis["intent"]["confidence"],
                all_scores=analysis["intent"]["all_scores"]
            ),
            quality=QualityScores(
                overall=analysis["quality"]["overall_score"],
                clarity=analysis["quality"]["clarity"],
                structure=analysis["quality"]["structure"],
                completeness=analysis["quality"]["completeness"],
                specificity=analysis["quality"]["specificity"],
                grammar=analysis["quality"]["grammar"],
                issues=analysis["quality"]["issues"],
                suggestions=analysis["quality"]["suggestions"]
            ),
            statistics=analysis["statistics"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rules")
async def list_enhancement_rules() -> dict:
    """List all available enhancement rules."""
    try:
        enhancer = PromptEnhancer()
        rules = enhancer.list_rules()

        return {
            "rules": rules,
            "total": len(rules)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
