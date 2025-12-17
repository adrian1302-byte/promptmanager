"""Validation API routes."""

from fastapi import APIRouter, HTTPException

from ..schemas import ValidateRequest, ValidateResponse, ValidationIssueResponse, ErrorResponse
from ...control import PromptValidator, ValidationCategory

router = APIRouter(prefix="/validate", tags=["validation"])


@router.post(
    "",
    response_model=ValidateResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def validate_prompt(request: ValidateRequest) -> ValidateResponse:
    """
    Validate a prompt against security and quality rules.

    Categories:
    - **security**: Injection patterns, PII, API keys
    - **structure**: Length, emptiness, brackets
    - **content**: Template variables, completeness
    - **quality**: Clear instructions, formatting
    - **format**: Whitespace, punctuation
    """
    try:
        # Parse categories if provided
        categories = None
        if request.categories:
            try:
                categories = [ValidationCategory(c) for c in request.categories]
            except ValueError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid category: {e}. "
                           f"Valid options: security, structure, content, quality, format"
                )

        validator = PromptValidator()
        result = validator.validate(request.prompt, categories=categories)

        return ValidateResponse(
            success=True,
            is_valid=result.is_valid,
            score=result.score,
            issues=[
                ValidationIssueResponse(
                    rule=issue.rule_name,
                    message=issue.message,
                    severity=issue.severity.value,
                    category=issue.category.value,
                    position=issue.position,
                    suggestion=issue.suggestion
                )
                for issue in result.issues
            ],
            error_count=len(result.errors),
            warning_count=len(result.warnings)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rules")
async def list_validation_rules() -> dict:
    """List all available validation rules."""
    try:
        validator = PromptValidator()
        rules = validator.list_rules()

        return {
            "rules": rules,
            "total": len(rules)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories")
async def list_validation_categories() -> dict:
    """List all validation categories."""
    return {
        "categories": [
            {
                "name": "security",
                "description": "Security checks: injection patterns, PII, API keys"
            },
            {
                "name": "structure",
                "description": "Structural checks: length, emptiness, brackets"
            },
            {
                "name": "content",
                "description": "Content checks: template variables, completeness"
            },
            {
                "name": "quality",
                "description": "Quality checks: clear instructions, best practices"
            },
            {
                "name": "format",
                "description": "Format checks: whitespace, punctuation"
            }
        ]
    }
