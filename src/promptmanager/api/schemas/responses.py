"""API response schemas."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class CompressResponse(BaseModel):
    """Response for prompt compression."""
    success: bool
    original_prompt: str
    compressed_prompt: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    strategy_used: str
    processing_time_ms: float


class EnhanceResponse(BaseModel):
    """Response for prompt enhancement."""
    success: bool
    original_prompt: str
    enhanced_prompt: str
    detected_intent: Optional[str] = None
    intent_confidence: float = 0.0
    quality_score: float = 0.0
    quality_improvement: float = 0.0
    applied_rules: List[str] = Field(default_factory=list)
    llm_enhanced: bool = False
    suggestions: List[str] = Field(default_factory=list)
    processing_time_ms: float = 0.0


class GenerateResponse(BaseModel):
    """Response for prompt generation."""
    success: bool
    generated_prompt: str
    template_used: str
    style_used: str
    style_confidence: float = 0.0
    variables_used: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    processing_time_ms: float = 0.0


class ValidationIssueResponse(BaseModel):
    """A single validation issue."""
    rule: str
    message: str
    severity: str
    category: str
    position: Optional[int] = None
    suggestion: Optional[str] = None


class ValidateResponse(BaseModel):
    """Response for prompt validation."""
    success: bool
    is_valid: bool
    score: float
    issues: List[ValidationIssueResponse] = Field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0


class IntentScores(BaseModel):
    """Intent detection scores."""
    primary: str
    confidence: float
    all_scores: Dict[str, float] = Field(default_factory=dict)


class QualityScores(BaseModel):
    """Quality assessment scores."""
    overall: float
    clarity: float
    structure: float
    completeness: float
    specificity: float
    grammar: float
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    """Response for prompt analysis."""
    success: bool
    intent: IntentScores
    quality: QualityScores
    statistics: Dict[str, Any] = Field(default_factory=dict)


class PipelineStepResponse(BaseModel):
    """Response for a single pipeline step."""
    step_name: str
    step_type: str
    success: bool
    duration_ms: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PipelineResponse(BaseModel):
    """Response for pipeline processing."""
    success: bool
    original_prompt: str
    processed_prompt: str
    steps: List[PipelineStepResponse] = Field(default_factory=list)
    total_duration_ms: float = 0.0


class HealthResponse(BaseModel):
    """Response for health check."""
    status: str
    version: str
    components: Dict[str, str] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Error response."""
    success: bool = False
    error: str
    detail: Optional[str] = None
