"""API request schemas."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class CompressRequest(BaseModel):
    """Request for prompt compression."""
    prompt: str = Field(..., description="The prompt to compress")
    target_ratio: float = Field(
        0.7,
        ge=0.1,
        le=1.0,
        description="Target compression ratio (0.1-1.0)"
    )
    strategy: str = Field(
        "hybrid",
        description="Compression strategy: lexical, statistical, code, hybrid"
    )
    preserve_code: bool = Field(
        True,
        description="Whether to preserve code blocks"
    )
    model: Optional[str] = Field(
        None,
        description="Model name for tokenization (default: gpt-4)"
    )


class EnhanceRequest(BaseModel):
    """Request for prompt enhancement."""
    prompt: str = Field(..., description="The prompt to enhance")
    mode: str = Field(
        "hybrid",
        description="Enhancement mode: rules_only, llm_only, hybrid, adaptive"
    )
    level: str = Field(
        "moderate",
        description="Enhancement level: minimal, light, moderate, aggressive"
    )
    detect_intent: bool = Field(
        True,
        description="Whether to detect prompt intent"
    )
    fix_grammar: bool = Field(
        True,
        description="Whether to fix grammar issues"
    )


class GenerateRequest(BaseModel):
    """Request for prompt generation."""
    task: str = Field(..., description="Task description")
    style: Optional[str] = Field(
        None,
        description="Prompt style: zero_shot, few_shot, chain_of_thought, code_generation"
    )
    template: Optional[str] = Field(
        None,
        description="Specific template to use"
    )
    examples: Optional[List[Dict[str, str]]] = Field(
        None,
        description="Examples for few-shot learning"
    )
    context: Optional[str] = Field(
        None,
        description="Additional context"
    )
    constraints: Optional[List[str]] = Field(
        None,
        description="Constraints to include"
    )
    language: Optional[str] = Field(
        None,
        description="Programming language for code generation"
    )
    variables: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional template variables"
    )


class ValidateRequest(BaseModel):
    """Request for prompt validation."""
    prompt: str = Field(..., description="The prompt to validate")
    categories: Optional[List[str]] = Field(
        None,
        description="Validation categories to check"
    )


class AnalyzeRequest(BaseModel):
    """Request for prompt analysis."""
    prompt: str = Field(..., description="The prompt to analyze")


class PipelineRequest(BaseModel):
    """Request for pipeline processing."""
    prompt: str = Field(..., description="The prompt to process")
    compress: bool = Field(True, description="Whether to compress")
    enhance: bool = Field(True, description="Whether to enhance")
    validate: bool = Field(True, description="Whether to validate")
    compression_ratio: float = Field(
        0.7,
        ge=0.1,
        le=1.0,
        description="Compression ratio"
    )
    enhancement_level: str = Field(
        "moderate",
        description="Enhancement level"
    )
