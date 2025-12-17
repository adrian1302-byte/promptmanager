"""Core type definitions for the prompt management system."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from enum import Enum
from datetime import datetime
import uuid


class PromptRole(Enum):
    """Role of a message in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ContentType(Enum):
    """Type of content in a prompt."""
    NATURAL_LANGUAGE = "natural_language"
    CODE = "code"
    STRUCTURED_DATA = "structured_data"
    MIXED = "mixed"
    CONVERSATION = "conversation"


@dataclass
class Message:
    """Individual message in a prompt conversation."""
    role: PromptRole
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format for LLM APIs."""
        return {"role": self.role.value, "content": self.content}

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "Message":
        """Create from dictionary format."""
        return cls(
            role=PromptRole(data["role"]),
            content=data["content"],
            metadata=data.get("metadata", {})
        )


@dataclass
class Prompt:
    """
    Core prompt representation - the fundamental data structure.

    Supports both single-message and multi-message (conversation) prompts.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    content_type: ContentType = ContentType.NATURAL_LANGUAGE

    @classmethod
    def from_text(
        cls,
        text: str,
        role: PromptRole = PromptRole.USER,
        system_message: Optional[str] = None
    ) -> "Prompt":
        """Create a prompt from plain text."""
        messages = []
        if system_message:
            messages.append(Message(role=PromptRole.SYSTEM, content=system_message))
        messages.append(Message(role=role, content=text))
        return cls(messages=messages)

    @classmethod
    def from_messages(cls, messages: List[Dict[str, str]]) -> "Prompt":
        """Create from list of message dictionaries (OpenAI format)."""
        return cls(messages=[Message.from_dict(m) for m in messages])

    @property
    def text(self) -> str:
        """Get concatenated text content from all messages."""
        return "\n\n".join(m.content for m in self.messages)

    @property
    def user_content(self) -> str:
        """Get content from user messages only."""
        return "\n\n".join(
            m.content for m in self.messages
            if m.role == PromptRole.USER
        )

    @property
    def system_content(self) -> Optional[str]:
        """Get system message content if present."""
        for m in self.messages:
            if m.role == PromptRole.SYSTEM:
                return m.content
        return None

    def to_openai_format(self) -> List[Dict[str, str]]:
        """Convert to OpenAI message format."""
        return [m.to_dict() for m in self.messages]

    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic message format."""
        system = None
        messages = []

        for m in self.messages:
            if m.role == PromptRole.SYSTEM:
                system = m.content
            else:
                messages.append(m.to_dict())

        result = {"messages": messages}
        if system:
            result["system"] = system
        return result

    def copy(self) -> "Prompt":
        """Create a deep copy of the prompt."""
        return Prompt(
            id=str(uuid.uuid4()),
            messages=[Message(role=m.role, content=m.content, metadata=m.metadata.copy())
                     for m in self.messages],
            metadata=self.metadata.copy(),
            content_type=self.content_type
        )


@dataclass
class TokenInfo:
    """Token counting information."""
    original_count: int
    processed_count: int
    tokenizer_name: str

    @property
    def reduction(self) -> int:
        """Number of tokens reduced."""
        return self.original_count - self.processed_count

    @property
    def reduction_ratio(self) -> float:
        """Ratio of tokens reduced (0-1)."""
        if self.original_count == 0:
            return 0.0
        return 1 - (self.processed_count / self.original_count)


@dataclass
class QualityMetrics:
    """Quality metrics for prompt processing."""
    clarity_score: float = 0.0  # 0-1
    structure_score: float = 0.0  # 0-1
    completeness_score: float = 0.0  # 0-1
    specificity_score: float = 0.0  # 0-1
    overall_score: float = 0.0  # 0-1
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingResult:
    """Base result from any prompt processing operation."""
    original: Prompt
    processed: Prompt
    success: bool = True
    error: Optional[str] = None
    processing_time_ms: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionResult(ProcessingResult):
    """Result from compression operations."""
    original_tokens: int = 0
    compressed_tokens: int = 0
    compression_ratio: float = 0.0
    strategy_used: str = ""
    quality_preserved: float = 1.0

    def __post_init__(self):
        if self.original_tokens > 0 and self.compression_ratio == 0.0:
            self.compression_ratio = 1 - (self.compressed_tokens / self.original_tokens)

    @property
    def tokens_saved(self) -> int:
        """Number of tokens saved."""
        return self.original_tokens - self.compressed_tokens


@dataclass
class EnhancementResult(ProcessingResult):
    """Result from enhancement operations."""
    improvements: List[str] = field(default_factory=list)
    original_quality: Optional[QualityMetrics] = None
    enhanced_quality: Optional[QualityMetrics] = None
    rules_applied: List[str] = field(default_factory=list)
    llm_enhanced: bool = False
    detected_intent: str = ""

    @property
    def quality_improvement(self) -> float:
        """Quality score improvement."""
        if self.original_quality and self.enhanced_quality:
            return self.enhanced_quality.overall_score - self.original_quality.overall_score
        return 0.0


@dataclass
class GenerationResult(ProcessingResult):
    """Result from generation operations."""
    task_description: str = ""
    style_used: str = ""
    template_used: str = ""
    slots_filled: Dict[str, Any] = field(default_factory=dict)
    optimization_iterations: int = 0
    estimated_tokens: int = 0
