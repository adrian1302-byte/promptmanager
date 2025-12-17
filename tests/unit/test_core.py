"""Tests for core module - types, exceptions, and base classes."""

import pytest
from promptmanager.core.types import (
    Message,
    Prompt,
    PromptRole,
    ContentType,
    ProcessingResult,
    CompressionResult,
    EnhancementResult,
    GenerationResult,
    QualityMetrics,
)
from promptmanager.core.exceptions import (
    PromptManagerError,
    CompressionError,
    EnhancementError,
    GenerationError,
    ValidationError,
    PipelineError,
    ProviderError,
    ConfigurationError,
)


class TestMessage:
    """Tests for Message dataclass."""

    def test_create_message(self):
        """Test creating a message."""
        msg = Message(role=PromptRole.USER, content="Hello")
        assert msg.role == PromptRole.USER
        assert msg.content == "Hello"
        assert msg.metadata == {}

    def test_message_with_metadata(self):
        """Test message with metadata."""
        msg = Message(
            role=PromptRole.SYSTEM,
            content="You are helpful",
            metadata={"source": "test"}
        )
        assert msg.metadata["source"] == "test"

    def test_message_to_dict(self):
        """Test converting message to dict."""
        msg = Message(role=PromptRole.USER, content="Test")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "Test"}

    def test_message_from_dict(self):
        """Test creating message from dict."""
        d = {"role": "assistant", "content": "Response"}
        msg = Message.from_dict(d)
        assert msg.role == PromptRole.ASSISTANT
        assert msg.content == "Response"


class TestPrompt:
    """Tests for Prompt dataclass."""

    def test_create_empty_prompt(self):
        """Test creating empty prompt."""
        p = Prompt()
        assert p.messages == []
        assert p.id is not None

    def test_from_text(self):
        """Test creating prompt from text."""
        p = Prompt.from_text("Hello world")
        assert len(p.messages) == 1
        assert p.messages[0].role == PromptRole.USER
        assert p.messages[0].content == "Hello world"

    def test_from_text_with_system(self):
        """Test creating prompt with system message."""
        p = Prompt.from_text("Query", system_message="You are helpful")
        assert len(p.messages) == 2
        assert p.messages[0].role == PromptRole.SYSTEM
        assert p.messages[1].role == PromptRole.USER

    def test_from_messages(self):
        """Test creating prompt from message list."""
        messages = [
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Hi"},
        ]
        p = Prompt.from_messages(messages)
        assert len(p.messages) == 2

    def test_text_property(self):
        """Test getting concatenated text."""
        p = Prompt.from_text("Hello", system_message="System")
        assert "System" in p.text
        assert "Hello" in p.text

    def test_user_content_property(self):
        """Test getting user content only."""
        p = Prompt.from_text("User message", system_message="System")
        assert p.user_content == "User message"

    def test_system_content_property(self):
        """Test getting system content."""
        p = Prompt.from_text("User", system_message="System message")
        assert p.system_content == "System message"

    def test_to_openai_format(self):
        """Test converting to OpenAI format."""
        p = Prompt.from_text("Test")
        fmt = p.to_openai_format()
        assert len(fmt) == 1
        assert fmt[0]["role"] == "user"
        assert fmt[0]["content"] == "Test"

    def test_to_anthropic_format(self):
        """Test converting to Anthropic format."""
        p = Prompt.from_text("Test", system_message="System")
        fmt = p.to_anthropic_format()
        assert fmt["system"] == "System"
        assert len(fmt["messages"]) == 1

    def test_copy(self):
        """Test copying prompt."""
        p1 = Prompt.from_text("Original")
        p2 = p1.copy()
        assert p1.id != p2.id
        assert p1.messages[0].content == p2.messages[0].content


class TestPromptRole:
    """Tests for PromptRole enum."""

    def test_roles(self):
        """Test role values."""
        assert PromptRole.SYSTEM.value == "system"
        assert PromptRole.USER.value == "user"
        assert PromptRole.ASSISTANT.value == "assistant"


class TestContentType:
    """Tests for ContentType enum."""

    def test_content_types(self):
        """Test content type values."""
        assert ContentType.NATURAL_LANGUAGE.value == "natural_language"
        assert ContentType.CODE.value == "code"
        assert ContentType.MIXED.value == "mixed"


class TestProcessingResult:
    """Tests for ProcessingResult dataclass."""

    def test_create_result(self):
        """Test creating processing result."""
        p = Prompt.from_text("Test")
        result = ProcessingResult(original=p, processed=p)
        assert result.success == True
        assert result.error is None

    def test_result_with_error(self):
        """Test result with error."""
        p = Prompt.from_text("Test")
        result = ProcessingResult(
            original=p,
            processed=p,
            success=False,
            error="Something went wrong"
        )
        assert result.success == False
        assert result.error == "Something went wrong"


class TestCompressionResult:
    """Tests for CompressionResult dataclass."""

    def test_compression_result(self):
        """Test compression result."""
        p = Prompt.from_text("Test")
        result = CompressionResult(
            original=p,
            processed=p,
            original_tokens=100,
            compressed_tokens=50
        )
        assert result.compression_ratio == 0.5
        assert result.tokens_saved == 50

    def test_tokens_saved(self):
        """Test tokens saved calculation."""
        p = Prompt.from_text("Test")
        result = CompressionResult(
            original=p,
            processed=p,
            original_tokens=200,
            compressed_tokens=150
        )
        assert result.tokens_saved == 50


class TestEnhancementResult:
    """Tests for EnhancementResult dataclass."""

    def test_enhancement_result(self):
        """Test enhancement result."""
        p = Prompt.from_text("Test")
        result = EnhancementResult(
            original=p,
            processed=p,
            improvements=["clarity", "structure"]
        )
        assert len(result.improvements) == 2

    def test_quality_improvement(self):
        """Test quality improvement calculation."""
        p = Prompt.from_text("Test")
        result = EnhancementResult(
            original=p,
            processed=p,
            original_quality=QualityMetrics(overall_score=0.5),
            enhanced_quality=QualityMetrics(overall_score=0.8)
        )
        # Use pytest.approx for floating point comparison
        assert result.quality_improvement == pytest.approx(0.3, abs=1e-10)


class TestGenerationResult:
    """Tests for GenerationResult dataclass."""

    def test_generation_result(self):
        """Test generation result."""
        p = Prompt.from_text("Generated")
        result = GenerationResult(
            original="task",
            processed=p,
            template_used="code_generation.j2",
            style_used="code"
        )
        assert result.template_used == "code_generation.j2"


class TestExceptions:
    """Tests for custom exceptions."""

    def test_prompt_manager_error(self):
        """Test base error."""
        with pytest.raises(PromptManagerError):
            raise PromptManagerError("Base error")

    def test_compression_error(self):
        """Test compression error."""
        with pytest.raises(CompressionError) as exc_info:
            raise CompressionError("Compression failed", strategy="hybrid")
        assert "Compression failed" in str(exc_info.value)

    def test_enhancement_error(self):
        """Test enhancement error."""
        with pytest.raises(EnhancementError):
            raise EnhancementError("Enhancement failed")

    def test_generation_error(self):
        """Test generation error."""
        with pytest.raises(GenerationError):
            raise GenerationError("Generation failed")

    def test_validation_error(self):
        """Test validation error."""
        with pytest.raises(ValidationError):
            raise ValidationError("Validation failed", validation_errors=["issue1"])

    def test_pipeline_error(self):
        """Test pipeline error."""
        with pytest.raises(PipelineError):
            raise PipelineError("Pipeline failed", step="compress")

    def test_provider_error(self):
        """Test provider error."""
        with pytest.raises(ProviderError):
            raise ProviderError("Provider failed", provider="openai")

    def test_configuration_error(self):
        """Test configuration error."""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Config error", config_key="api_key")

    def test_exception_inheritance(self):
        """Test exception inheritance."""
        assert issubclass(CompressionError, PromptManagerError)
        assert issubclass(EnhancementError, PromptManagerError)
        assert issubclass(GenerationError, PromptManagerError)
        assert issubclass(ValidationError, PromptManagerError)
        assert issubclass(PipelineError, PromptManagerError)


class TestQualityMetrics:
    """Tests for QualityMetrics dataclass."""

    def test_create_metrics(self):
        """Test creating quality metrics."""
        metrics = QualityMetrics(
            clarity_score=0.8,
            structure_score=0.7,
            completeness_score=0.9,
            overall_score=0.8
        )
        assert metrics.clarity_score == 0.8
        assert metrics.overall_score == 0.8

    def test_default_metrics(self):
        """Test default metric values."""
        metrics = QualityMetrics()
        assert metrics.clarity_score == 0.0
        assert metrics.overall_score == 0.0
