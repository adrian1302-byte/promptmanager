"""Custom exceptions for the prompt management system."""

from typing import Optional, Dict, Any


class PromptManagerError(Exception):
    """Base exception for all prompt manager errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class CompressionError(PromptManagerError):
    """Error during prompt compression."""

    def __init__(
        self,
        message: str,
        strategy: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message, details, cause)
        self.strategy = strategy
        if strategy:
            self.details["strategy"] = strategy


class EnhancementError(PromptManagerError):
    """Error during prompt enhancement."""

    def __init__(
        self,
        message: str,
        stage: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message, details, cause)
        self.stage = stage
        if stage:
            self.details["stage"] = stage


class GenerationError(PromptManagerError):
    """Error during prompt generation."""

    def __init__(
        self,
        message: str,
        template: Optional[str] = None,
        missing_slots: Optional[list] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message, details, cause)
        self.template = template
        self.missing_slots = missing_slots or []
        if template:
            self.details["template"] = template
        if missing_slots:
            self.details["missing_slots"] = missing_slots


class ProviderError(PromptManagerError):
    """Error from LLM provider."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message, details, cause)
        self.provider = provider
        self.status_code = status_code
        if provider:
            self.details["provider"] = provider
        if status_code:
            self.details["status_code"] = status_code


class ValidationError(PromptManagerError):
    """Error during prompt validation."""

    def __init__(
        self,
        message: str,
        validation_errors: Optional[list] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message, details, cause)
        self.validation_errors = validation_errors or []
        if validation_errors:
            self.details["validation_errors"] = validation_errors


class ConfigurationError(PromptManagerError):
    """Error in configuration."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message, details, cause)
        self.config_key = config_key
        if config_key:
            self.details["config_key"] = config_key


class TokenLimitError(PromptManagerError):
    """Error when token limit is exceeded."""

    def __init__(
        self,
        message: str,
        token_count: Optional[int] = None,
        token_limit: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message, details, cause)
        self.token_count = token_count
        self.token_limit = token_limit
        if token_count:
            self.details["token_count"] = token_count
        if token_limit:
            self.details["token_limit"] = token_limit


class TemplateError(PromptManagerError):
    """Error in template processing."""

    def __init__(
        self,
        message: str,
        template_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message, details, cause)
        self.template_name = template_name
        if template_name:
            self.details["template_name"] = template_name


class PipelineError(PromptManagerError):
    """Error during pipeline processing."""

    def __init__(
        self,
        message: str,
        step: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message, details, cause)
        self.step = step
        if step:
            self.details["step"] = step
