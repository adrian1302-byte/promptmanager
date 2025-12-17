"""Prompt validation rules and engine."""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"      # Must fix
    WARNING = "warning"  # Should fix
    INFO = "info"        # Suggestion


class ValidationCategory(Enum):
    """Categories of validation rules."""
    STRUCTURE = "structure"
    CONTENT = "content"
    SECURITY = "security"
    QUALITY = "quality"
    FORMAT = "format"


@dataclass
class ValidationIssue:
    """A single validation issue."""
    rule_name: str
    message: str
    severity: ValidationSeverity
    category: ValidationCategory
    position: Optional[int] = None  # Character position if applicable
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of prompt validation."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    score: float = 1.0  # 0-1 validation score
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> List[ValidationIssue]:
        """Get only error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get only warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "score": self.score,
            "issues": [
                {
                    "rule": i.rule_name,
                    "message": i.message,
                    "severity": i.severity.value,
                    "category": i.category.value,
                    "position": i.position,
                    "suggestion": i.suggestion,
                }
                for i in self.issues
            ],
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "metadata": self.metadata,
        }


@dataclass
class ValidationRule:
    """A validation rule definition."""
    name: str
    description: str
    category: ValidationCategory
    severity: ValidationSeverity
    check: Callable[[str, Dict[str, Any]], Optional[ValidationIssue]]
    enabled: bool = True


class PromptValidator:
    """
    Prompt validation engine.

    Validates prompts against configurable rules for
    security, quality, and format compliance.
    """

    def __init__(self, enable_all: bool = True):
        """
        Initialize the validator.

        Args:
            enable_all: Whether to enable all built-in rules
        """
        self.rules: List[ValidationRule] = []
        if enable_all:
            self._load_builtin_rules()

    def _load_builtin_rules(self) -> None:
        """Load built-in validation rules."""

        # Security rules
        self.add_rule(ValidationRule(
            name="no_injection_patterns",
            description="Check for potential injection patterns",
            category=ValidationCategory.SECURITY,
            severity=ValidationSeverity.ERROR,
            check=self._check_injection_patterns
        ))

        self.add_rule(ValidationRule(
            name="no_pii_exposure",
            description="Check for potential PII in prompts",
            category=ValidationCategory.SECURITY,
            severity=ValidationSeverity.WARNING,
            check=self._check_pii_exposure
        ))

        self.add_rule(ValidationRule(
            name="no_api_keys",
            description="Check for exposed API keys or secrets",
            category=ValidationCategory.SECURITY,
            severity=ValidationSeverity.ERROR,
            check=self._check_api_keys
        ))

        # Structure rules
        self.add_rule(ValidationRule(
            name="not_empty",
            description="Prompt must not be empty",
            category=ValidationCategory.STRUCTURE,
            severity=ValidationSeverity.ERROR,
            check=self._check_not_empty
        ))

        self.add_rule(ValidationRule(
            name="reasonable_length",
            description="Prompt should be within reasonable length",
            category=ValidationCategory.STRUCTURE,
            severity=ValidationSeverity.WARNING,
            check=self._check_reasonable_length
        ))

        self.add_rule(ValidationRule(
            name="balanced_brackets",
            description="Check for balanced brackets and quotes",
            category=ValidationCategory.FORMAT,
            severity=ValidationSeverity.WARNING,
            check=self._check_balanced_brackets
        ))

        # Quality rules
        self.add_rule(ValidationRule(
            name="has_clear_instruction",
            description="Prompt should have a clear instruction",
            category=ValidationCategory.QUALITY,
            severity=ValidationSeverity.INFO,
            check=self._check_clear_instruction
        ))

        self.add_rule(ValidationRule(
            name="no_excessive_whitespace",
            description="Check for excessive whitespace",
            category=ValidationCategory.FORMAT,
            severity=ValidationSeverity.INFO,
            check=self._check_excessive_whitespace
        ))

        # Content rules
        self.add_rule(ValidationRule(
            name="no_broken_templates",
            description="Check for unfilled template variables",
            category=ValidationCategory.CONTENT,
            severity=ValidationSeverity.ERROR,
            check=self._check_broken_templates
        ))

    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule."""
        self.rules.append(rule)

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name."""
        for i, rule in enumerate(self.rules):
            if rule.name == name:
                self.rules.pop(i)
                return True
        return False

    def enable_rule(self, name: str) -> bool:
        """Enable a rule by name."""
        for rule in self.rules:
            if rule.name == name:
                rule.enabled = True
                return True
        return False

    def disable_rule(self, name: str) -> bool:
        """Disable a rule by name."""
        for rule in self.rules:
            if rule.name == name:
                rule.enabled = False
                return True
        return False

    def validate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        categories: Optional[List[ValidationCategory]] = None,
        min_severity: ValidationSeverity = ValidationSeverity.INFO
    ) -> ValidationResult:
        """
        Validate a prompt against rules.

        Args:
            prompt: The prompt to validate
            context: Additional context for validation
            categories: Filter to specific categories (None = all)
            min_severity: Minimum severity to report

        Returns:
            ValidationResult with all issues found
        """
        context = context or {}
        issues: List[ValidationIssue] = []

        severity_order = {
            ValidationSeverity.INFO: 0,
            ValidationSeverity.WARNING: 1,
            ValidationSeverity.ERROR: 2,
        }
        min_sev_level = severity_order[min_severity]

        for rule in self.rules:
            if not rule.enabled:
                continue

            # Filter by category
            if categories and rule.category not in categories:
                continue

            # Filter by severity
            if severity_order[rule.severity] < min_sev_level:
                continue

            try:
                issue = rule.check(prompt, context)
                if issue:
                    issues.append(issue)
            except Exception:
                # Rule failed - skip silently
                continue

        # Calculate validation score
        score = self._calculate_score(issues)

        # Determine validity (no errors)
        is_valid = not any(i.severity == ValidationSeverity.ERROR for i in issues)

        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            score=score,
            metadata={
                "rules_checked": sum(1 for r in self.rules if r.enabled),
                "prompt_length": len(prompt),
            }
        )

    def _calculate_score(self, issues: List[ValidationIssue]) -> float:
        """Calculate validation score from issues."""
        if not issues:
            return 1.0

        # Deductions by severity
        deductions = {
            ValidationSeverity.ERROR: 0.3,
            ValidationSeverity.WARNING: 0.1,
            ValidationSeverity.INFO: 0.02,
        }

        total_deduction = sum(deductions[i.severity] for i in issues)
        return max(0.0, 1.0 - total_deduction)

    def list_rules(self) -> List[Dict[str, Any]]:
        """List all rules with their status."""
        return [
            {
                "name": r.name,
                "description": r.description,
                "category": r.category.value,
                "severity": r.severity.value,
                "enabled": r.enabled,
            }
            for r in self.rules
        ]

    # Built-in rule implementations
    def _check_injection_patterns(
        self,
        prompt: str,
        context: Dict[str, Any]
    ) -> Optional[ValidationIssue]:
        """Check for injection-like patterns."""
        dangerous_patterns = [
            r"ignore.*previous.*instructions",
            r"ignore.*above",
            r"disregard.*instructions",
            r"forget.*everything",
            r"you are now",
            r"new instructions:",
            r"system:\s*",
        ]

        prompt_lower = prompt.lower()
        for pattern in dangerous_patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                return ValidationIssue(
                    rule_name="no_injection_patterns",
                    message=f"Potential injection pattern detected: '{match.group()}'",
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.SECURITY,
                    position=match.start(),
                    suggestion="Remove or rephrase this section"
                )
        return None

    def _check_pii_exposure(
        self,
        prompt: str,
        context: Dict[str, Any]
    ) -> Optional[ValidationIssue]:
        """Check for PII patterns."""
        pii_patterns = [
            (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
            (r"\b\d{16}\b", "Credit card"),
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "Email"),
            (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "Phone number"),
        ]

        for pattern, pii_type in pii_patterns:
            match = re.search(pattern, prompt)
            if match:
                return ValidationIssue(
                    rule_name="no_pii_exposure",
                    message=f"Potential {pii_type} detected in prompt",
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.SECURITY,
                    position=match.start(),
                    suggestion=f"Consider removing or masking the {pii_type}"
                )
        return None

    def _check_api_keys(
        self,
        prompt: str,
        context: Dict[str, Any]
    ) -> Optional[ValidationIssue]:
        """Check for API keys or secrets."""
        secret_patterns = [
            (r"sk-[a-zA-Z0-9]{48}", "OpenAI API key"),
            (r"sk-ant-[a-zA-Z0-9-]+", "Anthropic API key"),
            (r"AIza[0-9A-Za-z-_]{35}", "Google API key"),
            (r"ghp_[a-zA-Z0-9]{36}", "GitHub token"),
            (r"Bearer\s+[a-zA-Z0-9-_.]+", "Bearer token"),
            (r"(api[_-]?key|apikey|secret[_-]?key)\s*[=:]\s*['\"][^'\"]+['\"]", "API key assignment"),
        ]

        for pattern, secret_type in secret_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                return ValidationIssue(
                    rule_name="no_api_keys",
                    message=f"Potential {secret_type} detected",
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.SECURITY,
                    position=match.start(),
                    suggestion="Remove API keys and secrets from prompts"
                )
        return None

    def _check_not_empty(
        self,
        prompt: str,
        context: Dict[str, Any]
    ) -> Optional[ValidationIssue]:
        """Check prompt is not empty."""
        if not prompt or not prompt.strip():
            return ValidationIssue(
                rule_name="not_empty",
                message="Prompt is empty",
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.STRUCTURE,
                suggestion="Provide prompt content"
            )
        return None

    def _check_reasonable_length(
        self,
        prompt: str,
        context: Dict[str, Any]
    ) -> Optional[ValidationIssue]:
        """Check prompt length is reasonable."""
        min_length = context.get("min_length", 10)
        max_length = context.get("max_length", 50000)

        if len(prompt) < min_length:
            return ValidationIssue(
                rule_name="reasonable_length",
                message=f"Prompt too short ({len(prompt)} chars, minimum {min_length})",
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.STRUCTURE,
                suggestion="Add more detail to your prompt"
            )

        if len(prompt) > max_length:
            return ValidationIssue(
                rule_name="reasonable_length",
                message=f"Prompt too long ({len(prompt)} chars, maximum {max_length})",
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.STRUCTURE,
                suggestion="Consider compressing or splitting the prompt"
            )

        return None

    def _check_balanced_brackets(
        self,
        prompt: str,
        context: Dict[str, Any]
    ) -> Optional[ValidationIssue]:
        """Check for balanced brackets and quotes."""
        pairs = {
            "(": ")",
            "[": "]",
            "{": "}",
        }

        stack = []
        for i, char in enumerate(prompt):
            if char in pairs:
                stack.append((char, i))
            elif char in pairs.values():
                if not stack:
                    return ValidationIssue(
                        rule_name="balanced_brackets",
                        message=f"Unmatched closing bracket '{char}'",
                        severity=ValidationSeverity.WARNING,
                        category=ValidationCategory.FORMAT,
                        position=i,
                        suggestion="Check bracket matching"
                    )
                opening, _ = stack.pop()
                if pairs[opening] != char:
                    return ValidationIssue(
                        rule_name="balanced_brackets",
                        message=f"Mismatched brackets: '{opening}' and '{char}'",
                        severity=ValidationSeverity.WARNING,
                        category=ValidationCategory.FORMAT,
                        position=i
                    )

        if stack:
            opening, pos = stack[0]
            return ValidationIssue(
                rule_name="balanced_brackets",
                message=f"Unclosed bracket '{opening}'",
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.FORMAT,
                position=pos
            )

        return None

    def _check_clear_instruction(
        self,
        prompt: str,
        context: Dict[str, Any]
    ) -> Optional[ValidationIssue]:
        """Check for clear instruction."""
        instruction_patterns = [
            r"^(please\s+)?(write|generate|create|explain|summarize|analyze|review|translate|find|list|describe|implement|solve|calculate)",
            r"^(your task|you are|you will|i need you to|can you|could you)",
            r"^(what|why|how|when|where|who)",
        ]

        prompt_start = prompt.strip()[:200].lower()

        for pattern in instruction_patterns:
            if re.search(pattern, prompt_start, re.IGNORECASE):
                return None

        return ValidationIssue(
            rule_name="has_clear_instruction",
            message="Prompt may lack a clear instruction",
            severity=ValidationSeverity.INFO,
            category=ValidationCategory.QUALITY,
            suggestion="Start with a clear action verb (e.g., 'Write', 'Explain', 'Analyze')"
        )

    def _check_excessive_whitespace(
        self,
        prompt: str,
        context: Dict[str, Any]
    ) -> Optional[ValidationIssue]:
        """Check for excessive whitespace."""
        if "   " in prompt or "\n\n\n" in prompt:
            return ValidationIssue(
                rule_name="no_excessive_whitespace",
                message="Excessive whitespace detected",
                severity=ValidationSeverity.INFO,
                category=ValidationCategory.FORMAT,
                suggestion="Clean up extra spaces and newlines"
            )
        return None

    def _check_broken_templates(
        self,
        prompt: str,
        context: Dict[str, Any]
    ) -> Optional[ValidationIssue]:
        """Check for unfilled template variables."""
        template_patterns = [
            r"\{\{\s*\w+\s*\}\}",  # {{ variable }}
            r"\$\{\s*\w+\s*\}",    # ${variable}
            r"\[\[.*?\]\]",        # [[placeholder]]
            r"<\w+>",              # <placeholder>
        ]

        for pattern in template_patterns:
            match = re.search(pattern, prompt)
            if match:
                return ValidationIssue(
                    rule_name="no_broken_templates",
                    message=f"Unfilled template variable: '{match.group()}'",
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.CONTENT,
                    position=match.start(),
                    suggestion="Fill in or remove the template variable"
                )

        return None


# Convenience function
def validate_prompt(
    prompt: str,
    categories: Optional[List[ValidationCategory]] = None
) -> ValidationResult:
    """
    Validate a prompt with default rules.

    Args:
        prompt: The prompt to validate
        categories: Optional category filter

    Returns:
        ValidationResult
    """
    validator = PromptValidator()
    return validator.validate(prompt, categories=categories)
