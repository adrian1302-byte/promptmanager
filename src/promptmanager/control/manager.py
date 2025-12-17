"""Prompt control manager for centralized prompt management."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path

from .versioning import (
    PromptVersioning,
    PromptVersion,
    PromptHistory,
    VersionStatus,
    FileVersioningBackend,
    MemoryVersioningBackend,
)
from .validation import (
    PromptValidator,
    ValidationResult,
    ValidationCategory,
    ValidationSeverity,
)


@dataclass
class ManagedPrompt:
    """A managed prompt with full metadata."""
    prompt_id: str
    name: str
    content: str
    version: int
    status: VersionStatus
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ControlConfig:
    """Configuration for prompt control."""
    storage_path: Optional[str] = None  # None = in-memory
    auto_validate: bool = True
    validation_on_create: bool = True
    validation_on_activate: bool = True
    allow_invalid_prompts: bool = False


class PromptControlManager:
    """
    Central manager for prompt control operations.

    Combines versioning, validation, and template management
    into a unified interface.
    """

    def __init__(self, config: Optional[ControlConfig] = None):
        """
        Initialize the control manager.

        Args:
            config: Control configuration
        """
        self.config = config or ControlConfig()

        # Initialize versioning backend
        if self.config.storage_path:
            backend = FileVersioningBackend(self.config.storage_path)
        else:
            backend = MemoryVersioningBackend()

        self.versioning = PromptVersioning(backend)
        self.validator = PromptValidator()

        # Template library
        self._template_library: Dict[str, str] = {}

    # Prompt Management
    def create_prompt(
        self,
        prompt_id: str,
        name: str,
        content: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        validate: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ManagedPrompt:
        """
        Create a new managed prompt.

        Args:
            prompt_id: Unique identifier for the prompt
            name: Human-readable name
            content: Prompt content
            description: Version description
            tags: Optional tags for categorization
            validate: Override auto_validate setting
            metadata: Optional metadata

        Returns:
            ManagedPrompt instance

        Raises:
            ValueError: If validation fails and allow_invalid_prompts is False
        """
        should_validate = validate if validate is not None else self.config.validation_on_create

        if should_validate:
            validation = self.validator.validate(content)
            if not validation.is_valid and not self.config.allow_invalid_prompts:
                errors = [i.message for i in validation.errors]
                raise ValueError(f"Prompt validation failed: {'; '.join(errors)}")

        version = self.versioning.create_prompt(
            prompt_id=prompt_id,
            name=name,
            content=content,
            description=description,
            tags=tags,
            metadata=metadata
        )

        return ManagedPrompt(
            prompt_id=prompt_id,
            name=name,
            content=content,
            version=version.version_number,
            status=version.status,
            tags=tags or [],
            metadata=metadata or {}
        )

    def update_prompt(
        self,
        prompt_id: str,
        content: str,
        description: str = "",
        auto_activate: bool = True,
        validate: Optional[bool] = None
    ) -> ManagedPrompt:
        """
        Create a new version of an existing prompt.

        Args:
            prompt_id: Prompt identifier
            content: New content
            description: Version description
            auto_activate: Whether to automatically activate
            validate: Override auto_validate setting

        Returns:
            ManagedPrompt with new version
        """
        should_validate = validate if validate is not None else self.config.auto_validate

        if should_validate:
            validation = self.validator.validate(content)
            if not validation.is_valid and not self.config.allow_invalid_prompts:
                errors = [i.message for i in validation.errors]
                raise ValueError(f"Prompt validation failed: {'; '.join(errors)}")

        version = self.versioning.create_version(
            prompt_id=prompt_id,
            content=content,
            description=description,
            auto_activate=auto_activate
        )

        history = self.versioning.get_history(prompt_id)

        return ManagedPrompt(
            prompt_id=prompt_id,
            name=history.name if history else prompt_id,
            content=content,
            version=version.version_number,
            status=version.status,
            tags=history.tags if history else [],
            metadata=version.metadata
        )

    def get_prompt(
        self,
        prompt_id: str,
        version: Optional[int] = None
    ) -> Optional[ManagedPrompt]:
        """
        Get a prompt by ID and optional version.

        Args:
            prompt_id: Prompt identifier
            version: Specific version (None for active)

        Returns:
            ManagedPrompt or None if not found
        """
        prompt_version = self.versioning.get_version(prompt_id, version)
        if not prompt_version:
            return None

        history = self.versioning.get_history(prompt_id)

        return ManagedPrompt(
            prompt_id=prompt_id,
            name=history.name if history else prompt_id,
            content=prompt_version.content,
            version=prompt_version.version_number,
            status=prompt_version.status,
            tags=history.tags if history else [],
            metadata=prompt_version.metadata
        )

    def get_content(
        self,
        prompt_id: str,
        version: Optional[int] = None
    ) -> Optional[str]:
        """Get just the prompt content."""
        return self.versioning.get_content(prompt_id, version)

    def activate_version(
        self,
        prompt_id: str,
        version: int,
        validate: Optional[bool] = None
    ) -> ManagedPrompt:
        """
        Activate a specific version.

        Args:
            prompt_id: Prompt identifier
            version: Version number to activate
            validate: Override validation setting

        Returns:
            ManagedPrompt with activated version
        """
        # Get content for validation
        prompt_version = self.versioning.get_version(prompt_id, version)
        if not prompt_version:
            raise ValueError(f"Version {version} not found for prompt '{prompt_id}'")

        should_validate = validate if validate is not None else self.config.validation_on_activate

        if should_validate:
            validation = self.validator.validate(prompt_version.content)
            if not validation.is_valid and not self.config.allow_invalid_prompts:
                errors = [i.message for i in validation.errors]
                raise ValueError(f"Cannot activate invalid prompt: {'; '.join(errors)}")

        activated = self.versioning.activate_version(prompt_id, version)
        history = self.versioning.get_history(prompt_id)

        return ManagedPrompt(
            prompt_id=prompt_id,
            name=history.name if history else prompt_id,
            content=activated.content,
            version=activated.version_number,
            status=activated.status,
            tags=history.tags if history else [],
            metadata=activated.metadata
        )

    def list_prompts(self, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all managed prompts.

        Args:
            tag: Optional filter by tag

        Returns:
            List of prompt summaries
        """
        prompts = self.versioning.list_prompts()

        if tag:
            prompts = [p for p in prompts if tag in p.get("tags", [])]

        return prompts

    def delete_prompt(self, prompt_id: str) -> bool:
        """Delete a prompt and all its versions."""
        return self.versioning.delete_prompt(prompt_id)

    def tag_prompt(self, prompt_id: str, tags: List[str]) -> None:
        """Add tags to a prompt."""
        self.versioning.tag_prompt(prompt_id, tags)

    def search_by_tag(self, tag: str) -> List[str]:
        """Find prompts with a specific tag."""
        return self.versioning.search_by_tag(tag)

    # Version History
    def get_history(self, prompt_id: str) -> Optional[PromptHistory]:
        """Get full version history."""
        return self.versioning.get_history(prompt_id)

    def compare_versions(
        self,
        prompt_id: str,
        version_a: int,
        version_b: int
    ) -> Dict[str, Any]:
        """Compare two versions of a prompt."""
        return self.versioning.compare_versions(prompt_id, version_a, version_b)

    # Validation
    def validate_prompt(
        self,
        content: str,
        categories: Optional[List[ValidationCategory]] = None
    ) -> ValidationResult:
        """
        Validate prompt content.

        Args:
            content: Prompt content to validate
            categories: Optional category filter

        Returns:
            ValidationResult
        """
        return self.validator.validate(content, categories=categories)

    def validate_stored_prompt(
        self,
        prompt_id: str,
        version: Optional[int] = None
    ) -> ValidationResult:
        """
        Validate a stored prompt.

        Args:
            prompt_id: Prompt identifier
            version: Specific version (None for active)

        Returns:
            ValidationResult
        """
        content = self.get_content(prompt_id, version)
        if not content:
            from .validation import ValidationIssue
            return ValidationResult(
                is_valid=False,
                issues=[ValidationIssue(
                    rule_name="exists",
                    message=f"Prompt '{prompt_id}' not found",
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.STRUCTURE
                )],
                score=0.0
            )

        return self.validator.validate(content)

    # Template Library
    def register_template(self, name: str, content: str) -> None:
        """
        Register a prompt template in the library.

        Args:
            name: Template name
            content: Template content
        """
        # Validate template
        if self.config.auto_validate:
            # Don't fail on templates with placeholders
            pass

        self._template_library[name] = content

    def get_template(self, name: str) -> Optional[str]:
        """Get a template from the library."""
        return self._template_library.get(name)

    def list_templates(self) -> List[str]:
        """List all registered templates."""
        return list(self._template_library.keys())

    def create_from_template(
        self,
        prompt_id: str,
        name: str,
        template_name: str,
        variables: Dict[str, Any],
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> ManagedPrompt:
        """
        Create a prompt from a template.

        Args:
            prompt_id: New prompt identifier
            name: Human-readable name
            template_name: Template to use
            variables: Variables to fill in template
            description: Version description
            tags: Optional tags

        Returns:
            ManagedPrompt
        """
        template = self._template_library.get(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")

        # Simple variable substitution
        content = template
        for key, value in variables.items():
            content = content.replace(f"{{{{{key}}}}}", str(value))
            content = content.replace(f"${{{key}}}", str(value))

        return self.create_prompt(
            prompt_id=prompt_id,
            name=name,
            content=content,
            description=description,
            tags=tags,
            metadata={"template": template_name, "variables": variables}
        )

    # Export/Import
    def export_prompt(self, prompt_id: str, include_history: bool = False) -> Dict[str, Any]:
        """
        Export a prompt to a dictionary.

        Args:
            prompt_id: Prompt identifier
            include_history: Whether to include version history

        Returns:
            Export dictionary
        """
        history = self.versioning.get_history(prompt_id)
        if not history:
            raise ValueError(f"Prompt '{prompt_id}' not found")

        if include_history:
            return history.to_dict()

        active = history.get_active_version()
        return {
            "prompt_id": history.prompt_id,
            "name": history.name,
            "content": active.content if active else "",
            "tags": history.tags,
            "metadata": history.metadata,
        }

    def import_prompt(
        self,
        data: Dict[str, Any],
        overwrite: bool = False
    ) -> ManagedPrompt:
        """
        Import a prompt from a dictionary.

        Args:
            data: Import dictionary
            overwrite: Whether to overwrite if exists

        Returns:
            ManagedPrompt
        """
        prompt_id = data["prompt_id"]

        # Check if exists
        existing = self.versioning.get_history(prompt_id)
        if existing and not overwrite:
            raise ValueError(f"Prompt '{prompt_id}' already exists")

        if existing:
            self.versioning.delete_prompt(prompt_id)

        # Check if full history or single prompt
        if "versions" in data:
            # Full history import
            history = PromptHistory.from_dict(data)
            self.versioning.backend.save(history)
            active = history.get_active_version()
            return ManagedPrompt(
                prompt_id=history.prompt_id,
                name=history.name,
                content=active.content if active else "",
                version=active.version_number if active else 0,
                status=active.status if active else VersionStatus.DRAFT,
                tags=history.tags,
                metadata=history.metadata
            )
        else:
            # Single prompt import
            return self.create_prompt(
                prompt_id=prompt_id,
                name=data.get("name", prompt_id),
                content=data["content"],
                tags=data.get("tags", []),
                metadata=data.get("metadata", {})
            )


# Convenience function
def create_control_manager(
    storage_path: Optional[str] = None,
    auto_validate: bool = True
) -> PromptControlManager:
    """
    Create a prompt control manager.

    Args:
        storage_path: Path for file storage (None for in-memory)
        auto_validate: Whether to auto-validate prompts

    Returns:
        PromptControlManager instance
    """
    config = ControlConfig(
        storage_path=storage_path,
        auto_validate=auto_validate
    )
    return PromptControlManager(config)
