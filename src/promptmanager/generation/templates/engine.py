"""Jinja2-based template engine for prompt generation."""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field

from jinja2 import Environment, FileSystemLoader, BaseLoader, TemplateNotFound, select_autoescape


@dataclass
class TemplateMetadata:
    """Metadata for a prompt template."""
    name: str
    description: str
    required_variables: List[str]
    optional_variables: List[str] = field(default_factory=list)
    recommended_for: List[str] = field(default_factory=list)  # Task types
    example_usage: Optional[str] = None


class DictLoader(BaseLoader):
    """Jinja2 loader that loads templates from a dictionary."""

    def __init__(self, templates: Dict[str, str]):
        self.templates = templates

    def get_source(self, environment: Environment, template: str):
        if template in self.templates:
            source = self.templates[template]
            return source, template, lambda: True
        raise TemplateNotFound(template)


class TemplateEngine:
    """
    Jinja2-based template engine for generating prompts.

    Provides a flexible templating system with custom filters,
    built-in templates, and support for user-defined templates.
    """

    BUILTIN_DIR = Path(__file__).parent / "builtin"

    def __init__(
        self,
        template_dirs: Optional[List[str]] = None,
        custom_templates: Optional[Dict[str, str]] = None,
        autoescape: bool = False
    ):
        """
        Initialize the template engine.

        Args:
            template_dirs: Additional directories to search for templates
            custom_templates: Dictionary of template_name -> template_content
            autoescape: Whether to auto-escape HTML (usually False for prompts)
        """
        self.template_dirs = template_dirs or []
        self.custom_templates = custom_templates or {}
        self.template_metadata: Dict[str, TemplateMetadata] = {}

        # Build loader chain
        loaders = []

        # Add custom templates loader
        if self.custom_templates:
            loaders.append(DictLoader(self.custom_templates))

        # Add custom directories
        for dir_path in self.template_dirs:
            if os.path.isdir(dir_path):
                loaders.append(FileSystemLoader(dir_path))

        # Add built-in templates
        if self.BUILTIN_DIR.exists():
            loaders.append(FileSystemLoader(str(self.BUILTIN_DIR)))

        # Create environment with choice loader
        from jinja2 import ChoiceLoader
        self.env = Environment(
            loader=ChoiceLoader(loaders) if loaders else None,
            autoescape=select_autoescape() if autoescape else False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Register custom filters
        self._register_filters()

        # Register built-in template metadata
        self._register_builtin_metadata()

    def _register_filters(self) -> None:
        """Register custom Jinja2 filters."""

        def format_examples(examples: List[Dict[str, str]], separator: str = "\n\n") -> str:
            """Format a list of input/output examples."""
            formatted = []
            for i, ex in enumerate(examples, 1):
                input_text = ex.get("input", ex.get("question", ""))
                output_text = ex.get("output", ex.get("answer", ""))
                formatted.append(f"Example {i}:\nInput: {input_text}\nOutput: {output_text}")
            return separator.join(formatted)

        def bullet_list(items: List[str], bullet: str = "-") -> str:
            """Convert a list to bullet points."""
            return "\n".join(f"{bullet} {item}" for item in items)

        def numbered_list(items: List[str]) -> str:
            """Convert a list to numbered items."""
            return "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))

        def wrap_code(code: str, language: str = "") -> str:
            """Wrap code in markdown code block."""
            return f"```{language}\n{code}\n```"

        def indent(text: str, spaces: int = 4) -> str:
            """Indent text by specified spaces."""
            prefix = " " * spaces
            return "\n".join(prefix + line for line in text.split("\n"))

        def truncate(text: str, max_length: int = 500, suffix: str = "...") -> str:
            """Truncate text to max length."""
            if len(text) <= max_length:
                return text
            return text[:max_length - len(suffix)] + suffix

        def capitalize_sentences(text: str) -> str:
            """Capitalize first letter of each sentence."""
            import re
            return re.sub(
                r'(^|[.!?]\s+)([a-z])',
                lambda m: m.group(1) + m.group(2).upper(),
                text
            )

        def json_format(obj: Any) -> str:
            """Format object as JSON."""
            import json
            return json.dumps(obj, indent=2)

        def strip_empty_lines(text: str) -> str:
            """Remove empty lines from text."""
            return "\n".join(line for line in text.split("\n") if line.strip())

        # Register all filters
        self.env.filters["format_examples"] = format_examples
        self.env.filters["bullet_list"] = bullet_list
        self.env.filters["numbered_list"] = numbered_list
        self.env.filters["wrap_code"] = wrap_code
        self.env.filters["indent"] = indent
        self.env.filters["truncate"] = truncate
        self.env.filters["capitalize_sentences"] = capitalize_sentences
        self.env.filters["json_format"] = json_format
        self.env.filters["strip_empty_lines"] = strip_empty_lines

    def _register_builtin_metadata(self) -> None:
        """Register metadata for built-in templates."""
        self.template_metadata = {
            "zero_shot.j2": TemplateMetadata(
                name="Zero-Shot",
                description="Direct task instruction without examples",
                required_variables=["task"],
                optional_variables=["context", "constraints", "output_format"],
                recommended_for=["simple_qa", "classification", "summarization"],
                example_usage='render("zero_shot.j2", task="Summarize this article")'
            ),
            "few_shot.j2": TemplateMetadata(
                name="Few-Shot",
                description="Task instruction with examples for in-context learning",
                required_variables=["task", "examples"],
                optional_variables=["context", "constraints", "output_format"],
                recommended_for=["classification", "formatting", "transformation"],
                example_usage='render("few_shot.j2", task="Classify sentiment", examples=[...])'
            ),
            "chain_of_thought.j2": TemplateMetadata(
                name="Chain of Thought",
                description="Step-by-step reasoning prompt for complex tasks",
                required_variables=["task"],
                optional_variables=["context", "constraints", "steps", "output_format"],
                recommended_for=["reasoning", "math", "complex_qa", "analysis"],
                example_usage='render("chain_of_thought.j2", task="Solve this problem")'
            ),
            "code_generation.j2": TemplateMetadata(
                name="Code Generation",
                description="Structured prompt for code generation tasks",
                required_variables=["task"],
                optional_variables=["language", "requirements", "context", "examples", "constraints"],
                recommended_for=["code_generation", "code_review", "debugging"],
                example_usage='render("code_generation.j2", task="Write a sorting function", language="Python")'
            ),
        }

    def render(
        self,
        template_name: str,
        variables: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Render a template with the given variables.

        Args:
            template_name: Name of the template file or registered template
            variables: Dictionary of template variables
            **kwargs: Additional variables as keyword arguments

        Returns:
            Rendered prompt string
        """
        # Merge variables
        all_vars = {**(variables or {}), **kwargs}

        try:
            template = self.env.get_template(template_name)
            return template.render(**all_vars).strip()
        except TemplateNotFound:
            raise ValueError(f"Template '{template_name}' not found")

    def render_string(self, template_string: str, variables: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        """
        Render a template from a string.

        Args:
            template_string: The template content as a string
            variables: Dictionary of template variables
            **kwargs: Additional variables as keyword arguments

        Returns:
            Rendered prompt string
        """
        all_vars = {**(variables or {}), **kwargs}
        template = self.env.from_string(template_string)
        return template.render(**all_vars).strip()

    def add_template(self, name: str, content: str, metadata: Optional[TemplateMetadata] = None) -> None:
        """
        Add a custom template.

        Args:
            name: Template name (used for retrieval)
            content: Template content (Jinja2 syntax)
            metadata: Optional metadata about the template
        """
        self.custom_templates[name] = content

        # Rebuild loader to include new template
        loaders = [DictLoader(self.custom_templates)]
        for dir_path in self.template_dirs:
            if os.path.isdir(dir_path):
                loaders.append(FileSystemLoader(dir_path))
        if self.BUILTIN_DIR.exists():
            loaders.append(FileSystemLoader(str(self.BUILTIN_DIR)))

        from jinja2 import ChoiceLoader
        self.env.loader = ChoiceLoader(loaders)

        if metadata:
            self.template_metadata[name] = metadata

    def add_filter(self, name: str, func: Callable) -> None:
        """
        Add a custom filter function.

        Args:
            name: Filter name to use in templates
            func: Filter function
        """
        self.env.filters[name] = func

    def list_templates(self) -> List[Dict[str, Any]]:
        """
        List all available templates with metadata.

        Returns:
            List of template information dictionaries
        """
        templates = []

        # Get built-in templates
        if self.BUILTIN_DIR.exists():
            for f in self.BUILTIN_DIR.glob("*.j2"):
                name = f.name
                meta = self.template_metadata.get(name)
                templates.append({
                    "name": name,
                    "type": "builtin",
                    "description": meta.description if meta else "",
                    "required_variables": meta.required_variables if meta else [],
                    "optional_variables": meta.optional_variables if meta else [],
                })

        # Get custom templates
        for name in self.custom_templates:
            meta = self.template_metadata.get(name)
            templates.append({
                "name": name,
                "type": "custom",
                "description": meta.description if meta else "",
                "required_variables": meta.required_variables if meta else [],
                "optional_variables": meta.optional_variables if meta else [],
            })

        return templates

    def get_template_info(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific template."""
        meta = self.template_metadata.get(template_name)
        if not meta:
            return None

        return {
            "name": meta.name,
            "description": meta.description,
            "required_variables": meta.required_variables,
            "optional_variables": meta.optional_variables,
            "recommended_for": meta.recommended_for,
            "example_usage": meta.example_usage,
        }

    def validate_variables(self, template_name: str, variables: Dict[str, Any]) -> List[str]:
        """
        Validate that all required variables are provided.

        Args:
            template_name: Template to validate against
            variables: Variables to check

        Returns:
            List of missing required variable names
        """
        meta = self.template_metadata.get(template_name)
        if not meta:
            return []

        missing = []
        for var in meta.required_variables:
            if var not in variables or variables[var] is None:
                missing.append(var)

        return missing
