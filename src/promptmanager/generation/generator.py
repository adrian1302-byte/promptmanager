"""Main prompt generation orchestrator."""

import asyncio
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from ..core.types import GenerationResult
from ..core.exceptions import GenerationError
from ..providers.base import LLMProvider

from .templates.engine import TemplateEngine
from .templates.slot_filler import SlotFiller, get_template_slots, SlotFillingResult
from .styles.style_registry import StyleRegistry, PromptStyle, StyleRecommendation


@dataclass
class GenerationConfig:
    """Configuration for prompt generation."""
    auto_select_style: bool = True
    auto_fill_slots: bool = True
    use_llm_slot_filling: bool = False
    default_style: PromptStyle = PromptStyle.ZERO_SHOT
    enhance_after_generation: bool = False


@dataclass
class DetailedGenerationResult:
    """Detailed result from prompt generation."""
    prompt: str
    template_used: str
    style_used: PromptStyle
    style_recommendation: Optional[StyleRecommendation] = None
    slot_filling: Optional[SlotFillingResult] = None
    variables_used: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0

    def to_generation_result(self) -> GenerationResult:
        """Convert to standard GenerationResult."""
        return GenerationResult(
            original=self.variables_used.get("task", ""),
            processed=self.prompt,
            task_description=self.variables_used.get("task", ""),
            template_used=self.template_used,
            style_used=self.style_used.value if self.style_used else "",
            slots_filled=self.variables_used,
            metadata={
                "style_confidence": self.style_recommendation.confidence if self.style_recommendation else 1.0,
                "warnings": self.warnings,
                "processing_time_ms": self.processing_time_ms,
            }
        )


class PromptGenerator:
    """
    Main orchestrator for prompt generation.

    Combines template rendering, slot filling, and style selection
    into a cohesive generation pipeline.
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        config: Optional[GenerationConfig] = None,
        template_dirs: Optional[List[str]] = None,
        custom_templates: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the prompt generator.

        Args:
            llm_provider: Optional LLM provider for advanced slot filling
            config: Generation configuration
            template_dirs: Additional template directories
            custom_templates: Custom templates as name -> content dict
        """
        self.config = config or GenerationConfig()
        self.llm_provider = llm_provider

        # Initialize components
        self.template_engine = TemplateEngine(
            template_dirs=template_dirs,
            custom_templates=custom_templates
        )
        self.slot_filler = SlotFiller(llm_client=llm_provider)
        self.style_registry = StyleRegistry()

    async def generate(
        self,
        task: str,
        style: Optional[PromptStyle] = None,
        template: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        examples: Optional[List[Dict[str, str]]] = None,
        context: Optional[str] = None,
        constraints: Optional[List[str]] = None,
        **kwargs
    ) -> DetailedGenerationResult:
        """
        Generate a prompt from a task description.

        Args:
            task: The task description
            style: Specific style to use (auto-selected if None)
            template: Specific template to use (overrides style)
            variables: Pre-filled template variables
            examples: Input/output examples for few-shot
            context: Additional context
            constraints: Constraints to include
            **kwargs: Additional template variables

        Returns:
            DetailedGenerationResult with generated prompt
        """
        import time
        start_time = time.time()

        result = DetailedGenerationResult(
            prompt="",
            template_used="",
            style_used=style or self.config.default_style,
            variables_used={}
        )

        try:
            # Prepare variables
            all_variables: Dict[str, Any] = {
                "task": task,
                **(variables or {}),
                **kwargs
            }

            if examples:
                all_variables["examples"] = examples
            if context:
                all_variables["context"] = context
            if constraints:
                all_variables["constraints"] = constraints

            # Step 1: Select template/style
            if template:
                # Use specified template
                result.template_used = template
                # Try to infer style from template name
                for s, definition in self.style_registry.styles.items():
                    if definition.template_name == template:
                        result.style_used = s
                        break
            elif style:
                # Use specified style
                result.style_used = style
                style_def = self.style_registry.get_style(style)
                if style_def:
                    result.template_used = style_def.template_name
                else:
                    raise GenerationError(f"Unknown style: {style}")
            elif self.config.auto_select_style:
                # Auto-select style based on task
                recommendation = self._recommend_style(
                    task=task,
                    has_examples=bool(examples),
                    variables=all_variables
                )
                result.style_recommendation = recommendation
                result.style_used = recommendation.style
                result.template_used = recommendation.template_name
            else:
                # Use default
                result.style_used = self.config.default_style
                style_def = self.style_registry.get_style(self.config.default_style)
                if style_def:
                    result.template_used = style_def.template_name

            # Step 2: Fill template slots
            if self.config.auto_fill_slots:
                template_slots = get_template_slots(result.template_used)

                if self.config.use_llm_slot_filling and self.llm_provider:
                    slot_result = await self.slot_filler.fill_slots_with_llm(
                        template_slots=template_slots,
                        task_description=task,
                        provided_values=all_variables
                    )
                else:
                    slot_result = self.slot_filler.fill_slots(
                        template_slots=template_slots,
                        task_description=task,
                        provided_values=all_variables,
                        context={"examples": examples, "context": context}
                    )

                result.slot_filling = slot_result
                all_variables.update(slot_result.filled_slots)

                if slot_result.missing_slots:
                    result.warnings.append(
                        f"Missing required slots: {', '.join(slot_result.missing_slots)}"
                    )

                if slot_result.warnings:
                    result.warnings.extend(slot_result.warnings)

            result.variables_used = all_variables

            # Step 3: Render template
            try:
                result.prompt = self.template_engine.render(
                    result.template_used,
                    all_variables
                )
            except Exception as e:
                raise GenerationError(f"Template rendering failed: {str(e)}")

            result.processing_time_ms = (time.time() - start_time) * 1000
            return result

        except GenerationError:
            raise
        except Exception as e:
            raise GenerationError(f"Prompt generation failed: {str(e)}") from e

    def generate_sync(
        self,
        task: str,
        style: Optional[PromptStyle] = None,
        template: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        examples: Optional[List[Dict[str, str]]] = None,
        context: Optional[str] = None,
        constraints: Optional[List[str]] = None,
        **kwargs
    ) -> DetailedGenerationResult:
        """Synchronous version of generate."""
        return asyncio.run(self.generate(
            task, style, template, variables, examples, context, constraints, **kwargs
        ))

    async def generate_from_template(
        self,
        template_name: str,
        variables: Dict[str, Any]
    ) -> str:
        """
        Generate a prompt directly from a template.

        Args:
            template_name: Name of the template
            variables: Template variables

        Returns:
            Rendered prompt string
        """
        return self.template_engine.render(template_name, variables)

    def generate_from_template_sync(
        self,
        template_name: str,
        variables: Dict[str, Any]
    ) -> str:
        """Synchronous version of generate_from_template."""
        return asyncio.run(self.generate_from_template(template_name, variables))

    async def quick_generate(
        self,
        task: str,
        style: Optional[PromptStyle] = None
    ) -> str:
        """
        Quick prompt generation with minimal configuration.

        Args:
            task: The task description
            style: Optional style hint

        Returns:
            Generated prompt string
        """
        result = await self.generate(task=task, style=style)
        return result.prompt

    def quick_generate_sync(self, task: str, style: Optional[PromptStyle] = None) -> str:
        """Synchronous version of quick_generate."""
        return asyncio.run(self.quick_generate(task, style))

    def add_template(
        self,
        name: str,
        content: str,
        description: str = "",
        required_variables: Optional[List[str]] = None
    ) -> None:
        """
        Add a custom template.

        Args:
            name: Template name
            content: Template content (Jinja2 syntax)
            description: Template description
            required_variables: List of required variable names
        """
        from .templates.engine import TemplateMetadata

        metadata = TemplateMetadata(
            name=name,
            description=description,
            required_variables=required_variables or [],
        ) if description or required_variables else None

        self.template_engine.add_template(name, content, metadata)

    def list_templates(self) -> List[Dict[str, Any]]:
        """List all available templates."""
        return self.template_engine.list_templates()

    def list_styles(self) -> List[Dict[str, Any]]:
        """List all available styles."""
        return self.style_registry.list_styles()

    def recommend_style(
        self,
        task: str,
        has_examples: bool = False,
        complexity: float = 0.5
    ) -> StyleRecommendation:
        """
        Get a style recommendation for a task.

        Args:
            task: The task description
            has_examples: Whether examples are available
            complexity: Estimated task complexity (0-1)

        Returns:
            StyleRecommendation
        """
        task_type = self._infer_task_type(task)
        return self.style_registry.recommend_style(
            task_type=task_type,
            has_examples=has_examples,
            task_complexity=complexity
        )

    def _recommend_style(
        self,
        task: str,
        has_examples: bool,
        variables: Dict[str, Any]
    ) -> StyleRecommendation:
        """Internal style recommendation."""
        task_type = self._infer_task_type(task)
        complexity = self._estimate_complexity(task)

        return self.style_registry.recommend_style(
            task_type=task_type,
            has_examples=has_examples,
            task_complexity=complexity
        )

    def _infer_task_type(self, task: str) -> str:
        """Infer task type from description."""
        task_lower = task.lower()

        # Code-related
        if any(kw in task_lower for kw in ["write code", "implement", "function", "class", "api", "script"]):
            return "code_generation"
        if any(kw in task_lower for kw in ["review", "code review"]):
            return "code_review"
        if any(kw in task_lower for kw in ["debug", "fix bug", "error"]):
            return "debugging"

        # Analysis/reasoning
        if any(kw in task_lower for kw in ["analyze", "analysis", "compare", "evaluate"]):
            return "analysis"
        if any(kw in task_lower for kw in ["solve", "calculate", "compute", "math"]):
            return "reasoning"

        # Simple tasks
        if any(kw in task_lower for kw in ["summarize", "summary"]):
            return "summarization"
        if any(kw in task_lower for kw in ["translate", "translation"]):
            return "translation"
        if any(kw in task_lower for kw in ["classify", "categorize", "label"]):
            return "classification"

        # Extraction
        if any(kw in task_lower for kw in ["extract", "find", "identify"]):
            return "extraction"

        # QA
        if "?" in task or any(kw in task_lower for kw in ["what", "why", "how", "explain"]):
            return "question_answering"

        # Creative
        if any(kw in task_lower for kw in ["write", "create", "generate"]):
            return "content_generation"

        return "general"

    def _estimate_complexity(self, task: str) -> float:
        """Estimate task complexity from description."""
        complexity = 0.3  # Base complexity

        # Length factor
        words = len(task.split())
        if words > 50:
            complexity += 0.2
        elif words > 20:
            complexity += 0.1

        # Complexity indicators
        complex_keywords = [
            "complex", "detailed", "comprehensive", "analyze",
            "multiple", "step by step", "thorough", "all"
        ]
        for kw in complex_keywords:
            if kw in task.lower():
                complexity += 0.1

        # Simple indicators
        simple_keywords = ["simple", "basic", "quick", "just"]
        for kw in simple_keywords:
            if kw in task.lower():
                complexity -= 0.1

        return max(0.0, min(1.0, complexity))


# Convenience functions

async def generate(
    task: str,
    style: Optional[PromptStyle] = None,
    examples: Optional[List[Dict[str, str]]] = None,
    context: Optional[str] = None,
    **kwargs
) -> GenerationResult:
    """
    Convenience function to generate a prompt.

    Args:
        task: The task description
        style: Optional style to use
        examples: Optional examples for few-shot
        context: Optional context
        **kwargs: Additional template variables

    Returns:
        GenerationResult
    """
    generator = PromptGenerator()
    result = await generator.generate(
        task=task,
        style=style,
        examples=examples,
        context=context,
        **kwargs
    )
    return result.to_generation_result()


def generate_sync(
    task: str,
    style: Optional[PromptStyle] = None,
    examples: Optional[List[Dict[str, str]]] = None,
    context: Optional[str] = None,
    **kwargs
) -> GenerationResult:
    """Synchronous version of generate."""
    return asyncio.run(generate(task, style, examples, context, **kwargs))


async def generate_code_prompt(
    task: str,
    language: str = "Python",
    requirements: Optional[List[str]] = None,
    **kwargs
) -> str:
    """
    Generate a code generation prompt.

    Args:
        task: The coding task
        language: Programming language
        requirements: Code requirements
        **kwargs: Additional variables

    Returns:
        Generated prompt string
    """
    generator = PromptGenerator()
    result = await generator.generate(
        task=task,
        style=PromptStyle.CODE_GENERATION,
        language=language,
        requirements=requirements or [],
        **kwargs
    )
    return result.prompt


def generate_code_prompt_sync(
    task: str,
    language: str = "Python",
    requirements: Optional[List[str]] = None,
    **kwargs
) -> str:
    """Synchronous version of generate_code_prompt."""
    return asyncio.run(generate_code_prompt(task, language, requirements, **kwargs))


async def generate_cot_prompt(task: str, steps: Optional[List[str]] = None, **kwargs) -> str:
    """
    Generate a chain-of-thought prompt.

    Args:
        task: The reasoning task
        steps: Optional custom reasoning steps
        **kwargs: Additional variables

    Returns:
        Generated prompt string
    """
    generator = PromptGenerator()
    result = await generator.generate(
        task=task,
        style=PromptStyle.CHAIN_OF_THOUGHT,
        steps=steps,
        **kwargs
    )
    return result.prompt


def generate_cot_prompt_sync(task: str, steps: Optional[List[str]] = None, **kwargs) -> str:
    """Synchronous version of generate_cot_prompt."""
    return asyncio.run(generate_cot_prompt(task, steps, **kwargs))
