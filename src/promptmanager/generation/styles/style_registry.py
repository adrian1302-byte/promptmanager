"""Prompt style registry for automatic style selection."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable


class PromptStyle(Enum):
    """Available prompt styles."""
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    CODE_GENERATION = "code_generation"
    INSTRUCTION = "instruction"
    CONVERSATIONAL = "conversational"
    STRUCTURED = "structured"


@dataclass
class StyleDefinition:
    """Definition of a prompt style."""
    name: str
    style: PromptStyle
    template_name: str
    description: str
    best_for: List[str]  # Task types this style works well for
    complexity_range: tuple = (0.0, 1.0)  # Min/max task complexity
    requires_examples: bool = False
    default_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StyleRecommendation:
    """Recommendation for a prompt style."""
    style: PromptStyle
    template_name: str
    confidence: float
    reasoning: str
    alternative_styles: List[PromptStyle] = field(default_factory=list)


class StyleRegistry:
    """
    Registry for prompt styles with automatic recommendation.

    Provides style selection based on task type, complexity,
    and available resources (examples, context).
    """

    def __init__(self):
        self.styles: Dict[PromptStyle, StyleDefinition] = {}
        self._load_builtin_styles()

    def _load_builtin_styles(self) -> None:
        """Load built-in style definitions."""

        # Zero-shot: Direct instruction without examples
        self.register_style(StyleDefinition(
            name="Zero-Shot",
            style=PromptStyle.ZERO_SHOT,
            template_name="zero_shot.j2",
            description="Direct task instruction without examples",
            best_for=[
                "simple_qa", "summarization", "translation",
                "sentiment_analysis", "classification", "extraction"
            ],
            complexity_range=(0.0, 0.5),
            requires_examples=False,
            default_params={}
        ))

        # Few-shot: Learning from examples
        self.register_style(StyleDefinition(
            name="Few-Shot",
            style=PromptStyle.FEW_SHOT,
            template_name="few_shot.j2",
            description="Task with examples for in-context learning",
            best_for=[
                "classification", "formatting", "transformation",
                "pattern_matching", "data_extraction", "style_transfer"
            ],
            complexity_range=(0.2, 0.8),
            requires_examples=True,
            default_params={"examples": []}
        ))

        # Chain-of-thought: Step-by-step reasoning
        self.register_style(StyleDefinition(
            name="Chain of Thought",
            style=PromptStyle.CHAIN_OF_THOUGHT,
            template_name="chain_of_thought.j2",
            description="Step-by-step reasoning for complex tasks",
            best_for=[
                "reasoning", "math", "complex_qa", "analysis",
                "problem_solving", "debugging", "planning"
            ],
            complexity_range=(0.5, 1.0),
            requires_examples=False,
            default_params={}
        ))

        # Code generation: Structured for coding tasks
        self.register_style(StyleDefinition(
            name="Code Generation",
            style=PromptStyle.CODE_GENERATION,
            template_name="code_generation.j2",
            description="Structured prompt for code generation",
            best_for=[
                "code_generation", "code_review", "debugging",
                "refactoring", "api_design", "testing"
            ],
            complexity_range=(0.3, 1.0),
            requires_examples=False,
            default_params={"language": "Python"}
        ))

    def register_style(self, definition: StyleDefinition) -> None:
        """Register a style definition."""
        self.styles[definition.style] = definition

    def get_style(self, style: PromptStyle) -> Optional[StyleDefinition]:
        """Get a style definition by enum."""
        return self.styles.get(style)

    def get_template_for_style(self, style: PromptStyle) -> Optional[str]:
        """Get the template name for a style."""
        definition = self.styles.get(style)
        return definition.template_name if definition else None

    def recommend_style(
        self,
        task_type: str,
        has_examples: bool = False,
        task_complexity: float = 0.5,
        context: Optional[Dict[str, Any]] = None
    ) -> StyleRecommendation:
        """
        Recommend a prompt style based on task characteristics.

        Args:
            task_type: Type of task (e.g., "code_generation", "qa")
            has_examples: Whether examples are available
            task_complexity: Estimated task complexity (0-1)
            context: Additional context for recommendation

        Returns:
            StyleRecommendation with suggested style
        """
        context = context or {}

        # Score each style
        style_scores: Dict[PromptStyle, float] = {}

        for style, definition in self.styles.items():
            score = 0.0
            reasons = []

            # Task type match
            if task_type.lower() in [t.lower() for t in definition.best_for]:
                score += 0.5
                reasons.append(f"Good match for {task_type}")

            # Partial task type match
            elif any(task_type.lower() in t.lower() or t.lower() in task_type.lower()
                     for t in definition.best_for):
                score += 0.3
                reasons.append(f"Partial match for {task_type}")

            # Complexity match
            min_c, max_c = definition.complexity_range
            if min_c <= task_complexity <= max_c:
                # Score higher for being in the middle of the range
                range_center = (min_c + max_c) / 2
                distance = abs(task_complexity - range_center)
                range_size = (max_c - min_c) / 2
                complexity_score = 0.3 * (1 - distance / range_size if range_size > 0 else 1)
                score += complexity_score
                reasons.append(f"Complexity {task_complexity:.1f} in range")

            # Example availability
            if definition.requires_examples:
                if has_examples:
                    score += 0.2
                    reasons.append("Examples available")
                else:
                    score -= 0.3
                    reasons.append("Missing required examples")
            elif has_examples and style == PromptStyle.FEW_SHOT:
                score += 0.15
                reasons.append("Can leverage available examples")

            # Special case: code tasks
            if "code" in task_type.lower() and style == PromptStyle.CODE_GENERATION:
                score += 0.2
                reasons.append("Optimized for code tasks")

            # Special case: reasoning/analysis tasks
            if any(t in task_type.lower() for t in ["reason", "analyz", "debug", "problem"]):
                if style == PromptStyle.CHAIN_OF_THOUGHT:
                    score += 0.15
                    reasons.append("Benefits from step-by-step reasoning")

            style_scores[style] = score

        # Sort by score
        sorted_styles = sorted(
            style_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        best_style, best_score = sorted_styles[0]
        alternatives = [s for s, _ in sorted_styles[1:3]]

        # Generate reasoning
        definition = self.styles[best_style]
        reasoning = f"{definition.name} recommended for {task_type}"
        if task_complexity > 0.5:
            reasoning += " (complex task)"
        if has_examples and best_style == PromptStyle.FEW_SHOT:
            reasoning += " (leveraging examples)"

        return StyleRecommendation(
            style=best_style,
            template_name=definition.template_name,
            confidence=min(best_score, 1.0),
            reasoning=reasoning,
            alternative_styles=alternatives
        )

    def list_styles(self) -> List[Dict[str, Any]]:
        """List all registered styles."""
        return [
            {
                "style": definition.style.value,
                "name": definition.name,
                "template": definition.template_name,
                "description": definition.description,
                "best_for": definition.best_for,
                "requires_examples": definition.requires_examples,
            }
            for definition in self.styles.values()
        ]


# Mapping from task types to recommended styles
TASK_STYLE_MAP: Dict[str, PromptStyle] = {
    # Code tasks
    "code_generation": PromptStyle.CODE_GENERATION,
    "code_review": PromptStyle.CODE_GENERATION,
    "debugging": PromptStyle.CHAIN_OF_THOUGHT,
    "refactoring": PromptStyle.CODE_GENERATION,

    # Analysis tasks
    "analysis": PromptStyle.CHAIN_OF_THOUGHT,
    "reasoning": PromptStyle.CHAIN_OF_THOUGHT,
    "problem_solving": PromptStyle.CHAIN_OF_THOUGHT,
    "math": PromptStyle.CHAIN_OF_THOUGHT,

    # Simple tasks
    "summarization": PromptStyle.ZERO_SHOT,
    "translation": PromptStyle.ZERO_SHOT,
    "qa": PromptStyle.ZERO_SHOT,
    "question_answering": PromptStyle.ZERO_SHOT,

    # Classification tasks
    "classification": PromptStyle.FEW_SHOT,
    "sentiment_analysis": PromptStyle.FEW_SHOT,
    "categorization": PromptStyle.FEW_SHOT,

    # Extraction tasks
    "data_extraction": PromptStyle.ZERO_SHOT,
    "extraction": PromptStyle.ZERO_SHOT,

    # Creative tasks
    "creative_writing": PromptStyle.ZERO_SHOT,
    "content_generation": PromptStyle.ZERO_SHOT,
}


def get_recommended_style(
    task_type: str,
    has_examples: bool = False,
    complexity: float = 0.5
) -> PromptStyle:
    """
    Quick function to get a recommended style.

    Args:
        task_type: The type of task
        has_examples: Whether examples are available
        complexity: Task complexity (0-1)

    Returns:
        Recommended PromptStyle
    """
    # Direct mapping first
    task_lower = task_type.lower()
    if task_lower in TASK_STYLE_MAP:
        return TASK_STYLE_MAP[task_lower]

    # Use registry for more nuanced recommendation
    registry = StyleRegistry()
    recommendation = registry.recommend_style(
        task_type=task_type,
        has_examples=has_examples,
        task_complexity=complexity
    )
    return recommendation.style
