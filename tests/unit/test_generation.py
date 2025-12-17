"""Tests for generation module."""

import pytest
import asyncio
from promptmanager.generation import (
    PromptGenerator,
    PromptStyle,
)
from promptmanager.generation.templates.engine import TemplateEngine
from promptmanager.generation.templates.slot_filler import SlotFiller, get_template_slots
from promptmanager.generation.styles.style_registry import StyleRegistry, StyleRecommendation
from promptmanager.core.exceptions import GenerationError


class TestPromptGenerator:
    """Tests for PromptGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create a generator instance."""
        return PromptGenerator()

    @pytest.mark.asyncio
    async def test_create_generator(self, generator):
        """Test creating a generator."""
        assert generator is not None

    @pytest.mark.asyncio
    async def test_generate_basic(self, generator):
        """Test basic prompt generation."""
        result = await generator.generate(task="Write a sorting function")
        assert result is not None
        assert result.prompt != ""

    @pytest.mark.asyncio
    async def test_generate_with_style(self, generator):
        """Test generation with specific style."""
        result = await generator.generate(
            task="Write a Python sorting function",
            style=PromptStyle.CODE_GENERATION
        )
        assert result.style_used == PromptStyle.CODE_GENERATION
        assert "python" in result.prompt.lower() or "code" in result.prompt.lower()

    @pytest.mark.asyncio
    async def test_generate_zero_shot(self, generator):
        """Test zero-shot generation."""
        result = await generator.generate(
            task="What is machine learning?",
            style=PromptStyle.ZERO_SHOT
        )
        assert result.style_used == PromptStyle.ZERO_SHOT

    @pytest.mark.asyncio
    async def test_generate_few_shot(self, generator):
        """Test few-shot generation with examples."""
        examples = [
            {"input": "2 + 2", "output": "4"},
            {"input": "3 + 3", "output": "6"},
        ]
        result = await generator.generate(
            task="Calculate 5 + 5",
            style=PromptStyle.FEW_SHOT,
            examples=examples
        )
        assert result.style_used == PromptStyle.FEW_SHOT
        # Should include examples in prompt
        assert "2 + 2" in result.prompt or "example" in result.prompt.lower()

    @pytest.mark.asyncio
    async def test_generate_chain_of_thought(self, generator):
        """Test chain-of-thought generation."""
        result = await generator.generate(
            task="Solve this math problem: 15% of 200",
            style=PromptStyle.CHAIN_OF_THOUGHT
        )
        assert result.style_used == PromptStyle.CHAIN_OF_THOUGHT
        # Should include step-by-step guidance
        assert "step" in result.prompt.lower()

    @pytest.mark.asyncio
    async def test_generate_code_generation(self, generator):
        """Test code generation style."""
        result = await generator.generate(
            task="Write a function to reverse a string",
            style=PromptStyle.CODE_GENERATION,
            language="Python"
        )
        assert result.style_used == PromptStyle.CODE_GENERATION
        assert "python" in result.prompt.lower()

    @pytest.mark.asyncio
    async def test_generate_with_context(self, generator):
        """Test generation with context."""
        result = await generator.generate(
            task="Continue this story",
            context="Once upon a time, there was a brave knight."
        )
        assert result.prompt is not None

    @pytest.mark.asyncio
    async def test_generate_with_constraints(self, generator):
        """Test generation with constraints."""
        result = await generator.generate(
            task="Write a poem",
            constraints=["Must rhyme", "Maximum 4 lines"]
        )
        assert result.prompt is not None

    @pytest.mark.asyncio
    async def test_auto_style_selection(self, generator):
        """Test automatic style selection."""
        # Code task should get code style
        result = await generator.generate(task="Write a Python function")
        assert result.style_recommendation is not None

    @pytest.mark.asyncio
    async def test_generate_from_template(self, generator):
        """Test direct template generation."""
        result = await generator.generate_from_template(
            "zero_shot.j2",
            {"task": "Test task"}
        )
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_quick_generate(self, generator):
        """Test quick generate method."""
        result = await generator.quick_generate("Write a hello world program")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_sync(self, generator):
        """Test synchronous generate."""
        result = generator.generate_sync(task="Test task")
        assert result is not None

    def test_add_template(self, generator):
        """Test adding custom template."""
        generator.add_template(
            "custom_test",
            "Custom: {{ task }}",
            description="Test template"
        )
        templates = generator.list_templates()
        names = [t["name"] for t in templates]
        assert "custom_test" in names

    def test_list_templates(self, generator):
        """Test listing templates."""
        templates = generator.list_templates()
        assert isinstance(templates, list)
        assert len(templates) > 0

    def test_list_styles(self, generator):
        """Test listing styles."""
        styles = generator.list_styles()
        assert isinstance(styles, list)
        assert len(styles) > 0

    def test_recommend_style(self, generator):
        """Test style recommendation."""
        recommendation = generator.recommend_style(
            task="Write Python code",
            has_examples=False
        )
        assert isinstance(recommendation, StyleRecommendation)


class TestPromptStyle:
    """Tests for PromptStyle enum."""

    def test_style_values(self):
        """Test style enum values."""
        assert PromptStyle.ZERO_SHOT.value == "zero_shot"
        assert PromptStyle.FEW_SHOT.value == "few_shot"
        assert PromptStyle.CHAIN_OF_THOUGHT.value == "chain_of_thought"
        assert PromptStyle.CODE_GENERATION.value == "code_generation"

    def test_style_from_string(self):
        """Test creating style from string."""
        style = PromptStyle("zero_shot")
        assert style == PromptStyle.ZERO_SHOT


class TestTemplateEngine:
    """Tests for TemplateEngine class."""

    @pytest.fixture
    def engine(self):
        """Create a template engine."""
        return TemplateEngine()

    def test_render_builtin_template(self, engine):
        """Test rendering built-in template."""
        result = engine.render("zero_shot.j2", {"task": "Test task"})
        assert "Test task" in result

    def test_render_with_variables(self, engine):
        """Test rendering with multiple variables."""
        result = engine.render("code_generation.j2", {
            "task": "Write a function",
            "language": "Python"
        })
        assert "Python" in result

    def test_list_templates(self, engine):
        """Test listing templates."""
        templates = engine.list_templates()
        assert len(templates) > 0

    def test_add_custom_template(self, engine):
        """Test adding custom template."""
        engine.add_template("test", "Hello {{ name }}!")
        result = engine.render("test", {"name": "World"})
        assert result == "Hello World!"

    def test_template_not_found(self, engine):
        """Test handling missing template."""
        with pytest.raises(Exception):
            engine.render("nonexistent_template.j2", {})


class TestSlotFiller:
    """Tests for SlotFiller class."""

    @pytest.fixture
    def filler(self):
        """Create a slot filler."""
        return SlotFiller()

    def test_fill_slots_basic(self, filler):
        """Test basic slot filling."""
        from promptmanager.generation.templates.slot_filler import SlotDefinition, SlotType
        slots = [
            SlotDefinition(name="task", slot_type=SlotType.TEXT, description="Task to perform"),
            SlotDefinition(name="language", slot_type=SlotType.TEXT, description="Programming language"),
        ]
        result = filler.fill_slots(
            template_slots=slots,
            task_description="Write Python code",
            provided_values={"language": "Python"}
        )
        assert "task" in result.filled_slots or "language" in result.filled_slots

    def test_extract_from_task(self, filler):
        """Test extracting values from task description."""
        from promptmanager.generation.templates.slot_filler import SlotDefinition, SlotType
        slots = [
            SlotDefinition(name="language", slot_type=SlotType.TEXT, description="Programming language"),
        ]
        result = filler.fill_slots(
            template_slots=slots,
            task_description="Write a Python function to sort a list",
            provided_values={}
        )
        # Should extract Python from task
        assert result.filled_slots.get("language") or "python" in str(result.filled_slots).lower()

    def test_missing_slots(self, filler):
        """Test handling missing required slots."""
        from promptmanager.generation.templates.slot_filler import SlotDefinition, SlotType
        slots = [
            SlotDefinition(name="required_field", slot_type=SlotType.TEXT, description="Required field", required=True),
        ]
        result = filler.fill_slots(
            template_slots=slots,
            task_description="Test task",
            provided_values={}
        )
        # Should report missing slots
        assert len(result.missing_slots) > 0 or "required_field" in result.filled_slots


class TestStyleRegistry:
    """Tests for StyleRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create a style registry."""
        return StyleRegistry()

    def test_get_style(self, registry):
        """Test getting a style."""
        style = registry.get_style(PromptStyle.CODE_GENERATION)
        assert style is not None

    def test_recommend_style_for_code(self, registry):
        """Test recommending style for code task."""
        recommendation = registry.recommend_style(
            task_type="code_generation",
            has_examples=False
        )
        assert recommendation.style == PromptStyle.CODE_GENERATION

    def test_recommend_style_with_examples(self, registry):
        """Test recommending style when examples provided."""
        recommendation = registry.recommend_style(
            task_type="general",
            has_examples=True
        )
        assert recommendation.style == PromptStyle.FEW_SHOT

    def test_recommend_style_complex_task(self, registry):
        """Test recommending style for complex task."""
        recommendation = registry.recommend_style(
            task_type="reasoning",
            has_examples=False,
            task_complexity=0.8
        )
        assert recommendation.style == PromptStyle.CHAIN_OF_THOUGHT

    def test_list_styles(self, registry):
        """Test listing styles."""
        styles = registry.list_styles()
        assert len(styles) > 0


class TestGetTemplateSlots:
    """Tests for get_template_slots function."""

    def test_get_slots(self):
        """Test getting template slots."""
        slots = get_template_slots("code_generation.j2")
        assert isinstance(slots, list)

    def test_returns_list(self):
        """Test that function returns a list."""
        slots = get_template_slots("zero_shot.j2")
        assert isinstance(slots, list)


class TestEdgeCases:
    """Tests for edge cases in generation."""

    @pytest.fixture
    def generator(self):
        """Create a generator."""
        return PromptGenerator()

    @pytest.mark.asyncio
    async def test_generate_empty_task(self, generator):
        """Test generating with empty task."""
        result = await generator.generate(task="")
        assert result is not None

    @pytest.mark.asyncio
    async def test_generate_very_long_task(self, generator):
        """Test generating with very long task."""
        long_task = "Write a function that " + "processes data and " * 50 + "returns results"
        result = await generator.generate(task=long_task)
        assert result is not None

    @pytest.mark.asyncio
    async def test_generate_special_characters(self, generator):
        """Test generating with special characters in task."""
        result = await generator.generate(
            task="Write a function for @user#123 with $pecial ch@rs!"
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_generate_unicode_task(self, generator):
        """Test generating with unicode in task."""
        result = await generator.generate(task="Write emoji parser for ")
        assert result is not None

    @pytest.mark.asyncio
    async def test_generate_with_empty_examples(self, generator):
        """Test generating with empty examples list."""
        result = await generator.generate(
            task="Test task",
            style=PromptStyle.FEW_SHOT,
            examples=[]
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_generate_with_extra_kwargs(self, generator):
        """Test generating with extra keyword arguments."""
        result = await generator.generate(
            task="Test task",
            custom_var="custom_value"
        )
        assert result is not None
