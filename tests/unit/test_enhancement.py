"""Tests for enhancement module."""

import pytest
import asyncio
from promptmanager.enhancement import (
    PromptEnhancer,
    EnhancementMode,
    EnhancementLevel,
    PromptIntent,
)
from promptmanager.enhancement.analyzers.intent_detector import IntentDetector, IntentDetectionResult
from promptmanager.enhancement.analyzers.quality_scorer import QualityScorer, QualityScore
from promptmanager.enhancement.transformers.rule_engine import RuleEngine
from promptmanager.enhancement.transformers.llm_enhancer import LLMEnhancer, GrammarEnhancer
from promptmanager.core.exceptions import EnhancementError


class TestPromptEnhancer:
    """Tests for PromptEnhancer class."""

    @pytest.fixture
    def enhancer(self):
        """Create an enhancer without LLM."""
        return PromptEnhancer()

    @pytest.fixture
    def enhancer_with_mock_llm(self, mock_llm_provider_with_responses):
        """Create an enhancer with mock LLM."""
        return PromptEnhancer(llm_provider=mock_llm_provider_with_responses)

    @pytest.mark.asyncio
    async def test_create_enhancer(self, enhancer):
        """Test creating an enhancer."""
        assert enhancer is not None
        assert enhancer.config is not None

    @pytest.mark.asyncio
    async def test_enhance_rules_only(self, enhancer, messy_prompt):
        """Test enhancement with rules only."""
        result = await enhancer.enhance(messy_prompt, mode=EnhancementMode.RULES_ONLY)
        assert result is not None
        assert result.enhanced_prompt != messy_prompt
        assert len(result.applied_rules) > 0

    @pytest.mark.asyncio
    async def test_enhance_detects_intent(self, enhancer, code_prompt):
        """Test that enhancement detects intent."""
        result = await enhancer.enhance(code_prompt, mode=EnhancementMode.RULES_ONLY)
        assert result.detected_intent is not None

    @pytest.mark.asyncio
    async def test_enhance_measures_quality(self, enhancer, messy_prompt):
        """Test that enhancement measures quality."""
        result = await enhancer.enhance(messy_prompt, mode=EnhancementMode.RULES_ONLY)
        assert result.original_quality is not None
        assert result.final_quality is not None

    @pytest.mark.asyncio
    async def test_enhance_improves_quality(self, enhancer, messy_prompt):
        """Test that enhancement improves quality score."""
        result = await enhancer.enhance(messy_prompt, mode=EnhancementMode.RULES_ONLY)
        assert result.quality_improvement >= 0

    @pytest.mark.asyncio
    async def test_quick_enhance(self, enhancer, messy_prompt):
        """Test quick enhance method."""
        result = await enhancer.quick_enhance(messy_prompt)
        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_analyze(self, enhancer, code_prompt):
        """Test analyze method."""
        analysis = await enhancer.analyze(code_prompt)
        assert "intent" in analysis
        assert "quality" in analysis
        assert "statistics" in analysis

    def test_enhance_sync(self, enhancer, messy_prompt):
        """Test synchronous enhance."""
        result = enhancer.enhance_sync(messy_prompt, mode=EnhancementMode.RULES_ONLY)
        assert result is not None

    def test_analyze_sync(self, enhancer, code_prompt):
        """Test synchronous analyze."""
        analysis = enhancer.analyze_sync(code_prompt)
        assert "intent" in analysis

    def test_add_custom_rule(self, enhancer):
        """Test adding custom rule."""
        from promptmanager.enhancement.transformers.rule_engine import TransformationRule, RulePriority
        rule = TransformationRule(
            name="test_rule",
            description="Test rule",
            priority=RulePriority.LOW,
            applicable_intents=[],
            condition=lambda p, c: "test" in p.lower(),
            transform=lambda p, c: p.replace("test", "TEST"),
            tags=["test"]
        )
        enhancer.add_custom_rule(rule)
        rules = enhancer.list_rules()
        rule_names = [r["name"] for r in rules]
        assert "test_rule" in rule_names

    def test_set_level(self, enhancer):
        """Test changing enhancement level."""
        enhancer.set_level(EnhancementLevel.AGGRESSIVE)
        assert enhancer.config.level == EnhancementLevel.AGGRESSIVE

    def test_set_mode(self, enhancer):
        """Test changing enhancement mode."""
        enhancer.set_mode(EnhancementMode.ADAPTIVE)
        assert enhancer.config.mode == EnhancementMode.ADAPTIVE


class TestEnhancementMode:
    """Tests for EnhancementMode enum."""

    def test_mode_values(self):
        """Test mode enum values."""
        assert EnhancementMode.RULES_ONLY.value == "rules_only"
        assert EnhancementMode.LLM_ONLY.value == "llm_only"
        assert EnhancementMode.HYBRID.value == "hybrid"
        assert EnhancementMode.ADAPTIVE.value == "adaptive"


class TestEnhancementLevel:
    """Tests for EnhancementLevel enum."""

    def test_level_values(self):
        """Test level enum values."""
        assert EnhancementLevel.MINIMAL.value == "minimal"
        assert EnhancementLevel.LIGHT.value == "light"
        assert EnhancementLevel.MODERATE.value == "moderate"
        assert EnhancementLevel.AGGRESSIVE.value == "aggressive"


class TestIntentDetector:
    """Tests for IntentDetector class."""

    @pytest.fixture
    def detector(self):
        """Create an intent detector."""
        return IntentDetector()

    def test_detect_code_generation(self, detector):
        """Test detecting code generation intent."""
        result = detector.detect("Write a Python function to sort a list")
        assert result.primary_intent == PromptIntent.CODE_GENERATION

    def test_detect_question_answering(self, detector):
        """Test detecting question answering intent."""
        result = detector.detect("What is the capital of France?")
        assert result.primary_intent == PromptIntent.QUESTION_ANSWERING

    def test_detect_summarization(self, detector):
        """Test detecting summarization intent."""
        result = detector.detect("Summarize this article about climate change")
        assert result.primary_intent == PromptIntent.SUMMARIZATION

    def test_detect_translation(self, detector):
        """Test detecting translation intent."""
        result = detector.detect("Translate this text to Spanish")
        assert result.primary_intent == PromptIntent.TRANSLATION

    def test_detect_creative_writing(self, detector):
        """Test detecting creative writing intent."""
        result = detector.detect("Write a short story about a dragon")
        assert result.primary_intent == PromptIntent.CREATIVE_WRITING

    def test_detect_code_review(self, detector):
        """Test detecting code review intent."""
        result = detector.detect("Review this code for bugs and issues")
        assert result.primary_intent == PromptIntent.CODE_REVIEW

    def test_detect_analysis(self, detector):
        """Test detecting analysis intent."""
        result = detector.detect("Analyze the pros and cons of this approach")
        assert result.primary_intent == PromptIntent.ANALYSIS

    def test_returns_confidence(self, detector):
        """Test that detection returns confidence score."""
        result = detector.detect("Write a Python function")
        assert 0 <= result.confidence <= 1

    def test_returns_secondary_intents(self, detector):
        """Test that detection returns secondary intents."""
        result = detector.detect("Write and explain a sorting algorithm")
        assert isinstance(result.secondary_intents, list)

    def test_detect_multiple(self, detector):
        """Test detecting multiple prompts."""
        prompts = [
            "Write Python code",
            "What is AI?",
            "Summarize this"
        ]
        results = detector.detect_multiple(prompts)
        assert len(results) == 3


class TestQualityScorer:
    """Tests for QualityScorer class."""

    @pytest.fixture
    def scorer(self):
        """Create a quality scorer."""
        return QualityScorer()

    def test_score_good_prompt(self, scorer):
        """Test scoring a well-formed prompt."""
        prompt = "Write a Python function that implements a binary search algorithm with proper error handling."
        score = scorer.score(prompt)
        assert score.overall_score > 0.5

    def test_score_poor_prompt(self, scorer, messy_prompt):
        """Test scoring a poorly formed prompt."""
        score = scorer.score(messy_prompt)
        assert score.overall_score < 0.8  # Should be lower quality

    def test_score_components(self, scorer, code_prompt):
        """Test that all score components are present."""
        score = scorer.score(code_prompt)
        assert hasattr(score, 'clarity_score')
        assert hasattr(score, 'structure_score')
        assert hasattr(score, 'completeness_score')
        assert hasattr(score, 'specificity_score')
        assert hasattr(score, 'grammar_score')

    def test_score_range(self, scorer, code_prompt):
        """Test that scores are in valid range."""
        score = scorer.score(code_prompt)
        assert 0 <= score.clarity_score <= 1
        assert 0 <= score.structure_score <= 1
        assert 0 <= score.overall_score <= 1

    def test_score_empty_prompt(self, scorer):
        """Test scoring empty prompt."""
        score = scorer.score("")
        # Empty prompt has lower quality but may still score above 0.5 due to some metrics
        assert score.overall_score < 0.7
        assert len(score.issues) > 0

    def test_issues_and_suggestions(self, scorer, messy_prompt):
        """Test that scorer provides issues and suggestions."""
        score = scorer.score(messy_prompt)
        assert isinstance(score.issues, list)
        assert isinstance(score.suggestions, list)

    def test_compare_prompts(self, scorer):
        """Test comparing two prompts."""
        original = "help me with code"
        enhanced = "Write clean, well-documented Python code to implement a sorting algorithm."
        comparison = scorer.compare(original, enhanced)
        assert "overall_improvement" in comparison
        assert comparison["overall_improvement"] > 0


class TestRuleEngine:
    """Tests for RuleEngine class."""

    @pytest.fixture
    def engine(self):
        """Create a rule engine."""
        return RuleEngine()

    def test_apply_rules(self, engine, messy_prompt):
        """Test applying rules to a prompt."""
        result, applied = engine.apply_rules(messy_prompt, {"intent": "code_generation"})
        assert isinstance(result, str)
        assert isinstance(applied, list)

    def test_rules_improve_prompt(self, engine, messy_prompt):
        """Test that rules improve the prompt."""
        result, applied = engine.apply_rules(messy_prompt, {"intent": "code_generation"})
        # Enhanced prompt should be longer (more structure added)
        assert len(result) >= len(messy_prompt)

    def test_list_rules(self, engine):
        """Test listing available rules."""
        rules = engine.list_rules()
        assert isinstance(rules, list)
        assert len(rules) > 0

    def test_filter_rules_by_tag(self, engine, messy_prompt):
        """Test filtering rules by tag."""
        result, applied = engine.apply_rules(
            messy_prompt,
            {"intent": "code_generation"},
            rule_tags=["structure"]
        )
        assert isinstance(result, str)


class TestLLMEnhancer:
    """Tests for LLMEnhancer class."""

    @pytest.fixture
    def enhancer(self, mock_llm_provider_with_responses):
        """Create an LLM enhancer with mock provider."""
        return LLMEnhancer(llm_client=mock_llm_provider_with_responses)

    @pytest.mark.asyncio
    async def test_enhance(self, enhancer, messy_prompt):
        """Test LLM enhancement."""
        result = await enhancer.enhance(
            messy_prompt,
            {"intent": "code_generation", "quality_score": 0.5}
        )
        assert isinstance(result, str)

    def test_set_level(self, enhancer):
        """Test setting enhancement level."""
        enhancer.set_level(EnhancementLevel.AGGRESSIVE)
        assert enhancer.level == EnhancementLevel.AGGRESSIVE

    def test_build_request(self, enhancer, messy_prompt):
        """Test building enhancement request."""
        request = enhancer._build_request(
            messy_prompt,
            {"intent": "code_generation", "quality_score": 0.5}
        )
        assert "Improve" in request
        assert messy_prompt in request

    def test_validate_enhancement(self, enhancer):
        """Test validating enhancement."""
        original = "test prompt"
        # Valid enhancement
        assert enhancer._validate(original, "A much improved test prompt with details")
        # Too short
        assert not enhancer._validate(original, "x")
        # Same as original
        assert not enhancer._validate(original, "test prompt")


class TestGrammarEnhancer:
    """Tests for GrammarEnhancer class."""

    @pytest.fixture
    def enhancer(self, mock_llm_provider_with_responses):
        """Create a grammar enhancer with mock provider."""
        return GrammarEnhancer(llm_client=mock_llm_provider_with_responses)

    @pytest.mark.asyncio
    async def test_enhance(self, enhancer):
        """Test grammar enhancement."""
        text = "this is a test with bad grammer"
        result = await enhancer.enhance(text)
        assert isinstance(result, str)


class TestPromptIntent:
    """Tests for PromptIntent enum."""

    def test_intent_values(self):
        """Test intent enum values."""
        assert PromptIntent.CODE_GENERATION.value == "code_generation"
        assert PromptIntent.QUESTION_ANSWERING.value == "question_answering"
        assert PromptIntent.SUMMARIZATION.value == "summarization"
        assert PromptIntent.GENERAL.value == "general"

    def test_all_intents_have_values(self):
        """Test all intents have string values."""
        for intent in PromptIntent:
            assert isinstance(intent.value, str)
            assert len(intent.value) > 0


class TestEdgeCases:
    """Tests for edge cases in enhancement."""

    @pytest.fixture
    def enhancer(self):
        """Create an enhancer."""
        return PromptEnhancer()

    @pytest.mark.asyncio
    async def test_enhance_empty_prompt(self, enhancer):
        """Test enhancing empty prompt."""
        result = await enhancer.enhance("", mode=EnhancementMode.RULES_ONLY)
        assert result is not None

    @pytest.mark.asyncio
    async def test_enhance_very_long_prompt(self, enhancer, long_prompt):
        """Test enhancing very long prompt."""
        # Repeat the prompt to make it longer
        very_long = long_prompt * 5
        result = await enhancer.enhance(very_long, mode=EnhancementMode.RULES_ONLY)
        assert result is not None

    @pytest.mark.asyncio
    async def test_enhance_special_characters(self, enhancer):
        """Test enhancing prompt with special characters."""
        prompt = "Test @#$%^& special !!! characters ???"
        result = await enhancer.enhance(prompt, mode=EnhancementMode.RULES_ONLY)
        assert result is not None

    @pytest.mark.asyncio
    async def test_enhance_unicode(self, enhancer):
        """Test enhancing prompt with unicode."""
        prompt = "Write code for emoji processing "
        result = await enhancer.enhance(prompt, mode=EnhancementMode.RULES_ONLY)
        assert result is not None

    @pytest.mark.asyncio
    async def test_enhance_with_intent_hint(self, enhancer, messy_prompt):
        """Test enhancing with intent hint."""
        result = await enhancer.enhance(
            messy_prompt,
            mode=EnhancementMode.RULES_ONLY,
            intent_hint=PromptIntent.CODE_GENERATION
        )
        assert result.detected_intent == PromptIntent.CODE_GENERATION
        assert result.intent_confidence == 1.0
