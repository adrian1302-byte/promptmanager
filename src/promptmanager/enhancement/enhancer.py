"""Main prompt enhancement orchestrator."""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple

from ..core.types import EnhancementResult
from ..core.exceptions import EnhancementError
from ..providers.base import LLMProvider

from .analyzers.intent_detector import IntentDetector, PromptIntent, IntentDetectionResult
from .analyzers.quality_scorer import QualityScorer, QualityScore
from .transformers.rule_engine import RuleEngine, TransformationRule
from .transformers.llm_enhancer import LLMEnhancer, EnhancementLevel, GrammarEnhancer


class EnhancementMode(Enum):
    """Enhancement processing modes."""
    RULES_ONLY = "rules_only"       # Only rule-based transformations
    LLM_ONLY = "llm_only"           # Only LLM-based enhancement
    HYBRID = "hybrid"               # Rules first, then LLM if needed
    ADAPTIVE = "adaptive"           # Choose based on quality score


@dataclass
class EnhancementConfig:
    """Configuration for prompt enhancement."""
    mode: EnhancementMode = EnhancementMode.HYBRID
    level: EnhancementLevel = EnhancementLevel.MODERATE
    quality_threshold: float = 0.7  # Skip LLM if quality >= threshold
    max_length_increase: float = 1.5  # Max prompt length increase ratio
    detect_intent: bool = True
    score_quality: bool = True
    fix_grammar: bool = True
    apply_best_practices: bool = True
    rule_tags: Optional[List[str]] = None  # Filter rules by tags


@dataclass
class DetailedEnhancementResult:
    """Detailed result from enhancement process."""
    original_prompt: str
    enhanced_prompt: str

    # Analysis results
    detected_intent: Optional[PromptIntent] = None
    intent_confidence: float = 0.0
    original_quality: Optional[QualityScore] = None
    final_quality: Optional[QualityScore] = None

    # Processing info
    applied_rules: List[str] = field(default_factory=list)
    llm_enhanced: bool = False
    grammar_fixed: bool = False

    # Metrics
    quality_improvement: float = 0.0
    token_change: int = 0
    processing_time_ms: float = 0.0

    # Suggestions
    suggestions: List[str] = field(default_factory=list)

    def to_enhancement_result(self) -> EnhancementResult:
        """Convert to standard EnhancementResult."""
        return EnhancementResult(
            original=self.original_prompt,
            processed=self.enhanced_prompt,
            improvements=self.applied_rules + (["llm_refinement"] if self.llm_enhanced else []),
            rules_applied=self.applied_rules,
            llm_enhanced=self.llm_enhanced,
            detected_intent=self.detected_intent.value if self.detected_intent else "",
            metadata={
                "intent": self.detected_intent.value if self.detected_intent else None,
                "intent_confidence": self.intent_confidence,
                "quality_improvement": self.quality_improvement,
                "grammar_fixed": self.grammar_fixed,
                "processing_time_ms": self.processing_time_ms,
                "suggestions": self.suggestions,
            }
        )


class PromptEnhancer:
    """
    Main orchestrator for prompt enhancement.

    Combines intent detection, quality scoring, rule-based transformations,
    and optional LLM-based refinement into a cohesive enhancement pipeline.
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        config: Optional[EnhancementConfig] = None
    ):
        """
        Initialize the prompt enhancer.

        Args:
            llm_provider: Optional LLM provider for LLM-based enhancements
            config: Enhancement configuration
        """
        self.config = config or EnhancementConfig()
        self.llm_provider = llm_provider

        # Initialize components
        self.intent_detector = IntentDetector(llm_client=llm_provider)
        self.quality_scorer = QualityScorer()
        self.rule_engine = RuleEngine()

        # LLM-based enhancers (require provider)
        self.llm_enhancer: Optional[LLMEnhancer] = None
        self.grammar_enhancer: Optional[GrammarEnhancer] = None

        if llm_provider:
            self.llm_enhancer = LLMEnhancer(
                llm_client=llm_provider,
                level=self.config.level,
                max_length_increase=self.config.max_length_increase
            )
            self.grammar_enhancer = GrammarEnhancer(llm_client=llm_provider)

    async def enhance(
        self,
        prompt: str,
        mode: Optional[EnhancementMode] = None,
        level: Optional[EnhancementLevel] = None,
        intent_hint: Optional[PromptIntent] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> DetailedEnhancementResult:
        """
        Enhance a prompt using the configured pipeline.

        Args:
            prompt: The prompt to enhance
            mode: Override enhancement mode
            level: Override enhancement level
            intent_hint: Known intent (skip detection)
            context: Additional context for enhancement

        Returns:
            DetailedEnhancementResult with enhanced prompt and metadata
        """
        import time
        start_time = time.time()

        mode = mode or self.config.mode
        level = level or self.config.level
        context = context or {}

        result = DetailedEnhancementResult(
            original_prompt=prompt,
            enhanced_prompt=prompt
        )

        try:
            # Step 1: Detect intent
            if intent_hint:
                result.detected_intent = intent_hint
                result.intent_confidence = 1.0
            elif self.config.detect_intent:
                intent_result = await self._detect_intent(prompt)
                result.detected_intent = intent_result.primary_intent
                result.intent_confidence = intent_result.confidence

            # Step 2: Score original quality
            if self.config.score_quality:
                result.original_quality = self.quality_scorer.score(prompt)

            # Build context for transformations
            enhancement_context = {
                "intent": result.detected_intent.value if result.detected_intent else "general",
                "quality_score": result.original_quality.overall_score if result.original_quality else 0.5,
                "original_length": len(prompt),
                **context
            }

            # Step 3: Apply enhancements based on mode
            current_prompt = prompt

            if mode == EnhancementMode.RULES_ONLY:
                current_prompt, result.applied_rules = self._apply_rules(
                    current_prompt, enhancement_context
                )

            elif mode == EnhancementMode.LLM_ONLY:
                if self.llm_enhancer:
                    current_prompt = await self._apply_llm_enhancement(
                        current_prompt, enhancement_context, level
                    )
                    result.llm_enhanced = current_prompt != prompt
                else:
                    raise EnhancementError("LLM enhancement requested but no provider configured")

            elif mode == EnhancementMode.HYBRID:
                # Rules first
                current_prompt, result.applied_rules = self._apply_rules(
                    current_prompt, enhancement_context
                )

                # Then LLM if quality still needs improvement
                if self.llm_enhancer:
                    quality_after_rules = self.quality_scorer.score(current_prompt)

                    if quality_after_rules.overall_score < self.config.quality_threshold:
                        enhancement_context["quality_score"] = quality_after_rules.overall_score
                        llm_result = await self._apply_llm_enhancement(
                            current_prompt, enhancement_context, level
                        )
                        if llm_result != current_prompt:
                            current_prompt = llm_result
                            result.llm_enhanced = True

            elif mode == EnhancementMode.ADAPTIVE:
                # Decide based on quality score
                if result.original_quality:
                    if result.original_quality.overall_score >= self.config.quality_threshold:
                        # High quality - minimal changes
                        current_prompt, result.applied_rules = self._apply_rules(
                            current_prompt, enhancement_context,
                            tags=["polish", "formatting"]
                        )
                    elif result.original_quality.overall_score >= 0.4:
                        # Medium quality - rules should suffice
                        current_prompt, result.applied_rules = self._apply_rules(
                            current_prompt, enhancement_context
                        )
                    else:
                        # Low quality - full pipeline
                        current_prompt, result.applied_rules = self._apply_rules(
                            current_prompt, enhancement_context
                        )
                        if self.llm_enhancer:
                            enhancement_context["quality_score"] = result.original_quality.overall_score
                            llm_result = await self._apply_llm_enhancement(
                                current_prompt, enhancement_context, EnhancementLevel.AGGRESSIVE
                            )
                            if llm_result != current_prompt:
                                current_prompt = llm_result
                                result.llm_enhanced = True

            # Step 4: Grammar fix if enabled
            if self.config.fix_grammar and self.grammar_enhancer:
                grammar_result = await self.grammar_enhancer.enhance(current_prompt)
                if grammar_result != current_prompt:
                    current_prompt = grammar_result
                    result.grammar_fixed = True

            result.enhanced_prompt = current_prompt

            # Step 5: Score final quality
            if self.config.score_quality:
                result.final_quality = self.quality_scorer.score(current_prompt)

                if result.original_quality:
                    result.quality_improvement = (
                        result.final_quality.overall_score -
                        result.original_quality.overall_score
                    )

            # Calculate token change
            result.token_change = len(current_prompt) - len(prompt)

            # Collect suggestions
            if result.final_quality and result.final_quality.suggestions:
                result.suggestions = result.final_quality.suggestions

            result.processing_time_ms = (time.time() - start_time) * 1000

            return result

        except Exception as e:
            raise EnhancementError(f"Enhancement failed: {str(e)}") from e

    def enhance_sync(
        self,
        prompt: str,
        mode: Optional[EnhancementMode] = None,
        level: Optional[EnhancementLevel] = None,
        intent_hint: Optional[PromptIntent] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> DetailedEnhancementResult:
        """Synchronous version of enhance."""
        return asyncio.run(self.enhance(prompt, mode, level, intent_hint, context))

    async def quick_enhance(self, prompt: str) -> str:
        """
        Quick enhancement using rules only.

        Args:
            prompt: The prompt to enhance

        Returns:
            Enhanced prompt string
        """
        result = await self.enhance(prompt, mode=EnhancementMode.RULES_ONLY)
        return result.enhanced_prompt

    def quick_enhance_sync(self, prompt: str) -> str:
        """Synchronous version of quick_enhance."""
        return asyncio.run(self.quick_enhance(prompt))

    async def analyze(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze a prompt without enhancing it.

        Args:
            prompt: The prompt to analyze

        Returns:
            Analysis results including intent and quality
        """
        # Detect intent
        intent_result = await self._detect_intent(prompt)

        # Score quality
        quality = self.quality_scorer.score(prompt)

        # Build scores dict from primary and secondary intents
        all_scores = {intent_result.primary_intent.value: intent_result.confidence}
        for intent, score in intent_result.secondary_intents:
            all_scores[intent.value] = score

        return {
            "intent": {
                "primary": intent_result.primary_intent.value,
                "confidence": intent_result.confidence,
                "all_scores": all_scores
            },
            "quality": {
                "overall_score": quality.overall_score,
                "clarity": quality.clarity_score,
                "structure": quality.structure_score,
                "completeness": quality.completeness_score,
                "specificity": quality.specificity_score,
                "grammar": quality.grammar_score,
                "issues": quality.issues,
                "suggestions": quality.suggestions
            },
            "statistics": {
                "length": len(prompt),
                "word_count": len(prompt.split()),
                "sentence_count": prompt.count('.') + prompt.count('!') + prompt.count('?')
            }
        }

    def analyze_sync(self, prompt: str) -> Dict[str, Any]:
        """Synchronous version of analyze."""
        return asyncio.run(self.analyze(prompt))

    def add_custom_rule(self, rule: TransformationRule) -> None:
        """Add a custom transformation rule."""
        self.rule_engine.add_rule(rule)

    def list_rules(self) -> List[Dict[str, Any]]:
        """List all available transformation rules."""
        return self.rule_engine.list_rules()

    def set_level(self, level: EnhancementLevel) -> None:
        """Change the enhancement level."""
        self.config.level = level
        if self.llm_enhancer:
            self.llm_enhancer.set_level(level)

    def set_mode(self, mode: EnhancementMode) -> None:
        """Change the enhancement mode."""
        self.config.mode = mode

    # Private methods
    async def _detect_intent(self, prompt: str) -> IntentDetectionResult:
        """Detect prompt intent."""
        # IntentDetector.detect is synchronous
        return self.intent_detector.detect(prompt)

    def _apply_rules(
        self,
        prompt: str,
        context: Dict[str, Any],
        tags: Optional[List[str]] = None
    ) -> Tuple[str, List[str]]:
        """Apply rule-based transformations."""
        rule_tags = tags or self.config.rule_tags
        return self.rule_engine.apply_rules(prompt, context, rule_tags)

    async def _apply_llm_enhancement(
        self,
        prompt: str,
        context: Dict[str, Any],
        level: EnhancementLevel
    ) -> str:
        """Apply LLM-based enhancement."""
        if not self.llm_enhancer:
            return prompt

        self.llm_enhancer.set_level(level)
        return await self.llm_enhancer.enhance(prompt, context)


# Convenience functions

async def enhance(
    prompt: str,
    mode: EnhancementMode = EnhancementMode.RULES_ONLY,
    level: EnhancementLevel = EnhancementLevel.MODERATE,
    llm_provider: Optional[LLMProvider] = None
) -> EnhancementResult:
    """
    Convenience function to enhance a prompt.

    Args:
        prompt: The prompt to enhance
        mode: Enhancement mode
        level: Enhancement level
        llm_provider: Optional LLM provider for LLM-based enhancements

    Returns:
        EnhancementResult with enhanced prompt
    """
    config = EnhancementConfig(mode=mode, level=level)
    enhancer = PromptEnhancer(llm_provider=llm_provider, config=config)
    result = await enhancer.enhance(prompt, mode=mode, level=level)
    return result.to_enhancement_result()


def enhance_sync(
    prompt: str,
    mode: EnhancementMode = EnhancementMode.RULES_ONLY,
    level: EnhancementLevel = EnhancementLevel.MODERATE,
    llm_provider: Optional[LLMProvider] = None
) -> EnhancementResult:
    """Synchronous version of enhance."""
    return asyncio.run(enhance(prompt, mode, level, llm_provider))


async def analyze_prompt(prompt: str, llm_provider: Optional[LLMProvider] = None) -> Dict[str, Any]:
    """
    Analyze a prompt to get intent and quality metrics.

    Args:
        prompt: The prompt to analyze
        llm_provider: Optional LLM provider for better intent detection

    Returns:
        Analysis results dictionary
    """
    enhancer = PromptEnhancer(llm_provider=llm_provider)
    return await enhancer.analyze(prompt)


def analyze_prompt_sync(prompt: str, llm_provider: Optional[LLMProvider] = None) -> Dict[str, Any]:
    """Synchronous version of analyze_prompt."""
    return asyncio.run(analyze_prompt(prompt, llm_provider))
