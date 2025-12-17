"""Intent detection for prompts."""

import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple


class PromptIntent(Enum):
    """Types of prompt intents."""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    CODE_EXPLANATION = "code_explanation"
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CREATIVE_WRITING = "creative_writing"
    DATA_EXTRACTION = "data_extraction"
    CLASSIFICATION = "classification"
    CONVERSATION = "conversation"
    INSTRUCTION_FOLLOWING = "instruction_following"
    ANALYSIS = "analysis"
    GENERAL = "general"


@dataclass
class IntentDetectionResult:
    """Result of intent detection."""
    primary_intent: PromptIntent
    confidence: float
    secondary_intents: List[Tuple[PromptIntent, float]] = field(default_factory=list)
    detected_signals: Dict[str, List[str]] = field(default_factory=dict)


class IntentDetector:
    """
    Detects the intent/use-case of a prompt.

    Uses a hybrid approach combining:
    1. Pattern matching (fast, high precision)
    2. Keyword analysis
    3. Structural analysis
    4. Optional LLM classification for ambiguous cases
    """

    # Intent patterns for rule-based detection
    INTENT_PATTERNS: Dict[PromptIntent, List[str]] = {
        PromptIntent.CODE_GENERATION: [
            r"\b(write|generate|create|implement|code|build|develop)\b.*\b(function|class|script|program|code|api|endpoint)\b",
            r"\b(python|javascript|java|typescript|go|rust|c\+\+)\b.*\b(code|function|class)\b",
            r"```\w*\n",  # Code blocks
            r"\b(implement|create)\b.*\b(method|algorithm|solution)\b",
        ],
        PromptIntent.CODE_REVIEW: [
            r"\b(review|analyze|check|audit|examine)\b.*\b(code|implementation|function|script)\b",
            r"\b(find|identify|spot)\b.*\b(bugs?|issues?|problems?|errors?|vulnerabilities)\b",
            r"\b(improve|optimize|refactor)\b.*\b(code|performance)\b",
        ],
        PromptIntent.CODE_EXPLANATION: [
            r"\b(explain|describe|what does)\b.*\b(code|function|script|this)\b",
            r"\bhow does\b.*\b(work|function|operate)\b",
            r"\b(walk me through|step.?by.?step)\b.*\b(code|logic)\b",
        ],
        PromptIntent.QUESTION_ANSWERING: [
            r"^(what|who|when|where|why|how|which|can|could|would|is|are|do|does)\b",
            r"\?$",
            r"\b(explain|describe|tell me|what is|what are|define)\b",
        ],
        PromptIntent.SUMMARIZATION: [
            r"\b(summarize|summary|summarise|tldr|brief|condense|shorten)\b",
            r"\b(key points|main points|highlights|overview)\b",
            r"\b(in (a )?few words|briefly)\b",
        ],
        PromptIntent.TRANSLATION: [
            r"\b(translate|translation|convert)\b.*\b(to|into|from)\b.*\b(english|spanish|french|german|chinese|japanese|korean|arabic|portuguese|russian|italian)\b",
            r"\b(in|to)\s+(english|spanish|french|german|chinese)\b",
        ],
        PromptIntent.CREATIVE_WRITING: [
            r"\b(write|compose|create|draft)\b.*\b(story|poem|essay|article|blog|narrative|novel|script)\b",
            r"\b(creative|imaginative|fiction|narrative)\b",
            r"\b(write about|story about|tale of)\b",
        ],
        PromptIntent.DATA_EXTRACTION: [
            r"\b(extract|parse|pull|get|retrieve)\b.*\b(data|information|fields|values|entities)\b",
            r"\b(json|xml|csv|table)\b.*\b(format|output|extract|convert)\b",
            r"\b(find all|list all|extract all)\b",
        ],
        PromptIntent.CLASSIFICATION: [
            r"\b(classify|categorize|label|tag|identify the type|determine the category)\b",
            r"\b(sentiment|category|class|type|kind)\b.*\b(of|for)\b",
            r"\b(positive|negative|neutral)\b.*\b(sentiment|tone)\b",
        ],
        PromptIntent.ANALYSIS: [
            r"\b(analyze|analyse|evaluate|assess|examine|investigate)\b",
            r"\b(pros and cons|advantages|disadvantages|compare|contrast)\b",
            r"\b(strengths|weaknesses|swot)\b",
        ],
    }

    def __init__(self, llm_client=None):
        """
        Initialize the intent detector.

        Args:
            llm_client: Optional LLM client for ambiguous cases
        """
        self.llm_client = llm_client
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for efficiency."""
        self._compiled_patterns = {
            intent: [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in patterns]
            for intent, patterns in self.INTENT_PATTERNS.items()
        }

    def detect(self, prompt: str) -> IntentDetectionResult:
        """
        Detect the intent of a prompt.

        Args:
            prompt: The prompt to analyze

        Returns:
            IntentDetectionResult with detected intent and confidence
        """
        scores: Dict[PromptIntent, float] = {intent: 0.0 for intent in PromptIntent}
        signals: Dict[str, List[str]] = {}

        # Stage 1: Pattern matching
        for intent, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(prompt)
                if matches:
                    scores[intent] += len(matches) * 0.3
                    signal_key = intent.value
                    if signal_key not in signals:
                        signals[signal_key] = []
                    for match in matches[:3]:
                        if isinstance(match, tuple):
                            match = match[0]
                        signals[signal_key].append(str(match)[:50])

        # Stage 2: Structural analysis
        scores = self._analyze_structure(prompt, scores)

        # Stage 3: Keyword frequency
        scores = self._analyze_keywords(prompt, scores)

        # Stage 4: LLM classification for low confidence
        max_score = max(scores.values()) if scores.values() else 0
        if max_score < 0.5 and self.llm_client:
            llm_result = self._llm_classify(prompt)
            if llm_result:
                scores[llm_result] = max(scores[llm_result], 0.6)

        # Normalize scores
        total = sum(scores.values()) or 1
        normalized = {k: v / total for k, v in scores.items()}

        # Sort by score
        sorted_intents = sorted(
            normalized.items(),
            key=lambda x: x[1],
            reverse=True
        )

        primary = sorted_intents[0] if sorted_intents else (PromptIntent.GENERAL, 0.0)

        # Filter secondary intents (score > 0.1)
        secondary = [
            (intent, score) for intent, score in sorted_intents[1:5]
            if score > 0.1
        ]

        return IntentDetectionResult(
            primary_intent=primary[0],
            confidence=primary[1],
            secondary_intents=secondary,
            detected_signals=signals
        )

    def _analyze_structure(
        self,
        prompt: str,
        scores: Dict[PromptIntent, float]
    ) -> Dict[PromptIntent, float]:
        """Analyze structural elements."""
        # Code blocks indicate code-related intent
        if "```" in prompt:
            scores[PromptIntent.CODE_GENERATION] += 0.2
            scores[PromptIntent.CODE_REVIEW] += 0.15
            scores[PromptIntent.CODE_EXPLANATION] += 0.15

        # Questions ending with ?
        if prompt.strip().endswith("?"):
            scores[PromptIntent.QUESTION_ANSWERING] += 0.25

        # Multiple questions suggest Q&A
        question_count = prompt.count("?")
        if question_count > 1:
            scores[PromptIntent.QUESTION_ANSWERING] += 0.1 * min(question_count, 3)

        # Structured output requests
        if re.search(r"\b(json|xml|csv|table|list)\b", prompt, re.IGNORECASE):
            scores[PromptIntent.DATA_EXTRACTION] += 0.2

        # Long prompts with context suggest analysis
        if len(prompt) > 1000:
            scores[PromptIntent.ANALYSIS] += 0.1
            scores[PromptIntent.SUMMARIZATION] += 0.1

        return scores

    def _analyze_keywords(
        self,
        prompt: str,
        scores: Dict[PromptIntent, float]
    ) -> Dict[PromptIntent, float]:
        """Analyze keyword frequency."""
        prompt_lower = prompt.lower()

        # Code keywords
        code_keywords = ["function", "class", "method", "variable", "return", "import", "def", "const", "let"]
        code_count = sum(1 for kw in code_keywords if kw in prompt_lower)
        if code_count >= 2:
            scores[PromptIntent.CODE_GENERATION] += 0.15

        # Analysis keywords
        analysis_keywords = ["analyze", "evaluate", "compare", "assess", "examine"]
        if any(kw in prompt_lower for kw in analysis_keywords):
            scores[PromptIntent.ANALYSIS] += 0.15

        # Creative keywords
        creative_keywords = ["creative", "story", "poem", "write about", "imagine"]
        if any(kw in prompt_lower for kw in creative_keywords):
            scores[PromptIntent.CREATIVE_WRITING] += 0.15

        return scores

    def _llm_classify(self, prompt: str) -> Optional[PromptIntent]:
        """Use LLM for classification when uncertain."""
        if not self.llm_client:
            return None

        classification_prompt = f"""Classify this prompt into ONE category:
- code_generation: Writing/generating code
- code_review: Reviewing/analyzing code
- question_answering: Answering questions
- summarization: Summarizing content
- translation: Translating languages
- creative_writing: Creative content
- data_extraction: Extracting data
- classification: Categorizing items
- analysis: Analyzing/evaluating
- general: Other

Prompt: "{prompt[:500]}"

Reply with ONLY the category name:"""

        try:
            response = self.llm_client.complete_text_sync(
                classification_prompt,
                max_tokens=20
            )
            category = response.strip().lower().replace(" ", "_")

            # Map to enum
            for intent in PromptIntent:
                if intent.value == category:
                    return intent
        except Exception:
            pass

        return None

    def detect_multiple(
        self,
        prompts: List[str]
    ) -> List[IntentDetectionResult]:
        """Detect intent for multiple prompts."""
        return [self.detect(p) for p in prompts]
