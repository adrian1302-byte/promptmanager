"""Quality scoring for prompts."""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class QualityScore:
    """Detailed quality scores for a prompt."""
    clarity_score: float = 0.0       # How clear and unambiguous (0-1)
    structure_score: float = 0.0     # How well organized (0-1)
    completeness_score: float = 0.0  # Whether all needed info present (0-1)
    specificity_score: float = 0.0   # How specific vs vague (0-1)
    grammar_score: float = 0.0       # Grammar and spelling (0-1)
    overall_score: float = 0.0       # Weighted average (0-1)
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "clarity": self.clarity_score,
            "structure": self.structure_score,
            "completeness": self.completeness_score,
            "specificity": self.specificity_score,
            "grammar": self.grammar_score,
            "overall": self.overall_score,
            "issues": self.issues,
            "suggestions": self.suggestions,
        }


class QualityScorer:
    """
    Scores prompt quality across multiple dimensions.

    Uses heuristics and patterns - no LLM required.
    All scores are normalized to 0-1 range.
    """

    # Weights for overall score
    WEIGHTS = {
        "clarity": 0.25,
        "structure": 0.20,
        "completeness": 0.25,
        "specificity": 0.20,
        "grammar": 0.10,
    }

    def score(self, prompt: str) -> QualityScore:
        """
        Score a prompt's quality.

        Args:
            prompt: The prompt to score

        Returns:
            QualityScore with detailed scores and suggestions
        """
        issues: List[str] = []
        suggestions: List[str] = []

        clarity = self._score_clarity(prompt, issues, suggestions)
        structure = self._score_structure(prompt, issues, suggestions)
        completeness = self._score_completeness(prompt, issues, suggestions)
        specificity = self._score_specificity(prompt, issues, suggestions)
        grammar = self._score_grammar(prompt, issues, suggestions)

        overall = (
            clarity * self.WEIGHTS["clarity"] +
            structure * self.WEIGHTS["structure"] +
            completeness * self.WEIGHTS["completeness"] +
            specificity * self.WEIGHTS["specificity"] +
            grammar * self.WEIGHTS["grammar"]
        )

        return QualityScore(
            clarity_score=round(clarity, 3),
            structure_score=round(structure, 3),
            completeness_score=round(completeness, 3),
            specificity_score=round(specificity, 3),
            grammar_score=round(grammar, 3),
            overall_score=round(overall, 3),
            issues=issues,
            suggestions=suggestions
        )

    def _score_clarity(
        self,
        prompt: str,
        issues: List[str],
        suggestions: List[str]
    ) -> float:
        """Score clarity of instructions."""
        score = 1.0

        # Penalize very short prompts
        if len(prompt) < 20:
            score -= 0.3
            issues.append("Prompt is too short")
            suggestions.append("Add more context and details")

        # Penalize ambiguous pronouns
        ambiguous = ["it", "this", "that", "thing", "stuff", "something", "things"]
        ambiguous_count = sum(
            len(re.findall(rf"\b{w}\b", prompt, re.IGNORECASE))
            for w in ambiguous
        )
        if ambiguous_count > 2:
            penalty = 0.1 * min(ambiguous_count - 2, 3)
            score -= penalty
            issues.append(f"Contains {ambiguous_count} ambiguous references")
            suggestions.append("Replace pronouns with specific nouns")

        # Check for clear action verbs
        action_verbs = [
            "write", "generate", "create", "explain", "summarize",
            "analyze", "list", "describe", "compare", "translate",
            "find", "calculate", "implement", "design", "review"
        ]
        has_action = any(v in prompt.lower() for v in action_verbs)
        if not has_action:
            score -= 0.2
            issues.append("No clear action verb")
            suggestions.append("Start with a clear action verb (write, create, explain, etc.)")

        # Check for vague requests
        vague_patterns = [
            r"\b(help me with|do something|make it better|improve)\b",
            r"\b(anything|whatever|somehow)\b",
        ]
        for pattern in vague_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                score -= 0.1
                issues.append("Contains vague request language")
                suggestions.append("Be more specific about what you want")
                break

        return max(0, min(1, score))

    def _score_structure(
        self,
        prompt: str,
        issues: List[str],
        suggestions: List[str]
    ) -> float:
        """Score structural organization."""
        score = 0.5  # Start at baseline

        # Reward section headers (markdown or bold)
        header_patterns = [
            r"^#+\s+\w",          # Markdown headers
            r"^\*\*[A-Z]",        # Bold headers
            r"^[A-Z][A-Za-z\s]+:$",  # Label: style
        ]
        header_count = sum(
            len(re.findall(p, prompt, re.MULTILINE))
            for p in header_patterns
        )
        if header_count > 0:
            score += min(0.2, header_count * 0.05)

        # Reward bullet points or numbered lists
        list_patterns = [
            r"^\s*[-*]\s+\w",     # Bullet points
            r"^\s*\d+[.)]\s+\w",  # Numbered lists
        ]
        list_count = sum(
            len(re.findall(p, prompt, re.MULTILINE))
            for p in list_patterns
        )
        if list_count > 0:
            score += min(0.15, list_count * 0.03)

        # Reward paragraph breaks for longer prompts
        line_count = prompt.count("\n") + 1
        words = len(prompt.split())

        if words > 50:
            if line_count > 3:
                score += 0.1
            elif line_count <= 1:
                score -= 0.15
                issues.append("Long prompt lacks line breaks")
                suggestions.append("Break into paragraphs or sections")

        # Penalize wall of text
        if len(prompt) > 500 and line_count < 3:
            score -= 0.2
            issues.append("Text wall without structure")
            suggestions.append("Add headers and bullet points for clarity")

        return max(0, min(1, score))

    def _score_completeness(
        self,
        prompt: str,
        issues: List[str],
        suggestions: List[str]
    ) -> float:
        """Score completeness of information."""
        score = 0.6  # Start at baseline

        prompt_lower = prompt.lower()

        # Check for context
        context_indicators = [
            "given", "context", "background", "scenario", "situation",
            "assuming", "based on", "considering"
        ]
        has_context = any(c in prompt_lower for c in context_indicators)
        if has_context:
            score += 0.1

        # Check for expected output format
        format_indicators = [
            "format", "output", "return", "response", "as",
            "should be", "in the form of", "structured as"
        ]
        has_format = any(f in prompt_lower for f in format_indicators)
        if has_format:
            score += 0.15
        else:
            suggestions.append("Specify the expected output format")

        # Check for constraints or requirements
        constraint_indicators = [
            "must", "should", "require", "need", "constraint", "limit",
            "maximum", "minimum", "at least", "at most", "only"
        ]
        has_constraints = any(c in prompt_lower for c in constraint_indicators)
        if has_constraints:
            score += 0.1

        # Check for examples
        if "example" in prompt_lower or "e.g." in prompt_lower:
            score += 0.1

        return max(0, min(1, score))

    def _score_specificity(
        self,
        prompt: str,
        issues: List[str],
        suggestions: List[str]
    ) -> float:
        """Score specificity vs vagueness."""
        score = 0.5  # Start at baseline

        # Reward specific numbers/quantities
        if re.search(r"\d+", prompt):
            score += 0.1

        # Reward technical terms (CamelCase, snake_case)
        technical_patterns = [
            r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b",  # CamelCase
            r"\b[a-z]+_[a-z]+\b",  # snake_case
            r"\b(?:API|SDK|JSON|XML|HTML|CSS|SQL|HTTP|REST)\b",
        ]
        tech_count = sum(
            len(re.findall(p, prompt))
            for p in technical_patterns
        )
        if tech_count > 0:
            score += min(0.15, tech_count * 0.03)

        # Penalize vague qualifiers
        vague_terms = [
            "good", "nice", "better", "best", "some", "few", "many",
            "various", "several", "certain", "proper", "appropriate"
        ]
        vague_count = sum(
            len(re.findall(rf"\b{t}\b", prompt, re.IGNORECASE))
            for t in vague_terms
        )
        if vague_count > 2:
            penalty = 0.1 * min(vague_count - 2, 3)
            score -= penalty
            issues.append(f"Contains {vague_count} vague qualifiers")
            suggestions.append("Replace vague terms with specific requirements")

        # Reward named entities (capitalized words that aren't sentence starts)
        sentences = re.split(r'[.!?]\s+', prompt)
        named_entities = 0
        for sentence in sentences:
            words = sentence.split()
            for i, word in enumerate(words):
                if i > 0 and word[0].isupper() and word.isalpha():
                    named_entities += 1

        if named_entities > 0:
            score += min(0.1, named_entities * 0.02)

        return max(0, min(1, score))

    def _score_grammar(
        self,
        prompt: str,
        issues: List[str],
        suggestions: List[str]
    ) -> float:
        """Score grammar and spelling (basic heuristics)."""
        score = 1.0

        # Check for double spaces
        if "  " in prompt:
            score -= 0.05

        # Check sentence capitalization
        sentences = re.split(r'[.!?]\s+', prompt)
        lowercase_starts = sum(
            1 for s in sentences
            if s and s[0].islower()
        )
        if lowercase_starts > 0:
            penalty = 0.1 * min(lowercase_starts, 3)
            score -= penalty
            issues.append("Some sentences not capitalized")

        # Check for basic punctuation at end
        stripped = prompt.rstrip()
        if stripped and not any(stripped.endswith(p) for p in ".!?:"):
            score -= 0.1
            suggestions.append("End with appropriate punctuation")

        # Check for repeated words
        words = prompt.lower().split()
        for i in range(len(words) - 1):
            if words[i] == words[i + 1] and len(words[i]) > 2:
                score -= 0.05
                issues.append(f"Repeated word: '{words[i]}'")
                break

        # Check for common typos/issues
        common_issues = [
            (r"\bi\b", "Lowercase 'i' should be capitalized"),
            (r"\bteh\b", "Possible typo: 'teh' -> 'the'"),
            (r"\bwith out\b", "Should be 'without'"),
        ]
        for pattern, message in common_issues:
            if re.search(pattern, prompt):
                score -= 0.05
                issues.append(message)

        return max(0, min(1, score))

    def compare(
        self,
        original: str,
        enhanced: str
    ) -> Dict[str, float]:
        """
        Compare quality scores between original and enhanced prompts.

        Returns dict with score differences.
        """
        original_score = self.score(original)
        enhanced_score = self.score(enhanced)

        return {
            "clarity_improvement": enhanced_score.clarity_score - original_score.clarity_score,
            "structure_improvement": enhanced_score.structure_score - original_score.structure_score,
            "completeness_improvement": enhanced_score.completeness_score - original_score.completeness_score,
            "specificity_improvement": enhanced_score.specificity_score - original_score.specificity_score,
            "grammar_improvement": enhanced_score.grammar_score - original_score.grammar_score,
            "overall_improvement": enhanced_score.overall_score - original_score.overall_score,
        }
