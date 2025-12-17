"""LLM-based prompt enhancement."""

from typing import Optional, Dict, Any
from enum import Enum


class EnhancementLevel(Enum):
    """Enhancement intensity levels."""
    MINIMAL = "minimal"     # Only critical fixes
    LIGHT = "light"         # Minor improvements
    MODERATE = "moderate"   # Balanced improvements
    AGGRESSIVE = "aggressive"  # Comprehensive rewriting


class LLMEnhancer:
    """
    LLM-based prompt enhancement for nuanced improvements.

    Handles context-dependent improvements that rule-based
    approaches cannot address.
    """

    SYSTEM_PROMPT = """You are an expert prompt engineer. Your task is to improve prompts
while preserving their original intent. Apply these best practices:

1. CLARITY: Make instructions explicit and unambiguous
2. STRUCTURE: Use clear sections and formatting where helpful
3. CONTEXT: Add relevant context where missing
4. SPECIFICITY: Replace vague terms with specific requirements
5. COMPLETENESS: Ensure all necessary information is included

IMPORTANT RULES:
- Preserve the original intent and meaning
- Do not add unnecessary complexity
- Keep improvements proportional to the original prompt length
- Do not add examples unless the original has them
- Output ONLY the improved prompt, no explanations or commentary"""

    def __init__(
        self,
        llm_client,
        level: EnhancementLevel = EnhancementLevel.MODERATE,
        max_length_increase: float = 1.5
    ):
        """
        Initialize the LLM enhancer.

        Args:
            llm_client: LLM provider client
            level: Enhancement intensity level
            max_length_increase: Maximum allowed length increase ratio
        """
        self.llm_client = llm_client
        self.level = level
        self.max_length_increase = max_length_increase

    async def enhance(
        self,
        prompt: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Enhance a prompt using LLM.

        Args:
            prompt: The prompt to enhance
            context: Context including intent, quality scores, etc.

        Returns:
            Enhanced prompt
        """
        enhancement_request = self._build_request(prompt, context)

        try:
            response = await self.llm_client.complete(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": enhancement_request}
                ],
                max_tokens=len(prompt) * 3,
                temperature=0.3  # Lower for consistency
            )

            enhanced = response.content.strip()

            # Clean up common LLM artifacts
            enhanced = self._clean_response(enhanced)

            # Validate enhancement
            if self._validate(prompt, enhanced):
                return enhanced
            else:
                return prompt

        except Exception:
            return prompt

    def enhance_sync(
        self,
        prompt: str,
        context: Dict[str, Any]
    ) -> str:
        """Synchronous version of enhance."""
        import asyncio
        return asyncio.run(self.enhance(prompt, context))

    def _clean_response(self, response: str) -> str:
        """Clean common LLM artifacts from the response."""
        import re

        cleaned = response.strip()

        # Remove triple quotes at start (with optional whitespace/newlines)
        cleaned = re.sub(r'^["\']{3}\s*', '', cleaned)
        # Remove triple quotes at end
        cleaned = re.sub(r'\s*["\']{3}$', '', cleaned)

        # Remove single/double quotes wrapping entire response
        if (cleaned.startswith('"') and cleaned.endswith('"')) or \
           (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1].strip()

        # Remove trailing single/double quote (common LLM artifact)
        cleaned = re.sub(r'["\']$', '', cleaned)

        # Remove markdown code blocks
        cleaned = re.sub(r'^```\w*\n?', '', cleaned)
        cleaned = re.sub(r'\n?```$', '', cleaned)
        cleaned = cleaned.strip()

        # Remove common prefixes
        prefixes_to_remove = [
            "Here is the improved prompt:",
            "Here's the improved prompt:",
            "Improved prompt:",
            "Enhanced prompt:",
            "The improved prompt is:",
        ]
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()

        return cleaned

    def _build_request(self, prompt: str, context: Dict) -> str:
        """Build the enhancement request."""
        intent = context.get("intent", "general")
        quality_score = context.get("quality_score", 0.5)

        level_instructions = {
            EnhancementLevel.MINIMAL: "Make only critical fixes. Keep changes minimal.",
            EnhancementLevel.LIGHT: "Make minor improvements for clarity. Preserve most of the original.",
            EnhancementLevel.MODERATE: "Improve clarity, structure, and completeness. Balanced changes.",
            EnhancementLevel.AGGRESSIVE: "Comprehensively improve the prompt. Restructure if needed.",
        }

        # Allow more length for enhancement - at least 5x the original
        max_length = int(len(prompt) * max(self.max_length_increase, 5.0))

        request = f"""Improve the following {intent} prompt.

Enhancement level: {self.level.value}
Instructions: {level_instructions[self.level]}

Current quality score: {quality_score:.1%}
Maximum length: {max_length} characters

Original prompt:
{prompt}

Output ONLY the improved prompt (no quotes, no explanations):"""

        return request

    def _validate(self, original: str, enhanced: str) -> bool:
        """Validate the enhancement is acceptable."""
        # Check not empty
        if not enhanced or len(enhanced) < 10:
            return False

        # Check length constraint - allow up to 5x increase for short prompts
        max_len = len(original) * max(self.max_length_increase, 5.0)
        if len(enhanced) > max_len:
            return False

        # Check not too short (at least 50% of original, or 10 chars minimum)
        min_len = max(10, len(original) * 0.5)
        if len(enhanced) < min_len:
            return False

        # Check not identical (normalized comparison)
        if enhanced.strip().lower() == original.strip().lower():
            return False

        # Check for common LLM issues (these indicate the LLM didn't follow instructions)
        bad_patterns = [
            "here is the improved",
            "i have improved",
            "as an ai",
            "i cannot",
            "i'm sorry",
        ]
        enhanced_lower = enhanced.lower()
        if any(p in enhanced_lower for p in bad_patterns):
            return False

        return True

    def set_level(self, level: EnhancementLevel) -> None:
        """Change enhancement level."""
        self.level = level


class GrammarEnhancer:
    """Specialized enhancer for grammar and clarity."""

    GRAMMAR_PROMPT = """Fix grammar, spelling, and clarity issues in the following text.
Make minimal changes - only fix actual errors, do not rephrase or restructure.

Text:
\"\"\"
{text}
\"\"\"

Return ONLY the corrected text:"""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    async def enhance(self, text: str) -> str:
        """Fix grammar and clarity issues."""
        try:
            response = await self.llm_client.complete(
                messages=[
                    {"role": "user", "content": self.GRAMMAR_PROMPT.format(text=text)}
                ],
                max_tokens=len(text) * 2,
                temperature=0.1
            )
            return response.content.strip()
        except Exception:
            return text

    def enhance_sync(self, text: str) -> str:
        """Synchronous version."""
        import asyncio
        return asyncio.run(self.enhance(text))
