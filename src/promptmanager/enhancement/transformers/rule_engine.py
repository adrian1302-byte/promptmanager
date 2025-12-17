"""Rule-based transformation engine for prompt enhancement."""

import re
from dataclasses import dataclass, field
from typing import List, Callable, Optional, Dict, Any, Tuple
from enum import Enum


class RulePriority(Enum):
    """Priority levels for transformation rules."""
    CRITICAL = 0    # Must apply first
    HIGH = 1        # Important improvements
    MEDIUM = 2      # Nice-to-have improvements
    LOW = 3         # Minor polish


@dataclass
class TransformationRule:
    """A single transformation rule."""
    name: str
    description: str
    priority: RulePriority
    applicable_intents: List[str]  # Empty = applies to all
    condition: Callable[[str, Dict], bool]
    transform: Callable[[str, Dict], str]
    tags: List[str] = field(default_factory=list)


class RuleEngine:
    """
    Rule-based engine for deterministic prompt transformations.

    Provides fast, predictable improvements without LLM calls.
    """

    def __init__(self):
        self.rules: List[TransformationRule] = []
        self._load_builtin_rules()

    def _load_builtin_rules(self) -> None:
        """Load built-in transformation rules."""

        # Rule 1: Add clear task instruction prefix
        self.rules.append(TransformationRule(
            name="add_task_prefix",
            description="Add clear task instruction at the beginning",
            priority=RulePriority.HIGH,
            applicable_intents=[],
            condition=lambda p, c: not self._has_clear_instruction(p),
            transform=self._add_task_prefix,
            tags=["structure", "clarity"]
        ))

        # Rule 2: Structure code generation prompts
        self.rules.append(TransformationRule(
            name="structure_code_prompt",
            description="Add structure to code generation prompts",
            priority=RulePriority.HIGH,
            applicable_intents=["code_generation", "code_review"],
            condition=lambda p, c: (
                c.get("intent") in ["code_generation", "code_review"] and
                "##" not in p
            ),
            transform=self._structure_code_prompt,
            tags=["structure", "code"]
        ))

        # Rule 3: Add output format specification
        self.rules.append(TransformationRule(
            name="add_output_format",
            description="Specify expected output format",
            priority=RulePriority.MEDIUM,
            applicable_intents=[],
            condition=lambda p, c: not self._has_output_format(p),
            transform=self._add_output_format,
            tags=["structure", "clarity"]
        ))

        # Rule 4: Fix common grammar issues
        self.rules.append(TransformationRule(
            name="fix_grammar",
            description="Fix common grammar and punctuation issues",
            priority=RulePriority.LOW,
            applicable_intents=[],
            condition=lambda p, c: True,
            transform=self._fix_grammar,
            tags=["grammar", "polish"]
        ))

        # Rule 5: Add chain-of-thought for complex tasks
        self.rules.append(TransformationRule(
            name="add_cot_instruction",
            description="Add step-by-step reasoning instruction",
            priority=RulePriority.MEDIUM,
            applicable_intents=["question_answering", "analysis", "code_generation"],
            condition=self._is_complex_task,
            transform=self._add_cot_instruction,
            tags=["reasoning", "best_practice"]
        ))

        # Rule 6: Normalize whitespace
        self.rules.append(TransformationRule(
            name="normalize_whitespace",
            description="Clean up excessive whitespace",
            priority=RulePriority.LOW,
            applicable_intents=[],
            condition=lambda p, c: "  " in p or "\n\n\n" in p,
            transform=self._normalize_whitespace,
            tags=["formatting", "polish"]
        ))

        # Rule 7: Add specificity to vague requests
        self.rules.append(TransformationRule(
            name="improve_specificity",
            description="Add specificity prompts for vague requests",
            priority=RulePriority.MEDIUM,
            applicable_intents=[],
            condition=self._is_vague_request,
            transform=self._add_specificity_hints,
            tags=["clarity", "specificity"]
        ))

    def apply_rules(
        self,
        prompt: str,
        context: Dict[str, Any],
        rule_tags: Optional[List[str]] = None
    ) -> Tuple[str, List[str]]:
        """
        Apply applicable rules to the prompt.

        Args:
            prompt: The prompt to transform
            context: Context including detected intent, scores, etc.
            rule_tags: Optional filter by tags

        Returns:
            Tuple of (transformed_prompt, list_of_applied_rule_names)
        """
        # Filter rules by tags
        rules = self.rules
        if rule_tags:
            rules = [r for r in rules if any(t in r.tags for t in rule_tags)]

        # Sort by priority
        rules = sorted(rules, key=lambda r: r.priority.value)

        applied_rules = []
        current_prompt = prompt

        for rule in rules:
            # Check intent applicability
            if rule.applicable_intents:
                if context.get("intent") not in rule.applicable_intents:
                    continue

            # Check condition
            if rule.condition(current_prompt, context):
                try:
                    new_prompt = rule.transform(current_prompt, context)
                    if new_prompt != current_prompt:
                        current_prompt = new_prompt
                        applied_rules.append(rule.name)
                except Exception:
                    # Skip failed rules
                    continue

        return current_prompt, applied_rules

    def add_rule(self, rule: TransformationRule) -> None:
        """Add a custom rule."""
        self.rules.append(rule)

    def list_rules(self) -> List[Dict[str, Any]]:
        """List all rules with metadata."""
        return [
            {
                "name": r.name,
                "description": r.description,
                "priority": r.priority.value,
                "tags": r.tags,
                "intents": r.applicable_intents or ["all"],
            }
            for r in self.rules
        ]

    # Helper methods
    def _has_clear_instruction(self, prompt: str) -> bool:
        """Check if prompt has a clear instruction verb."""
        patterns = [
            r"^(please\s+)?(write|generate|create|explain|summarize|analyze|review|translate|find|list|describe|implement)",
            r"^(your task|you are|you will|i need you to|can you|could you|please)",
        ]
        for pattern in patterns:
            if re.match(pattern, prompt.strip(), re.IGNORECASE):
                return True
        return False

    def _has_output_format(self, prompt: str) -> bool:
        """Check if prompt specifies output format."""
        keywords = [
            "format", "output", "return", "respond", "as json", "as a list",
            "in markdown", "structured", "should include", "should be",
            "in the form", "provide", "give me"
        ]
        prompt_lower = prompt.lower()
        return any(kw in prompt_lower for kw in keywords)

    def _is_complex_task(self, prompt: str, context: Dict) -> bool:
        """Determine if task is complex enough for CoT."""
        complexity_signals = [
            len(prompt) > 200,
            "step" in prompt.lower(),
            "multiple" in prompt.lower(),
            "compare" in prompt.lower(),
            "analyze" in prompt.lower(),
            "explain" in prompt.lower() and "why" in prompt.lower(),
            prompt.count("and") > 2,
            "?" in prompt and len(prompt) > 100,
        ]
        return sum(complexity_signals) >= 2

    def _is_vague_request(self, prompt: str, context: Dict) -> bool:
        """Check if request is vague."""
        vague_patterns = [
            r"\b(help me with|do something|make it better|improve)\b",
            r"\b(good|nice|better)\b",
            r"^(can you|could you)\s+\w+\s*\?*$",
        ]
        for pattern in vague_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                return True
        return len(prompt.split()) < 10

    # Transformation methods
    def _add_task_prefix(self, prompt: str, context: Dict) -> str:
        """Add a clear task instruction prefix."""
        intent = context.get("intent", "general")

        prefixes = {
            "code_generation": "Write code to accomplish the following:\n\n",
            "question_answering": "Please answer the following question:\n\n",
            "summarization": "Summarize the following:\n\n",
            "code_review": "Review the following code and provide feedback:\n\n",
            "translation": "Translate the following:\n\n",
            "analysis": "Analyze the following:\n\n",
            "creative_writing": "Write the following:\n\n",
            "data_extraction": "Extract the following information:\n\n",
            "general": "Complete the following task:\n\n",
        }

        prefix = prefixes.get(intent, prefixes["general"])

        # Don't add if prompt already starts with similar phrase
        if prompt.strip().lower().startswith(prefix.lower()[:20]):
            return prompt

        return prefix + prompt

    def _structure_code_prompt(self, prompt: str, context: Dict) -> str:
        """Add structure to code generation prompts."""
        # Detect language
        lang_match = re.search(
            r"\b(python|javascript|typescript|java|go|rust|c\+\+|ruby|php|swift|kotlin)\b",
            prompt,
            re.IGNORECASE
        )

        sections = ["## Task", prompt, ""]

        # Add requirements section
        sections.append("## Requirements")
        sections.append("- Write clean, well-documented code")
        sections.append("- Follow best practices and conventions")

        if lang_match:
            lang = lang_match.group(1)
            sections.append(f"- Use {lang} programming language")

        sections.append("")
        sections.append("## Expected Output")
        sections.append("Provide the complete implementation with comments explaining key parts.")

        return "\n".join(sections)

    def _add_output_format(self, prompt: str, context: Dict) -> str:
        """Add output format specification."""
        intent = context.get("intent", "general")

        format_specs = {
            "data_extraction": "\n\nProvide the output as valid JSON.",
            "summarization": "\n\nProvide a concise summary in 2-3 paragraphs or bullet points.",
            "classification": "\n\nReturn the classification with confidence level.",
            "analysis": "\n\nProvide a structured analysis with clear sections.",
            "code_generation": "",  # Handled by structure_code_prompt
            "question_answering": "\n\nProvide a clear, well-organized response.",
        }

        suffix = format_specs.get(intent, "\n\nProvide a clear, well-organized response.")

        if not suffix:
            return prompt

        return prompt + suffix

    def _fix_grammar(self, prompt: str, context: Dict) -> str:
        """Apply common grammar fixes."""
        result = prompt

        # Fix double spaces
        result = re.sub(r" {2,}", " ", result)

        # Fix missing space after punctuation
        result = re.sub(r"([.!?])([A-Z])", r"\1 \2", result)

        # Capitalize first letter
        if result and result[0].islower():
            result = result[0].upper() + result[1:]

        # Ensure ends with punctuation for statements
        result = result.rstrip()
        if result and result[-1] not in ".!?:\"'`":
            # Don't add if it's a code block or list
            if not result.endswith("```") and not re.search(r"^\s*[-*]\s", result.split("\n")[-1]):
                result = result + "."

        return result

    def _add_cot_instruction(self, prompt: str, context: Dict) -> str:
        """Add chain-of-thought instruction."""
        # Check if CoT already present
        if "step by step" in prompt.lower() or "step-by-step" in prompt.lower():
            return prompt

        cot_instruction = (
            "\n\nThink through this step by step:\n"
            "1. First, understand what is being asked\n"
            "2. Break down the problem into parts\n"
            "3. Solve each part systematically\n"
            "4. Verify and provide the final answer"
        )

        return prompt + cot_instruction

    def _normalize_whitespace(self, prompt: str, context: Dict) -> str:
        """Normalize whitespace."""
        # Replace multiple spaces
        result = re.sub(r" {2,}", " ", prompt)

        # Replace 3+ newlines with 2
        result = re.sub(r"\n{3,}", "\n\n", result)

        # Remove trailing whitespace on lines
        result = "\n".join(line.rstrip() for line in result.split("\n"))

        return result.strip()

    def _add_specificity_hints(self, prompt: str, context: Dict) -> str:
        """Add hints for more specific requests."""
        specificity_prompt = (
            "\n\nPlease be specific about:\n"
            "- The desired outcome\n"
            "- Any constraints or requirements\n"
            "- The format of the expected output"
        )

        return prompt + specificity_prompt
