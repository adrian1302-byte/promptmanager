"""Intelligent slot filling for prompt templates."""

import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum


class SlotType(Enum):
    """Types of template slots."""
    TEXT = "text"           # Free-form text
    LIST = "list"           # List of items
    CODE = "code"           # Code snippet
    EXAMPLES = "examples"   # Input/output examples
    JSON = "json"           # JSON structure
    ENUM = "enum"           # Choice from options


@dataclass
class SlotDefinition:
    """Definition of a template slot."""
    name: str
    slot_type: SlotType
    description: str
    required: bool = True
    default: Any = None
    options: Optional[List[str]] = None  # For ENUM type
    validation_pattern: Optional[str] = None


@dataclass
class FilledSlot:
    """Result of slot filling."""
    name: str
    value: Any
    source: str  # "provided", "extracted", "default", "inferred"
    confidence: float = 1.0


@dataclass
class SlotFillingResult:
    """Result of the slot filling process."""
    filled_slots: Dict[str, Any]
    missing_slots: List[str]
    warnings: List[str] = field(default_factory=list)
    slot_details: List[FilledSlot] = field(default_factory=list)


class SlotFiller:
    """
    Intelligent slot filler for prompt templates.

    Extracts and infers values for template variables from
    task descriptions and context.
    """

    # Patterns for extracting common information
    EXTRACTION_PATTERNS = {
        "language": [
            r"\b(python|javascript|typescript|java|go|rust|c\+\+|ruby|php|swift|kotlin|scala|r|matlab|sql)\b",
        ],
        "format": [
            r"\b(json|xml|csv|yaml|markdown|html|text|table)\b",
            r"output.*?\b(json|xml|csv|yaml|markdown|html|text|table)\b",
            r"format.*?\b(json|xml|csv|yaml|markdown|html|text|table)\b",
        ],
        "length": [
            r"(\d+)\s*(?:words|sentences|paragraphs|lines)",
            r"(?:about|around|approximately)\s*(\d+)",
            r"(?:max|maximum|up to)\s*(\d+)",
        ],
        "tone": [
            r"\b(formal|informal|professional|casual|friendly|technical|simple)\b",
        ],
    }

    # Default values for common slots
    SLOT_DEFAULTS = {
        "output_format": "Provide a clear, well-organized response.",
        "constraints": [],
        "context": "",
        "examples": [],
        "language": "Python",
        "tone": "professional",
    }

    def __init__(self, llm_client=None):
        """
        Initialize the slot filler.

        Args:
            llm_client: Optional LLM client for advanced inference
        """
        self.llm_client = llm_client

    def fill_slots(
        self,
        template_slots: List[SlotDefinition],
        task_description: str,
        provided_values: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        use_defaults: bool = True
    ) -> SlotFillingResult:
        """
        Fill template slots from task description and provided values.

        Args:
            template_slots: List of slot definitions
            task_description: The task description to extract from
            provided_values: Explicitly provided values
            context: Additional context for inference
            use_defaults: Whether to use default values for missing slots

        Returns:
            SlotFillingResult with filled values
        """
        provided_values = provided_values or {}
        context = context or {}

        filled_slots: Dict[str, Any] = {}
        missing_slots: List[str] = []
        warnings: List[str] = []
        slot_details: List[FilledSlot] = []

        for slot in template_slots:
            # Priority 1: Explicitly provided value
            if slot.name in provided_values:
                value = provided_values[slot.name]
                filled_slots[slot.name] = value
                slot_details.append(FilledSlot(
                    name=slot.name,
                    value=value,
                    source="provided",
                    confidence=1.0
                ))
                continue

            # Priority 2: Extract from task description
            extracted = self._extract_value(slot, task_description)
            if extracted is not None:
                filled_slots[slot.name] = extracted[0]
                slot_details.append(FilledSlot(
                    name=slot.name,
                    value=extracted[0],
                    source="extracted",
                    confidence=extracted[1]
                ))
                continue

            # Priority 3: Infer from context
            inferred = self._infer_value(slot, context, task_description)
            if inferred is not None:
                filled_slots[slot.name] = inferred[0]
                slot_details.append(FilledSlot(
                    name=slot.name,
                    value=inferred[0],
                    source="inferred",
                    confidence=inferred[1]
                ))
                continue

            # Priority 4: Use default value
            if use_defaults and slot.default is not None:
                filled_slots[slot.name] = slot.default
                slot_details.append(FilledSlot(
                    name=slot.name,
                    value=slot.default,
                    source="default",
                    confidence=0.5
                ))
                continue

            # Priority 5: Use global defaults
            if use_defaults and slot.name in self.SLOT_DEFAULTS:
                filled_slots[slot.name] = self.SLOT_DEFAULTS[slot.name]
                slot_details.append(FilledSlot(
                    name=slot.name,
                    value=self.SLOT_DEFAULTS[slot.name],
                    source="default",
                    confidence=0.3
                ))
                continue

            # Slot is missing
            if slot.required:
                missing_slots.append(slot.name)
            else:
                # Optional slot with no value
                filled_slots[slot.name] = None

        return SlotFillingResult(
            filled_slots=filled_slots,
            missing_slots=missing_slots,
            warnings=warnings,
            slot_details=slot_details
        )

    async def fill_slots_with_llm(
        self,
        template_slots: List[SlotDefinition],
        task_description: str,
        provided_values: Optional[Dict[str, Any]] = None
    ) -> SlotFillingResult:
        """
        Fill slots using LLM for complex inference.

        Args:
            template_slots: List of slot definitions
            task_description: The task description
            provided_values: Explicitly provided values

        Returns:
            SlotFillingResult with filled values
        """
        if not self.llm_client:
            return self.fill_slots(template_slots, task_description, provided_values)

        # First, do pattern-based filling
        result = self.fill_slots(
            template_slots,
            task_description,
            provided_values,
            use_defaults=False
        )

        # If we have missing required slots, use LLM
        if result.missing_slots and self.llm_client:
            llm_filled = await self._llm_infer_slots(
                result.missing_slots,
                template_slots,
                task_description
            )

            for slot_name, value in llm_filled.items():
                result.filled_slots[slot_name] = value
                result.slot_details.append(FilledSlot(
                    name=slot_name,
                    value=value,
                    source="inferred",
                    confidence=0.7
                ))

            # Update missing slots
            result.missing_slots = [
                s for s in result.missing_slots
                if s not in llm_filled
            ]

        return result

    def _extract_value(
        self,
        slot: SlotDefinition,
        text: str
    ) -> Optional[Tuple[Any, float]]:
        """Extract a slot value from text using patterns."""

        # Check if we have patterns for this slot
        patterns = self.EXTRACTION_PATTERNS.get(slot.name, [])

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1) if match.groups() else match.group(0)

                # Convert to appropriate type
                if slot.slot_type == SlotType.LIST:
                    value = [value]
                elif slot.slot_type == SlotType.ENUM:
                    if slot.options and value.lower() not in [o.lower() for o in slot.options]:
                        continue

                return (value, 0.8)

        # Special handling for task slot
        if slot.name == "task":
            # The task is usually the main description
            return (text, 0.9)

        return None

    def _infer_value(
        self,
        slot: SlotDefinition,
        context: Dict[str, Any],
        task_description: str
    ) -> Optional[Tuple[Any, float]]:
        """Infer a slot value from context."""

        # Check context for matching key
        if slot.name in context:
            return (context[slot.name], 0.85)

        # Special inference rules
        if slot.name == "language" and "code" in task_description.lower():
            # Default to Python for code tasks
            return ("Python", 0.6)

        if slot.name == "output_format":
            # Infer from task type
            if "json" in task_description.lower() or "api" in task_description.lower():
                return ("JSON", 0.7)
            if "list" in task_description.lower():
                return ("bullet list", 0.7)
            if "table" in task_description.lower():
                return ("table", 0.7)

        if slot.name == "constraints":
            # Extract constraint-like phrases
            constraints = []
            constraint_patterns = [
                r"must\s+(?:be|have|include)\s+([^.]+)",
                r"should\s+(?:be|have|include)\s+([^.]+)",
                r"(?:do not|don't|avoid)\s+([^.]+)",
            ]
            for pattern in constraint_patterns:
                matches = re.findall(pattern, task_description, re.IGNORECASE)
                constraints.extend(matches)

            if constraints:
                return (constraints, 0.7)

        return None

    async def _llm_infer_slots(
        self,
        slot_names: List[str],
        all_slots: List[SlotDefinition],
        task_description: str
    ) -> Dict[str, Any]:
        """Use LLM to infer missing slot values."""
        if not self.llm_client:
            return {}

        # Build prompt for LLM
        slot_descriptions = []
        for slot in all_slots:
            if slot.name in slot_names:
                desc = f"- {slot.name}: {slot.description}"
                if slot.slot_type == SlotType.ENUM and slot.options:
                    desc += f" (options: {', '.join(slot.options)})"
                slot_descriptions.append(desc)

        prompt = f"""Given this task description:
\"\"\"{task_description}\"\"\"

Extract or infer values for these template variables:
{chr(10).join(slot_descriptions)}

Respond with a JSON object mapping variable names to their values.
If you cannot determine a value, omit it from the response.
Only output the JSON, no explanation."""

        try:
            response = await self.llm_client.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )

            # Parse JSON response
            import json
            content = response.content.strip()
            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            return json.loads(content)
        except Exception:
            return {}

    def create_slot_definition(
        self,
        name: str,
        slot_type: SlotType = SlotType.TEXT,
        description: str = "",
        required: bool = True,
        default: Any = None,
        options: Optional[List[str]] = None
    ) -> SlotDefinition:
        """Create a slot definition."""
        return SlotDefinition(
            name=name,
            slot_type=slot_type,
            description=description,
            required=required,
            default=default,
            options=options
        )


# Predefined slot definitions for common templates
COMMON_SLOTS = {
    "task": SlotDefinition(
        name="task",
        slot_type=SlotType.TEXT,
        description="The main task or question to accomplish",
        required=True
    ),
    "context": SlotDefinition(
        name="context",
        slot_type=SlotType.TEXT,
        description="Background context or additional information",
        required=False,
        default=""
    ),
    "examples": SlotDefinition(
        name="examples",
        slot_type=SlotType.EXAMPLES,
        description="Input/output examples for few-shot learning",
        required=False,
        default=[]
    ),
    "constraints": SlotDefinition(
        name="constraints",
        slot_type=SlotType.LIST,
        description="Constraints or requirements to follow",
        required=False,
        default=[]
    ),
    "output_format": SlotDefinition(
        name="output_format",
        slot_type=SlotType.TEXT,
        description="Expected format of the output",
        required=False,
        default="Provide a clear, well-organized response."
    ),
    "language": SlotDefinition(
        name="language",
        slot_type=SlotType.TEXT,
        description="Programming language for code tasks",
        required=False,
        default="Python"
    ),
    "requirements": SlotDefinition(
        name="requirements",
        slot_type=SlotType.LIST,
        description="Specific requirements for the solution",
        required=False,
        default=[]
    ),
    "steps": SlotDefinition(
        name="steps",
        slot_type=SlotType.LIST,
        description="Steps to follow in reasoning",
        required=False,
        default=[]
    ),
}


def get_template_slots(template_name: str) -> List[SlotDefinition]:
    """Get slot definitions for a built-in template."""
    template_slots = {
        "zero_shot.j2": [
            COMMON_SLOTS["task"],
            COMMON_SLOTS["context"],
            COMMON_SLOTS["constraints"],
            COMMON_SLOTS["output_format"],
        ],
        "few_shot.j2": [
            COMMON_SLOTS["task"],
            COMMON_SLOTS["examples"],
            COMMON_SLOTS["context"],
            COMMON_SLOTS["constraints"],
            COMMON_SLOTS["output_format"],
        ],
        "chain_of_thought.j2": [
            COMMON_SLOTS["task"],
            COMMON_SLOTS["context"],
            COMMON_SLOTS["steps"],
            COMMON_SLOTS["constraints"],
            COMMON_SLOTS["output_format"],
        ],
        "code_generation.j2": [
            COMMON_SLOTS["task"],
            COMMON_SLOTS["language"],
            COMMON_SLOTS["requirements"],
            COMMON_SLOTS["context"],
            COMMON_SLOTS["examples"],
            COMMON_SLOTS["constraints"],
        ],
    }

    return template_slots.get(template_name, [COMMON_SLOTS["task"]])
