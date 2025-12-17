"""Tiktoken-based token counter for OpenAI models."""

from typing import List, Optional
from .base import TokenCounter


class TiktokenCounter(TokenCounter):
    """
    Token counter using OpenAI's tiktoken library.

    Supports all OpenAI models and provides accurate token counts.
    """

    # Model to encoding mapping
    MODEL_ENCODINGS = {
        # GPT-4 models
        "gpt-4": "cl100k_base",
        "gpt-4-0314": "cl100k_base",
        "gpt-4-0613": "cl100k_base",
        "gpt-4-32k": "cl100k_base",
        "gpt-4-32k-0314": "cl100k_base",
        "gpt-4-32k-0613": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4-turbo-preview": "cl100k_base",
        "gpt-4-1106-preview": "cl100k_base",
        "gpt-4-vision-preview": "cl100k_base",
        # GPT-4o models
        "gpt-4o": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        "gpt-4o-2024-05-13": "o200k_base",
        # GPT-3.5 models
        "gpt-3.5-turbo": "cl100k_base",
        "gpt-3.5-turbo-0301": "cl100k_base",
        "gpt-3.5-turbo-0613": "cl100k_base",
        "gpt-3.5-turbo-16k": "cl100k_base",
        "gpt-3.5-turbo-16k-0613": "cl100k_base",
        "gpt-3.5-turbo-instruct": "cl100k_base",
        # Embeddings
        "text-embedding-ada-002": "cl100k_base",
        "text-embedding-3-small": "cl100k_base",
        "text-embedding-3-large": "cl100k_base",
        # Legacy models
        "text-davinci-003": "p50k_base",
        "text-davinci-002": "p50k_base",
        "davinci": "r50k_base",
        "curie": "r50k_base",
        "babbage": "r50k_base",
        "ada": "r50k_base",
    }

    def __init__(self, model: str = "gpt-4"):
        """
        Initialize TiktokenCounter for a specific model.

        Args:
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
        """
        try:
            import tiktoken
            self._tiktoken = tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is required for TiktokenCounter. "
                "Install with: pip install tiktoken"
            )

        self._model = model
        self.name = f"tiktoken-{model}"

        # Get encoding for model
        if model in self.MODEL_ENCODINGS:
            self._encoding = tiktoken.get_encoding(self.MODEL_ENCODINGS[model])
        else:
            try:
                self._encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback to cl100k_base (GPT-4 encoding)
                self._encoding = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return len(self._encoding.encode(text))

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self._encoding.encode(text)

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        return self._encoding.decode(tokens)

    def count_messages(self, messages: List[dict]) -> int:
        """
        Count tokens for chat messages (includes overhead).

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Total token count including message overhead
        """
        # Token overhead per message varies by model
        # This is an approximation for GPT-4 / GPT-3.5
        tokens_per_message = 4  # <im_start>{role}\n{content}<im_end>\n
        tokens_per_name = -1  # if name is present, role is omitted

        total = 0
        for message in messages:
            total += tokens_per_message
            for key, value in message.items():
                total += self.count(str(value))
                if key == "name":
                    total += tokens_per_name

        total += 3  # every reply is primed with <|im_start|>assistant

        return total

    @property
    def encoding_name(self) -> str:
        """Get the encoding name."""
        return self._encoding.name

    @classmethod
    def get_encoding_for_model(cls, model: str) -> str:
        """Get the encoding name for a model."""
        return cls.MODEL_ENCODINGS.get(model, "cl100k_base")

    @classmethod
    def list_supported_models(cls) -> List[str]:
        """List all supported models."""
        return list(cls.MODEL_ENCODINGS.keys())
