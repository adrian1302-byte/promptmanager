"""HuggingFace tokenizer-based token counter."""

from typing import List, Optional
from .base import TokenCounter


class HuggingFaceCounter(TokenCounter):
    """
    Token counter using HuggingFace transformers tokenizers.

    Supports any model available on HuggingFace Hub.
    """

    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize HuggingFaceCounter for a specific model.

        Args:
            model_name: HuggingFace model name (e.g., "gpt2", "meta-llama/Llama-2-7b")
        """
        try:
            from transformers import AutoTokenizer
            self._AutoTokenizer = AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for HuggingFaceCounter. "
                "Install with: pip install transformers"
            )

        self._model_name = model_name
        self.name = f"hf-{model_name.split('/')[-1]}"

        # Load tokenizer
        self._tokenizer = self._AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

    def count(self, text: str) -> int:
        """Count tokens using HuggingFace tokenizer."""
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        return self._tokenizer.decode(tokens)

    def count_with_special_tokens(self, text: str) -> int:
        """Count tokens including special tokens (BOS, EOS, etc.)."""
        return len(self._tokenizer.encode(text, add_special_tokens=True))

    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self._tokenizer.vocab_size

    def tokenize(self, text: str) -> List[str]:
        """Get string tokens (not IDs)."""
        return self._tokenizer.tokenize(text)

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @classmethod
    def list_common_models(cls) -> List[str]:
        """List commonly used models."""
        return [
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl",
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-13b-hf",
            "meta-llama/Llama-2-70b-hf",
            "mistralai/Mistral-7B-v0.1",
            "mistralai/Mixtral-8x7B-v0.1",
            "google/gemma-2b",
            "google/gemma-7b",
            "Qwen/Qwen-7B",
            "bigscience/bloom-560m",
            "EleutherAI/gpt-neo-125m",
            "EleutherAI/gpt-neo-1.3B",
            "EleutherAI/gpt-neo-2.7B",
            "EleutherAI/gpt-j-6B",
        ]
