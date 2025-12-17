"""Lexical compression strategy using rule-based transformations."""

import re
from typing import Optional, Set, Dict, List
from .base import CompressionStrategy, CompressionConfig


class LexicalCompressor(CompressionStrategy):
    """
    Fast lexical compression using rule-based transformations.

    Techniques:
    - Stopword removal (context-aware)
    - Whitespace normalization
    - Redundant punctuation removal
    - Common phrase abbreviation
    - Filler word removal

    This is the fastest strategy with no external dependencies.
    """

    name = "lexical"
    description = "Fast rule-based compression using lexical transformations"
    supports_streaming = True
    requires_external_model = False

    # Stopwords that are safe to remove in most contexts
    SAFE_STOPWORDS: Set[str] = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after",
        "above", "below", "between", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all",
        "each", "few", "more", "most", "other", "some", "such", "only",
        "own", "same", "so", "than", "too", "very", "just", "also"
    }

    # Words to NEVER remove (important for instructions)
    PROTECTED_WORDS: Set[str] = {
        "not", "don't", "doesn't", "didn't", "won't", "wouldn't", "couldn't",
        "shouldn't", "cannot", "never", "no", "none", "neither", "nor",
        "always", "must", "required", "important", "critical", "essential",
        "error", "warning", "caution", "note", "if", "else", "then",
        "return", "output", "input", "result", "response", "answer",
        "true", "false", "null", "none", "undefined"
    }

    # Common phrase abbreviations
    PHRASE_ABBREVIATIONS: Dict[str, str] = {
        "for example": "e.g.",
        "that is": "i.e.",
        "in other words": "i.e.",
        "and so on": "etc.",
        "as soon as possible": "ASAP",
        "with respect to": "w.r.t.",
        "in order to": "to",
        "due to the fact that": "because",
        "in the event that": "if",
        "at this point in time": "now",
        "in spite of the fact that": "although",
        "make sure that": "ensure",
        "a large number of": "many",
        "a small number of": "few",
        "the majority of": "most",
        "prior to": "before",
        "subsequent to": "after",
        "in addition to": "besides",
        "with regard to": "regarding",
        "in accordance with": "per",
        "on the other hand": "alternatively",
        "as a result of": "because of",
        "in the case of": "for",
        "by means of": "via",
        "in terms of": "regarding",
    }

    # Filler phrases to remove
    FILLER_PHRASES: List[str] = [
        "basically", "essentially", "actually", "literally", "honestly",
        "frankly", "obviously", "clearly", "certainly", "definitely",
        "absolutely", "simply", "really", "very much", "kind of",
        "sort of", "you know", "i mean", "like i said", "as i mentioned",
        "it is worth noting that", "it should be noted that",
        "it is important to note that", "needless to say",
        "it goes without saying", "as a matter of fact",
        "to be honest", "in my opinion", "i think that",
        "it seems that", "it appears that",
    ]

    def __init__(self):
        # Pre-compile regex patterns for efficiency
        self._filler_patterns = [
            re.compile(r'\b' + re.escape(f) + r'\b', re.IGNORECASE)
            for f in self.FILLER_PHRASES
        ]
        self._phrase_patterns = {
            re.compile(r'\b' + re.escape(p) + r'\b', re.IGNORECASE): abbrev
            for p, abbrev in self.PHRASE_ABBREVIATIONS.items()
        }

    def compress(
        self,
        text: str,
        config: CompressionConfig,
        content_type: Optional[str] = None
    ) -> str:
        """Apply lexical compression techniques."""
        # Handle preserved sections
        prefix, middle, suffix = self._split_preserved_sections(text, config)

        if not middle:
            return text

        result = middle

        # 1. Normalize whitespace
        result = self._normalize_whitespace(result)

        # 2. Apply phrase abbreviations
        result = self._abbreviate_phrases(result)

        # 3. Remove filler phrases
        result = self._remove_fillers(result)

        # 4. Remove stopwords (if aggressive or needed for ratio)
        current_ratio = len(result) / len(middle) if middle else 1.0
        if config.aggressive_mode or current_ratio > config.target_ratio:
            result = self._remove_stopwords(result, config)

        # 5. Compress punctuation
        result = self._compress_punctuation(result)

        # 6. Final whitespace cleanup
        result = self._normalize_whitespace(result)

        # Reconstruct with preserved sections
        parts = [p for p in [prefix, result, suffix] if p]
        return " ".join(parts)

    def estimate_compression_ratio(self, text: str) -> float:
        """Estimate achievable compression ratio."""
        words = text.lower().split()
        if not words:
            return 1.0

        # Count removable stopwords
        removable = sum(
            1 for w in words
            if w in self.SAFE_STOPWORDS
            and w not in self.PROTECTED_WORDS
        )

        # Estimate ~50% of stopwords can be safely removed
        stopword_reduction = (removable / len(words)) * 0.5

        # Estimate 5-10% from phrase abbreviations
        phrase_reduction = 0.07

        # Estimate 3-5% from whitespace/punctuation
        whitespace_reduction = 0.04

        total_reduction = stopword_reduction + phrase_reduction + whitespace_reduction
        return max(0.5, 1 - total_reduction)  # Cap at 50% reduction

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize all whitespace to single spaces."""
        # Replace multiple spaces/tabs with single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Normalize line breaks (keep paragraph structure)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove trailing whitespace on lines
        text = re.sub(r' +\n', '\n', text)
        return text.strip()

    def _abbreviate_phrases(self, text: str) -> str:
        """Replace common phrases with abbreviations."""
        result = text
        for pattern, abbrev in self._phrase_patterns.items():
            result = pattern.sub(abbrev, result)
        return result

    def _remove_fillers(self, text: str) -> str:
        """Remove filler words and phrases."""
        result = text
        for pattern in self._filler_patterns:
            # Remove filler with surrounding punctuation cleanup
            result = pattern.sub('', result)

        # Clean up double spaces from removals
        result = re.sub(r' +', ' ', result)
        return result

    def _remove_stopwords(
        self,
        text: str,
        config: CompressionConfig
    ) -> str:
        """Remove stopwords while preserving meaning."""
        lines = text.split('\n')
        processed_lines = []

        for line in lines:
            # Check if line should be preserved
            if self._should_preserve(line, config.preserve_patterns):
                processed_lines.append(line)
                continue

            # Process words in the line
            words = line.split()
            filtered_words = []

            for i, word in enumerate(words):
                clean_word = re.sub(r'[^\w]', '', word).lower()

                # Keep protected words
                if clean_word in self.PROTECTED_WORDS:
                    filtered_words.append(word)
                    continue

                # Keep first word of sentence
                if i == 0 or (i > 0 and words[i-1].endswith('.')):
                    filtered_words.append(word)
                    continue

                # Skip safe stopwords
                if clean_word in self.SAFE_STOPWORDS:
                    continue

                filtered_words.append(word)

            processed_lines.append(' '.join(filtered_words))

        return '\n'.join(processed_lines)

    def _compress_punctuation(self, text: str) -> str:
        """Remove redundant punctuation."""
        # Keep ellipsis but remove excess
        text = re.sub(r'\.{4,}', '...', text)
        # Remove multiple punctuation
        text = re.sub(r'!{2,}', '!', text)
        text = re.sub(r'\?{2,}', '?', text)
        text = re.sub(r',{2,}', ',', text)
        # Remove space before punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text
