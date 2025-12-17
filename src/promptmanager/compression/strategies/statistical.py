"""Statistical compression using information-theoretic measures."""

import re
import math
from typing import Optional, Dict, List, Tuple
from collections import Counter
from .base import CompressionStrategy, CompressionConfig


class StatisticalCompressor(CompressionStrategy):
    """
    Statistical compression using information-theoretic measures.

    Techniques:
    - TF-IDF based token importance scoring
    - Self-information scoring
    - Sentence importance ranking
    - Redundancy detection and removal
    """

    name = "statistical"
    description = "Statistical compression using TF-IDF and information theory"
    supports_streaming = False
    requires_external_model = False

    # Pre-computed IDF values for common words (approximate)
    COMMON_WORD_IDF: Dict[str, float] = {
        "the": 0.1, "is": 0.2, "a": 0.15, "to": 0.18, "and": 0.12,
        "of": 0.11, "in": 0.19, "that": 0.25, "it": 0.22, "for": 0.21,
        "you": 0.3, "i": 0.28, "this": 0.35, "be": 0.2, "on": 0.25,
        "with": 0.3, "as": 0.28, "at": 0.32, "by": 0.35, "from": 0.33,
    }

    def compress(
        self,
        text: str,
        config: CompressionConfig,
        content_type: Optional[str] = None
    ) -> str:
        """Apply statistical compression."""
        # Handle preserved sections
        prefix, middle, suffix = self._split_preserved_sections(text, config)

        if not middle:
            return text

        # Split into sentences
        sentences = self._split_sentences(middle)

        if len(sentences) <= 2:
            # Too short for sentence-level compression
            result = self._compress_tokens(middle, config)
        else:
            # Score and select sentences
            result = self._compress_sentences(sentences, config)

        # Reconstruct with preserved sections
        parts = [p for p in [prefix, result, suffix] if p]
        return " ".join(parts)

    def estimate_compression_ratio(self, text: str) -> float:
        """Estimate achievable compression ratio."""
        sentences = self._split_sentences(text)

        if len(sentences) <= 2:
            return 0.8  # Limited compression for short texts

        # Estimate based on redundancy
        redundancy = self._estimate_redundancy(sentences)

        # Higher redundancy = more compression potential
        return max(0.4, 0.9 - redundancy * 0.5)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Handle common abbreviations
        text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr)\.\s', r'\1_DOT_ ', text)
        text = re.sub(r'\b(e\.g|i\.e|etc|vs)\.\s', r'\1_DOT_ ', text)

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Restore dots in abbreviations
        sentences = [s.replace('_DOT_', '.') for s in sentences]

        return [s.strip() for s in sentences if s.strip()]

    def _compress_sentences(
        self,
        sentences: List[str],
        config: CompressionConfig
    ) -> str:
        """Compress by selecting important sentences."""
        # Score sentences
        scored = self._score_sentences(sentences)

        # Calculate target sentence count
        total_chars = sum(len(s) for s in sentences)
        target_chars = int(total_chars * config.target_ratio)

        # Select sentences greedily by importance
        selected_indices = set()
        current_chars = 0

        for idx, score in scored:
            if current_chars >= target_chars:
                break
            selected_indices.add(idx)
            current_chars += len(sentences[idx])

        # Ensure at least first sentence is included
        if 0 not in selected_indices and sentences:
            selected_indices.add(0)

        # Return sentences in original order
        result = [
            sentences[i] for i in range(len(sentences))
            if i in selected_indices
        ]

        return ' '.join(result)

    def _score_sentences(
        self,
        sentences: List[str]
    ) -> List[Tuple[int, float]]:
        """Score sentences by importance using TF-IDF."""
        # Build document-level statistics
        all_words = []
        sentence_words = []

        for sentence in sentences:
            words = self._tokenize(sentence)
            sentence_words.append(words)
            all_words.extend(words)

        # Document term frequency
        doc_tf = Counter(all_words)
        total_words = len(all_words)

        # Score each sentence
        scored = []
        for idx, words in enumerate(sentence_words):
            if not words:
                scored.append((idx, 0.0))
                continue

            score = 0.0
            for word in words:
                # TF component
                tf = doc_tf[word] / total_words if total_words > 0 else 0

                # IDF component
                idf = self.COMMON_WORD_IDF.get(word, 2.0)

                # Self-information approximation
                self_info = -math.log(tf + 0.001) if tf > 0 else 5.0

                score += tf * idf * self_info

            # Normalize by sentence length
            score /= len(words)

            # Position bias: boost first and last sentences
            if idx == 0:
                score *= 1.5
            elif idx == len(sentences) - 1:
                score *= 1.2
            elif idx < 3:
                score *= 1.1

            scored.append((idx, score))

        # Sort by score descending
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def _compress_tokens(
        self,
        text: str,
        config: CompressionConfig
    ) -> str:
        """Compress at token level for short texts."""
        words = text.split()

        if len(words) <= 5:
            return text

        # Score each word
        word_freq = Counter(w.lower() for w in words)
        word_scores = []

        for i, word in enumerate(words):
            clean_word = word.lower()
            freq = word_freq[clean_word]

            # IDF-like scoring: rare words are more important
            idf = self.COMMON_WORD_IDF.get(clean_word, 2.0)

            # Position weight: words at beginning/end matter more
            position_weight = 1.0
            if i < 3 or i >= len(words) - 3:
                position_weight = 1.5

            score = idf * position_weight / (freq + 1)
            word_scores.append((i, word, score))

        # Select top words by score
        target_count = max(3, int(len(words) * config.target_ratio))

        sorted_words = sorted(word_scores, key=lambda x: x[2], reverse=True)
        selected_indices = set(idx for idx, _, _ in sorted_words[:target_count])

        # Always include first and last few words
        for i in range(min(2, len(words))):
            selected_indices.add(i)
        for i in range(max(0, len(words) - 2), len(words)):
            selected_indices.add(i)

        # Reconstruct in original order
        result = [words[i] for i in sorted(selected_indices)]

        return ' '.join(result)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r'\b\w+\b', text.lower())

    def _estimate_redundancy(self, sentences: List[str]) -> float:
        """Estimate text redundancy using n-gram overlap."""
        if len(sentences) <= 1:
            return 0.0

        all_ngrams = []
        for sentence in sentences:
            words = self._tokenize(sentence)
            # Generate 2-grams and 3-grams
            for n in [2, 3]:
                for i in range(len(words) - n + 1):
                    ngram = tuple(words[i:i+n])
                    all_ngrams.append(ngram)

        if not all_ngrams:
            return 0.0

        ngram_counts = Counter(all_ngrams)
        repeated = sum(1 for count in ngram_counts.values() if count > 1)

        return repeated / len(ngram_counts) if ngram_counts else 0.0
