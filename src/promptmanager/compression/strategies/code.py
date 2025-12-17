"""Code-specific compression strategy."""

import re
from typing import Optional, Dict, List
from .base import CompressionStrategy, CompressionConfig


class CodeCompressor(CompressionStrategy):
    """
    Specialized compressor for code content.

    Techniques:
    - Comment removal/summarization
    - Whitespace normalization (preserve indentation)
    - Docstring compression
    - Empty line reduction
    - Preserve essential code structure
    """

    name = "code"
    description = "Specialized compression for code content"
    supports_streaming = True
    requires_external_model = False

    # Language detection patterns
    LANGUAGE_PATTERNS: Dict[str, List[str]] = {
        "python": [
            r'\bdef\s+\w+\s*\(', r'\bclass\s+\w+[:\(]', r'\bimport\s+\w+',
            r'\bfrom\s+\w+\s+import', r'if\s+__name__\s*==',
            r'@\w+', r'\blambda\s+\w+:'
        ],
        "javascript": [
            r'\bfunction\s+\w+', r'\bconst\s+\w+\s*=', r'\blet\s+\w+\s*=',
            r'=>', r'\basync\s+function', r'\brequire\s*\(',
            r'\bmodule\.exports', r'\bexport\s+(default|const|function)'
        ],
        "typescript": [
            r'\binterface\s+\w+', r'\btype\s+\w+\s*=', r':\s*\w+\[\]',
            r'<[A-Z]\w*>', r'\bas\s+\w+'
        ],
        "java": [
            r'\bpublic\s+class', r'\bprivate\s+\w+', r'@Override',
            r'\bvoid\s+\w+\s*\(', r'\bthrows\s+\w+'
        ],
        "go": [
            r'\bfunc\s+\w+', r'\bpackage\s+\w+', r'\btype\s+\w+\s+struct',
            r':=', r'\bgo\s+\w+', r'\bdefer\s+'
        ],
        "rust": [
            r'\bfn\s+\w+', r'\blet\s+mut', r'\bimpl\s+\w+',
            r'\bpub\s+fn', r'->\s*\w+', r'\bmatch\s+\w+'
        ],
    }

    # Important comment markers to preserve
    IMPORTANT_MARKERS = ['todo', 'fixme', 'hack', 'note:', 'bug:', 'warning:', 'important']

    def compress(
        self,
        text: str,
        config: CompressionConfig,
        content_type: Optional[str] = None
    ) -> str:
        """Apply code-specific compression."""
        # Detect language
        language = self._detect_language(text)

        result = text

        # 1. Remove/compress comments
        result = self._compress_comments(result, language, config)

        # 2. Compress docstrings (Python)
        if language == "python":
            result = self._compress_docstrings(result, config)

        # 3. Normalize code whitespace
        result = self._normalize_code_whitespace(result)

        # 4. Reduce empty lines
        result = self._reduce_empty_lines(result)

        # 5. Compress multi-line strings if aggressive
        if config.aggressive_mode:
            result = self._compress_multiline_strings(result)

        return result

    def estimate_compression_ratio(self, text: str) -> float:
        """Estimate achievable compression ratio for code."""
        lines = text.split('\n')
        total_lines = len(lines)

        if total_lines == 0:
            return 1.0

        comment_lines = 0
        empty_lines = 0
        docstring_lines = 0
        in_docstring = False

        for line in lines:
            stripped = line.strip()

            if not stripped:
                empty_lines += 1
                continue

            # Count single-line comments
            if stripped.startswith('#') or stripped.startswith('//'):
                comment_lines += 1
                continue

            # Track docstrings
            if '"""' in stripped or "'''" in stripped:
                in_docstring = not in_docstring
                docstring_lines += 1
            elif in_docstring:
                docstring_lines += 1

        # Estimate reduction
        removable = (
            comment_lines * 0.8 +
            empty_lines * 0.5 +
            docstring_lines * 0.3
        )

        reduction = removable / total_lines if total_lines > 0 else 0

        return max(0.5, 1 - reduction)

    def _detect_language(self, text: str) -> str:
        """Detect programming language from code patterns."""
        scores = {}

        for language, patterns in self.LANGUAGE_PATTERNS.items():
            matches = sum(
                1 for pattern in patterns
                if re.search(pattern, text, re.MULTILINE)
            )
            scores[language] = matches

        if scores:
            best = max(scores, key=scores.get)
            if scores[best] >= 2:
                return best

        return "unknown"

    def _compress_comments(
        self,
        text: str,
        language: str,
        config: CompressionConfig
    ) -> str:
        """Remove or compress comments based on language."""
        lines = text.split('\n')
        result_lines = []

        # Determine comment pattern
        if language in ["python"]:
            single_comment = r'^\s*#'
            inline_comment = r'#.*$'
        elif language in ["javascript", "typescript", "java", "go", "rust"]:
            single_comment = r'^\s*//'
            inline_comment = r'//.*$'
        else:
            single_comment = r'^\s*(#|//)'
            inline_comment = r'(#|//).*$'

        in_multiline = False
        multiline_start = r'/\*'
        multiline_end = r'\*/'

        for line in lines:
            stripped = line.strip()

            # Handle multi-line comments
            if re.search(multiline_start, stripped) and not re.search(multiline_end, stripped):
                in_multiline = True
                if not config.aggressive_mode:
                    result_lines.append(line)
                continue

            if in_multiline:
                if re.search(multiline_end, stripped):
                    in_multiline = False
                if not config.aggressive_mode:
                    result_lines.append(line)
                continue

            # Check if line is only a comment
            if re.match(single_comment, line):
                # Keep important comments
                if any(marker in stripped.lower() for marker in self.IMPORTANT_MARKERS):
                    result_lines.append(line)
                elif not config.aggressive_mode:
                    # Keep but maybe shorten
                    result_lines.append(line.strip())
                # else: remove in aggressive mode
            else:
                # Handle inline comments
                if config.aggressive_mode:
                    line = re.sub(inline_comment, '', line).rstrip()
                result_lines.append(line)

        return '\n'.join(result_lines)

    def _compress_docstrings(
        self,
        text: str,
        config: CompressionConfig
    ) -> str:
        """Compress Python docstrings."""

        def compress_match(match):
            quote = match.group(1)
            content = match.group(2)

            if config.aggressive_mode:
                # Extract first sentence only
                lines = content.strip().split('\n')
                first_line = lines[0].strip() if lines else ""
                # Truncate long first lines
                if len(first_line) > 80:
                    first_line = first_line[:77] + "..."
                return f'{quote}{first_line}{quote}'

            return match.group(0)

        # Match triple-quoted strings
        pattern = r'(\"\"\"|\'\'\')([\s\S]*?)\1'
        return re.sub(pattern, compress_match, text)

    def _normalize_code_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving code structure."""
        lines = text.split('\n')
        result_lines = []

        for line in lines:
            # Remove trailing whitespace
            line = line.rstrip()

            # Don't modify indentation (important for Python, etc.)
            result_lines.append(line)

        return '\n'.join(result_lines)

    def _reduce_empty_lines(self, text: str) -> str:
        """Reduce multiple consecutive empty lines."""
        # Replace 3+ empty lines with 2
        return re.sub(r'\n{3,}', '\n\n', text)

    def _compress_multiline_strings(self, text: str) -> str:
        """Compress multi-line strings to single line (aggressive only)."""
        # This is risky and may break code, use only in aggressive mode
        lines = text.split('\n')
        result = []
        in_string = False
        string_buffer = []
        quote_char = None

        for line in lines:
            # Detect triple quotes
            for quote in ['"""', "'''"]:
                count = line.count(quote)
                if count % 2 == 1:  # Odd number = toggle
                    if in_string and quote == quote_char:
                        string_buffer.append(line)
                        combined = ' '.join(l.strip() for l in string_buffer)
                        result.append(combined)
                        string_buffer = []
                        in_string = False
                        quote_char = None
                        break
                    elif not in_string:
                        in_string = True
                        quote_char = quote
                        string_buffer.append(line)
                        break
            else:
                if in_string:
                    string_buffer.append(line)
                else:
                    result.append(line)

        # Handle unclosed strings
        if string_buffer:
            result.extend(string_buffer)

        return '\n'.join(result)
