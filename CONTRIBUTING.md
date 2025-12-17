# Contributing to PromptManager

Thank you for your interest in contributing to PromptManager! This document provides guidelines and information for contributors.

## Code of Conduct

Please be respectful and constructive in all interactions. We're building something together.

## Getting Started

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/promptmanager.git
   cd promptmanager
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # .venv\Scripts\activate   # Windows
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run tests to verify setup**
   ```bash
   pytest
   ```

### Development Workflow

1. **Create a branch for your feature/fix**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes**

3. **Run tests and checks**
   ```bash
   # Run tests
   pytest

   # Run with coverage
   pytest --cov=promptmanager

   # Type checking
   mypy src/promptmanager

   # Linting
   ruff check src/

   # Format code
   ruff format src/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```

   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation only
   - `style:` - Formatting, no code change
   - `refactor:` - Code change that neither fixes a bug nor adds a feature
   - `test:` - Adding tests
   - `chore:` - Maintenance tasks

5. **Push and create a Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Project Structure

```
promptmanager/
├── src/promptmanager/
│   ├── __init__.py         # Main PromptManager class
│   ├── core/               # Core types, exceptions, base classes
│   │   ├── types.py        # Prompt, Message, Result dataclasses
│   │   ├── exceptions.py   # Custom exceptions
│   │   └── base.py         # Abstract base classes
│   ├── compression/        # Compression module
│   │   ├── compressor.py   # Main PromptCompressor
│   │   ├── strategies/     # Compression strategies
│   │   └── tokenizers/     # Token counting
│   ├── enhancement/        # Enhancement module
│   │   ├── enhancer.py     # Main PromptEnhancer
│   │   ├── analyzers/      # Intent detection, quality scoring
│   │   └── transformers/   # Rule engine, LLM enhancer
│   ├── generation/         # Generation module
│   │   ├── generator.py    # Main PromptGenerator
│   │   ├── templates/      # Jinja2 templates
│   │   └── styles/         # Style registry
│   ├── control/            # Control module
│   │   ├── manager.py      # Version management
│   │   └── validation.py   # Prompt validation
│   ├── pipeline/           # Pipeline orchestration
│   ├── providers/          # LLM provider integrations
│   ├── api/                # FastAPI REST server
│   └── cli/                # CLI commands
├── tests/
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── conftest.py         # Pytest fixtures
└── pyproject.toml          # Project configuration
```

## Guidelines

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all function signatures
- Write docstrings for public functions and classes
- Keep functions focused and under 50 lines when possible
- Use meaningful variable and function names

### Testing

- Write tests for all new functionality
- Maintain or improve code coverage
- Use pytest fixtures for common setup
- Test edge cases and error conditions

```python
# Example test structure
class TestMyFeature:
    """Tests for MyFeature."""

    @pytest.fixture
    def feature(self):
        """Create feature instance."""
        return MyFeature()

    def test_basic_functionality(self, feature):
        """Test basic use case."""
        result = feature.do_something("input")
        assert result is not None

    def test_edge_case(self, feature):
        """Test edge case handling."""
        result = feature.do_something("")
        assert result == expected_value
```

### Documentation

- Update README.md if adding new features
- Add docstrings to new public APIs
- Include usage examples in docstrings

```python
def compress(self, text: str, ratio: float = 0.5) -> CompressionResult:
    """
    Compress text to reduce token count.

    Args:
        text: The text to compress
        ratio: Target compression ratio (0.1-1.0)

    Returns:
        CompressionResult with compressed text and metrics

    Example:
        >>> compressor = PromptCompressor()
        >>> result = compressor.compress("long text...", ratio=0.5)
        >>> print(result.compression_ratio)
    """
```

### Adding New Features

#### New Compression Strategy

1. Create `src/promptmanager/compression/strategies/your_strategy.py`
2. Inherit from `CompressionStrategy`
3. Implement required methods
4. Register in `strategies/__init__.py`
5. Add tests in `tests/unit/test_compression.py`

#### New Enhancement Rule

1. Add rule to `enhancement/rules/builtin.yaml` or
2. Create rule programmatically in `transformers/rule_engine.py`
3. Add tests for the rule

#### New LLM Provider

1. Create `src/promptmanager/providers/your_provider.py`
2. Inherit from `LLMProvider`
3. Implement required methods
4. Add to `providers/__init__.py`
5. Document in README

## Pull Request Process

1. **Ensure all tests pass**
2. **Update documentation** if needed
3. **Add entry to CHANGELOG** (if applicable)
4. **Request review** from maintainers
5. **Address feedback** promptly
6. **Squash commits** if requested

### PR Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Type hints added
- [ ] Linting passes (`ruff check`)
- [ ] Tests pass (`pytest`)
- [ ] Type checking passes (`mypy`)

## Reporting Issues

### Bug Reports

Include:
- Python version
- PromptManager version
- Minimal reproduction code
- Expected vs actual behavior
- Full error traceback

### Feature Requests

Include:
- Use case description
- Proposed API (if applicable)
- Alternatives considered

## Questions?

- Open a [Discussion](https://github.com/hesham-haroun/promptmanager/discussions)
- Check existing [Issues](https://github.com/hesham-haroun/promptmanager/issues)

## Recognition

Contributors will be recognized in:
- GitHub contributors page
- Release notes (for significant contributions)

Thank you for contributing!
