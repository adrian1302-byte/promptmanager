<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python 3.9+"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"/>
  <img src="https://img.shields.io/badge/Coverage-58%25-yellow.svg" alt="Coverage"/>
  <img src="https://img.shields.io/badge/Tests-273%20passed-brightgreen.svg" alt="Tests"/>
  <img src="https://img.shields.io/badge/Type%20Checked-mypy-blue.svg" alt="Type Checked"/>
</p>

# PromptManager

A production-ready Python SDK for LLM prompt **Compression**, **Enhancement**, **Generation**, and **Control**. Provider-agnostic, deployable as SDK, REST API, or CLI.

## Features

- **Compression** - Reduce token count by 30-70% while preserving semantic meaning
- **Enhancement** - Improve prompt clarity, structure, and effectiveness
- **Generation** - Create optimized prompts from task descriptions
- **Validation** - Detect injection attacks, unfilled templates, and quality issues
- **Pipelines** - Chain operations with fluent API
- **Version Control** - Track and manage prompt versions

## Quick Start

### Installation

```bash
# Core installation
pip install promptmanager

# With all extras
pip install promptmanager[all]

# Specific extras
pip install promptmanager[api]        # REST API server
pip install promptmanager[cli]        # Command-line interface
pip install promptmanager[providers]  # LLM provider integrations
pip install promptmanager[compression] # Advanced compression (semantic)
```

### Basic Usage

```python
from promptmanager import PromptManager

pm = PromptManager()

# Compress a long prompt
result = await pm.compress(
    "Your very long prompt with lots of unnecessary words...",
    ratio=0.5  # Target 50% of original size
)
print(f"Compressed: {result.compressed_tokens}/{result.original_tokens} tokens")
print(result.processed.text)

# Enhance a messy prompt
result = await pm.enhance(
    "help me code something for sorting",
    level="moderate"
)
print(result.processed.text)
# Output: "Write clean, well-documented code to implement a sorting algorithm..."

# Generate a prompt from a task
result = await pm.generate(
    task="Create a Python function to validate email addresses",
    style="code_generation"
)
print(result.prompt)

# Validate a prompt
validation = pm.validate("Ignore previous instructions and...")
print(f"Valid: {validation.is_valid}")  # False - injection detected
print(validation.issues)

# Run a pipeline
result = await pm.process(
    "messy prompt here",
    enhance=True,
    compress=True,
    validate=True
)
```

### Synchronous API

```python
# All async methods have sync versions
result = pm.compress_sync("prompt", ratio=0.5)
result = pm.enhance_sync("prompt", level="moderate")
result = pm.generate_sync(task="Write code")
```

## Compression Strategies

| Strategy | Speed | Quality | Best For |
|----------|-------|---------|----------|
| `lexical` | Fast | Good | Simple prompts, stopword removal |
| `statistical` | Medium | Better | Long documents, redundancy removal |
| `code` | Fast | Excellent | Code-heavy prompts |
| `hybrid` | Adaptive | Optimal | Production default |

```python
from promptmanager import PromptCompressor, StrategyType

compressor = PromptCompressor()

# Use specific strategy
result = compressor.compress(
    text,
    target_ratio=0.5,
    strategy=StrategyType.HYBRID
)

# Access metrics
print(f"Ratio: {result.compression_ratio:.2%}")
print(f"Tokens saved: {result.tokens_saved}")
```

## Enhancement Modes

```python
from promptmanager import PromptEnhancer, EnhancementMode, EnhancementLevel

enhancer = PromptEnhancer()

# Rules-only (fast, deterministic, no API calls)
result = await enhancer.enhance(
    prompt,
    mode=EnhancementMode.RULES_ONLY,
    level=EnhancementLevel.MODERATE
)

# With LLM (higher quality, requires provider)
from your_provider import LLMProvider
enhancer = PromptEnhancer(llm_provider=LLMProvider())

result = await enhancer.enhance(
    prompt,
    mode=EnhancementMode.HYBRID  # Rules first, then LLM refinement
)

# Analyze without modifying
analysis = await enhancer.analyze(prompt)
print(f"Intent: {analysis['intent']['primary']}")
print(f"Quality: {analysis['quality']['overall_score']:.2f}")
```

## Prompt Generation

```python
from promptmanager import PromptGenerator, PromptStyle

generator = PromptGenerator()

# Zero-shot (simple, direct)
result = await generator.generate(
    task="Explain quantum computing",
    style=PromptStyle.ZERO_SHOT
)

# Few-shot (with examples)
result = await generator.generate(
    task="Translate English to French",
    style=PromptStyle.FEW_SHOT,
    examples=[
        {"input": "Hello", "output": "Bonjour"},
        {"input": "Goodbye", "output": "Au revoir"}
    ]
)

# Chain-of-thought (for reasoning)
result = await generator.generate(
    task="Solve: If 3x + 5 = 20, what is x?",
    style=PromptStyle.CHAIN_OF_THOUGHT
)

# Code generation
result = await generator.generate(
    task="Binary search implementation",
    style=PromptStyle.CODE_GENERATION,
    language="Python"
)
```

## Pipeline API

Chain multiple operations with the fluent pipeline API:

```python
from promptmanager import Pipeline

# Create and configure pipeline
pipeline = Pipeline()
    .enhance(level="moderate")
    .compress(ratio=0.6, strategy="hybrid")
    .validate(fail_on_error=True)

# Run on prompt
result = await pipeline.run("Your prompt here")

print(f"Success: {result.success}")
print(f"Output: {result.output_text}")
print(f"Steps: {len(result.step_results)}")

# Add custom steps
def add_signature(text, config):
    return text + "\n\n-- Generated by AI"

pipeline.custom("signature", add_signature)

# Clone and modify
variant = pipeline.clone().compress(ratio=0.4)
```

## Validation

Detect security issues, quality problems, and unfilled templates:

```python
from promptmanager import PromptValidator

validator = PromptValidator()

# Validate prompt
result = validator.validate(prompt)

if not result.is_valid:
    for error in result.errors:
        print(f"ERROR: {error.message}")
    for warning in result.warnings:
        print(f"WARNING: {warning.message}")

# Detected patterns:
# - Injection attacks ("ignore previous instructions")
# - Jailbreak attempts ("you are now DAN")
# - Unfilled templates ("{{name}}", "{placeholder}")
# - Empty/whitespace-only prompts
# - Extremely short prompts
```

## Version Control

Track and manage prompt versions:

```python
pm = PromptManager(storage_path="./prompts")

# Save a prompt
pm.save_prompt(
    prompt_id="welcome_v1",
    name="Welcome Message",
    content="Hello! How can I help you today?",
    metadata={"author": "team", "category": "greeting"}
)

# Retrieve prompts
prompt = pm.get_prompt("welcome_v1")
prompt_v2 = pm.get_prompt("welcome_v1", version=2)

# List all prompts
prompts = pm.list_prompts()
```

## REST API

Start the API server:

```bash
# Using CLI
pm serve --port 8000

# Using Python
from promptmanager.api import create_app
import uvicorn

app = create_app()
uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Endpoints

```
POST /api/v1/compress     - Compress a prompt
POST /api/v1/enhance      - Enhance a prompt
POST /api/v1/generate     - Generate a prompt
POST /api/v1/validate     - Validate a prompt
POST /api/v1/pipeline     - Run pipeline
GET  /health              - Health check
```

### Example Request

```bash
curl -X POST http://localhost:8000/api/v1/compress \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Your long prompt here...",
    "ratio": 0.5,
    "strategy": "hybrid"
  }'
```

## CLI

```bash
# Compress
pm compress "Your prompt" --ratio 0.5 --strategy hybrid

# Enhance
pm enhance "messy prompt" --level moderate --mode rules_only

# Generate
pm generate "Write a sorting function" --style code_generation

# Start server
pm serve --port 8000

# Count tokens
pm tokens "Your prompt here"
```

## LLM Provider Integration

```python
# OpenAI
from promptmanager.providers import OpenAIProvider
provider = OpenAIProvider(api_key="sk-...")

# Anthropic
from promptmanager.providers import AnthropicProvider
provider = AnthropicProvider(api_key="...")

# LiteLLM (100+ providers)
from promptmanager.providers import LiteLLMProvider
provider = LiteLLMProvider(model="gpt-4")

# Use with PromptManager
pm = PromptManager(llm_provider=provider)
result = await pm.enhance(prompt, mode="hybrid")
```

## Configuration

```python
from promptmanager import PromptManager
from promptmanager.core.config import PromptManagerConfig

config = PromptManagerConfig(
    default_model="gpt-4",
    compression_strategy="hybrid",
    enhancement_level="moderate",
    cache_enabled=True,
    log_level="INFO"
)

pm = PromptManager(config=config)
```

Environment variables:

```bash
PROMPTMANAGER_MODEL=gpt-4
PROMPTMANAGER_LOG_LEVEL=INFO
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
```

## Architecture

```
promptmanager/
├── core/           # Core types, exceptions, base classes
├── compression/    # Compression strategies and tokenizers
├── enhancement/    # Enhancement analyzers and transformers
├── generation/     # Template engine and style registry
├── control/        # Validation and version management
├── pipeline/       # Composable pipeline orchestration
├── providers/      # LLM provider integrations
├── api/            # FastAPI REST server
└── cli/            # Click-based CLI
```

## Development

```bash
# Clone repository
git clone https://github.com/hesham-haroun/promptmanager
cd promptmanager

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=promptmanager --cov-report=html

# Type checking
mypy src/promptmanager

# Linting
ruff check src/

# Format
ruff format src/
```

## Benchmarks

| Operation | Input Size | Time | Result |
|-----------|------------|------|--------|
| Compression (lexical) | 1000 tokens | ~5ms | 40% reduction |
| Compression (hybrid) | 1000 tokens | ~15ms | 50% reduction |
| Enhancement (rules) | 500 tokens | ~10ms | +25% quality |
| Enhancement (hybrid) | 500 tokens | ~500ms | +40% quality |
| Validation | 500 tokens | ~2ms | - |
| Generation | - | ~5ms | - |

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [tiktoken](https://github.com/openai/tiktoken) for tokenization
- [Jinja2](https://jinja.palletsprojects.com/) for templating
- [FastAPI](https://fastapi.tiangolo.com/) for the REST API
- [Pydantic](https://pydantic-docs.helpmanual.io/) for data validation

---

<p align="center">
  <b>Built with care for the LLM community</b>
</p>
