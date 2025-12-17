# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-XX

### Added

- **Compression Module**
  - Lexical compression strategy (stopwords, filler phrases)
  - Statistical compression strategy (TF-IDF scoring)
  - Code compression strategy (comments, whitespace, docstrings)
  - Hybrid compression strategy (adaptive, content-aware)
  - Token counting with tiktoken
  - Configurable compression ratios (0.1-1.0)

- **Enhancement Module**
  - Rule-based enhancement engine
  - LLM-based enhancement (optional)
  - Intent detection (code, Q&A, summarization, etc.)
  - Quality scoring (clarity, structure, completeness)
  - Hybrid mode (rules + LLM)
  - Multiple enhancement levels (minimal, light, moderate, aggressive)

- **Generation Module**
  - Jinja2-based template engine
  - Built-in templates (zero-shot, few-shot, chain-of-thought, code)
  - Style recommendation based on task
  - Intelligent slot filling
  - Custom template support

- **Control Module**
  - Prompt validation engine
  - Injection attack detection
  - Template variable detection
  - Quality checks
  - Version control for prompts
  - In-memory and file-based storage

- **Pipeline Module**
  - Fluent API for chaining operations
  - Custom step support
  - Error handling modes (stop, skip, continue)
  - Factory functions for common pipelines

- **API Module**
  - FastAPI REST server
  - Endpoints for compress, enhance, generate, validate
  - Health check endpoint
  - Request/response schemas

- **CLI Module**
  - Command-line interface
  - Commands: compress, enhance, generate, serve, tokens

- **Providers**
  - OpenAI provider
  - Anthropic provider
  - LiteLLM adapter (100+ providers)

- **Core**
  - Type-safe dataclasses (Prompt, Message, Results)
  - Custom exceptions hierarchy
  - Configuration management with Pydantic

### Technical

- Python 3.9+ support
- Async-first design with sync wrappers
- 273 tests with 58% coverage
- Type hints throughout
- PEP 561 compliant (py.typed)

## [Unreleased]

### Planned

- Semantic compression strategy (embeddings-based)
- Streaming compression support
- Caching layer for repeated operations
- Prometheus metrics
- OpenTelemetry tracing
- Web UI dashboard
