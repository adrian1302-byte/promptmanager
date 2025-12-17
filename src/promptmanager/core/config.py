"""Configuration management using Pydantic settings."""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, List
from functools import lru_cache


class ProviderSettings(BaseSettings):
    """LLM Provider configuration."""

    openai_api_key: Optional[str] = Field(None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, alias="ANTHROPIC_API_KEY")
    default_provider: str = Field("openai", alias="PM_DEFAULT_PROVIDER")
    default_model: str = Field("gpt-4", alias="PM_DEFAULT_MODEL")
    timeout: float = Field(30.0, alias="PM_PROVIDER_TIMEOUT")
    max_retries: int = Field(3, alias="PM_MAX_RETRIES")

    model_config = {"env_prefix": "", "extra": "ignore"}


class CompressionSettings(BaseSettings):
    """Compression module configuration."""

    default_strategy: str = Field("hybrid", alias="PM_COMPRESSION_STRATEGY")
    default_ratio: float = Field(0.5, alias="PM_COMPRESSION_RATIO")
    quality_threshold: float = Field(0.85, alias="PM_QUALITY_THRESHOLD")
    default_tokenizer: str = Field("gpt-4", alias="PM_DEFAULT_TOKENIZER")

    # Model-based compression settings
    small_model_name: str = Field("gpt2", alias="PM_SMALL_MODEL")
    device: str = Field("cpu", alias="PM_DEVICE")

    model_config = {"env_prefix": "", "extra": "ignore"}


class EnhancementSettings(BaseSettings):
    """Enhancement module configuration."""

    default_level: str = Field("moderate", alias="PM_ENHANCEMENT_LEVEL")
    use_llm_enhancement: bool = Field(True, alias="PM_USE_LLM_ENHANCEMENT")
    quality_threshold: float = Field(0.8, alias="PM_ENHANCEMENT_QUALITY_THRESHOLD")
    max_length_increase: float = Field(1.5, alias="PM_MAX_LENGTH_INCREASE")

    model_config = {"env_prefix": "", "extra": "ignore"}


class GenerationSettings(BaseSettings):
    """Generation module configuration."""

    default_style: str = Field("auto", alias="PM_GENERATION_STYLE")
    template_dir: Optional[str] = Field(None, alias="PM_TEMPLATE_DIR")
    optimize_by_default: bool = Field(True, alias="PM_OPTIMIZE_PROMPTS")

    model_config = {"env_prefix": "", "extra": "ignore"}


class ControlSettings(BaseSettings):
    """Control module configuration."""

    storage_backend: str = Field("file", alias="PM_STORAGE_BACKEND")
    storage_path: str = Field("./prompts", alias="PM_STORAGE_PATH")
    enable_versioning: bool = Field(True, alias="PM_ENABLE_VERSIONING")

    model_config = {"env_prefix": "", "extra": "ignore"}


class APISettings(BaseSettings):
    """API server configuration."""

    host: str = Field("0.0.0.0", alias="PM_API_HOST")
    port: int = Field(8000, alias="PM_API_PORT")
    debug: bool = Field(False, alias="PM_DEBUG")
    cors_origins: List[str] = Field(["*"], alias="PM_CORS_ORIGINS")
    rate_limit: int = Field(100, alias="PM_RATE_LIMIT")

    model_config = {"env_prefix": "", "extra": "ignore"}


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    level: str = Field("INFO", alias="PM_LOG_LEVEL")
    format: str = Field("json", alias="PM_LOG_FORMAT")

    model_config = {"env_prefix": "", "extra": "ignore"}


class Settings(BaseSettings):
    """Root configuration aggregating all settings."""

    provider: ProviderSettings = Field(default_factory=ProviderSettings)
    compression: CompressionSettings = Field(default_factory=CompressionSettings)
    enhancement: EnhancementSettings = Field(default_factory=EnhancementSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    control: ControlSettings = Field(default_factory=ControlSettings)
    api: APISettings = Field(default_factory=APISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    model_config = {
        "env_file": ".env",
        "env_nested_delimiter": "__",
        "extra": "ignore"
    }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def reload_settings() -> Settings:
    """Reload settings (clears cache)."""
    get_settings.cache_clear()
    return get_settings()
