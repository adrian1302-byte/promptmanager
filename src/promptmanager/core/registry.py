"""Registry pattern for extensible component management."""

from typing import Dict, Type, TypeVar, Generic, Optional, Callable, List, Any
from abc import ABC

T = TypeVar('T')


class Registry(Generic[T], ABC):
    """
    Generic registry for managing pluggable components.

    Supports registration via decorators or explicit registration,
    with optional aliasing for convenience.
    """

    def __init__(self):
        self._items: Dict[str, Type[T]] = {}
        self._instances: Dict[str, T] = {}
        self._aliases: Dict[str, str] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        aliases: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator to register a class.

        Usage:
            @registry.register("my_item", aliases=["mi"])
            class MyItem:
                ...
        """
        def decorator(cls: Type[T]) -> Type[T]:
            self._items[name] = cls
            self._metadata[name] = metadata or {}

            for alias in (aliases or []):
                self._aliases[alias] = name

            return cls
        return decorator

    def register_class(
        self,
        name: str,
        cls: Type[T],
        aliases: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Explicitly register a class (non-decorator form)."""
        self._items[name] = cls
        self._metadata[name] = metadata or {}

        for alias in (aliases or []):
            self._aliases[alias] = name

    def get_class(self, name: str) -> Type[T]:
        """Get a registered class by name or alias."""
        resolved_name = self._aliases.get(name, name)

        if resolved_name not in self._items:
            available = list(self._items.keys())
            raise KeyError(
                f"'{name}' not found in registry. Available: {available}"
            )

        return self._items[resolved_name]

    def get_instance(self, name: str, *args, **kwargs) -> T:
        """
        Get or create a singleton instance.

        Creates the instance on first access, returns cached instance thereafter.
        """
        resolved_name = self._aliases.get(name, name)

        if resolved_name not in self._instances:
            cls = self.get_class(resolved_name)
            self._instances[resolved_name] = cls(*args, **kwargs)

        return self._instances[resolved_name]

    def create(self, name: str, *args, **kwargs) -> T:
        """Create a new instance (not cached)."""
        cls = self.get_class(name)
        return cls(*args, **kwargs)

    def list_registered(self) -> List[str]:
        """List all registered names."""
        return list(self._items.keys())

    def list_all(self) -> List[str]:
        """List all names including aliases."""
        return list(self._items.keys()) + list(self._aliases.keys())

    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a registered item."""
        resolved_name = self._aliases.get(name, name)
        return self._metadata.get(resolved_name, {})

    def is_registered(self, name: str) -> bool:
        """Check if a name is registered."""
        resolved_name = self._aliases.get(name, name)
        return resolved_name in self._items

    def clear_instances(self) -> None:
        """Clear all cached instances."""
        self._instances.clear()

    def unregister(self, name: str) -> None:
        """Unregister an item."""
        if name in self._items:
            del self._items[name]
        if name in self._instances:
            del self._instances[name]
        if name in self._metadata:
            del self._metadata[name]

        # Remove any aliases pointing to this name
        aliases_to_remove = [
            alias for alias, target in self._aliases.items()
            if target == name
        ]
        for alias in aliases_to_remove:
            del self._aliases[alias]


class ProviderRegistry(Registry):
    """
    Specialized registry for LLM providers.

    Includes provider-specific functionality like API key validation
    and model availability checking.
    """

    def get_provider(self, name: str, config: Any) -> Any:
        """
        Get a configured provider instance.

        Args:
            name: Provider name (e.g., "openai", "anthropic")
            config: Provider configuration

        Returns:
            Configured provider instance
        """
        cls = self.get_class(name)
        return cls(config)

    def get_available_providers(self) -> List[str]:
        """Get list of providers that have valid configuration."""
        from .config import get_settings
        settings = get_settings()

        available = []

        if settings.provider.openai_api_key:
            available.append("openai")
        if settings.provider.anthropic_api_key:
            available.append("anthropic")

        # LiteLLM is always available as a fallback
        if self.is_registered("litellm"):
            available.append("litellm")

        return available


class CompressionRegistry(Registry):
    """Specialized registry for compression strategies."""

    def get_strategy(self, name: str, **kwargs) -> Any:
        """Get a compression strategy instance."""
        return self.create(name, **kwargs)

    def list_strategies(self) -> List[Dict[str, Any]]:
        """List all strategies with their metadata."""
        strategies = []
        for name in self.list_registered():
            cls = self.get_class(name)
            strategies.append({
                "name": name,
                "description": getattr(cls, "description", ""),
                "requires_model": getattr(cls, "requires_external_model", False),
                **self.get_metadata(name)
            })
        return strategies


class EnhancementRegistry(Registry):
    """Specialized registry for enhancement transformers."""
    pass


class TemplateRegistry(Registry):
    """Specialized registry for prompt templates."""

    def get_template(self, name: str) -> Any:
        """Get a template by name."""
        return self.get_class(name)

    def list_by_style(self, style: str) -> List[str]:
        """List templates matching a style."""
        return [
            name for name in self.list_registered()
            if self.get_metadata(name).get("style") == style
        ]


# Global registry instances
provider_registry = ProviderRegistry()
compression_registry = CompressionRegistry()
enhancement_registry = EnhancementRegistry()
template_registry = TemplateRegistry()
