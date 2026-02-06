"""
Dynamic provider registry for the Atlas LLM inference framework.

Providers register themselves via the ``@ProviderRegistry.register``
decorator or by calling ``ProviderRegistry.add()`` at runtime.
"""

from typing import Dict, List, Type

from atlas.core.base import BaseProvider
from atlas.core.exceptions import ProviderNotFoundError


class ProviderRegistry:
    """
    Central registry that maps provider names to provider classes.

    Usage::

        @ProviderRegistry.register("ollama")
        class OllamaProvider(BaseProvider):
            ...

        # Or register at runtime:
        ProviderRegistry.add("custom", CustomProvider)

        # Resolve:
        cls = ProviderRegistry.get("ollama")
    """

    _providers: Dict[str, Type[BaseProvider]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a provider class under *name*."""

        def decorator(provider_class: Type[BaseProvider]):
            cls._providers[name] = provider_class
            return provider_class

        return decorator

    @classmethod
    def add(cls, name: str, provider_class: Type[BaseProvider]) -> None:
        """Register a provider class under *name* at runtime."""
        cls._providers[name] = provider_class

    @classmethod
    def get(cls, name: str) -> Type[BaseProvider]:
        """Return the provider class for *name* or raise ``ProviderNotFoundError``."""
        if name not in cls._providers:
            raise ProviderNotFoundError(name)
        return cls._providers[name]

    @classmethod
    def list_providers(cls) -> List[str]:
        """Return all registered provider names."""
        return list(cls._providers.keys())

    @classmethod
    def clear(cls) -> None:
        """Remove all registered providers (useful for testing)."""
        cls._providers.clear()
