"""
Unified exceptions for the Atlas LLM inference framework.

All provider-specific errors are wrapped into these unified exceptions
so that user-facing code never needs to handle provider-specific errors.
"""


class AtlasError(Exception):
    """Base exception for all Atlas errors."""


class ProviderNotFoundError(AtlasError):
    """Raised when a requested provider is not registered."""

    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(
            f"Provider '{provider}' not found. "
            f"Make sure it is installed and registered."
        )


class ProviderError(AtlasError):
    """Raised when a provider encounters an error during inference."""

    def __init__(self, provider: str, message: str):
        self.provider = provider
        super().__init__(f"[{provider}] {message}")


class ConnectionError(ProviderError):
    """Raised when a provider cannot connect to its backend."""

    def __init__(self, provider: str, url: str, detail: str = ""):
        self.url = url
        msg = f"Could not connect to {url}"
        if detail:
            msg += f": {detail}"
        super().__init__(provider, msg)


class GenerationError(ProviderError):
    """Raised when text generation fails."""


class StreamingError(ProviderError):
    """Raised when streaming generation fails."""


class EmbeddingError(ProviderError):
    """Raised when embedding generation fails."""


class NotSupportedError(ProviderError):
    """Raised when a provider does not support a requested operation."""

    def __init__(self, provider: str, operation: str):
        self.operation = operation
        super().__init__(provider, f"Operation '{operation}' is not supported")


class ConfigurationError(AtlasError):
    """Raised when configuration is invalid or missing."""
