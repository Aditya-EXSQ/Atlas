"""
Atlas – Provider-agnostic LLM inference framework.
===================================================

Quick start::

    import asyncio
    from atlas import LLMClient

    async def main():
        client = LLMClient(provider="ollama", model="llama3")
        response = await client.generate("Hello!")
        print(response.text)

    asyncio.run(main())

See ``atlas.client.LLMClient`` for the full API.
"""

from atlas.client import LLMClient
from atlas.config import ProviderConfig
from atlas.core.base import BaseProvider
from atlas.core.exceptions import (
    AtlasError,
    ConfigurationError,
    ConnectionError,
    EmbeddingError,
    GenerationError,
    NotSupportedError,
    ProviderError,
    ProviderNotFoundError,
    StreamingError,
)
from atlas.core.models import (
    EmbeddingRequest,
    EmbeddingResponse,
    GenerateRequest,
    GenerateResponse,
    StreamChunk,
)
from atlas.core.registry import ProviderRegistry

__all__ = [
    # Client
    "LLMClient",
    # Config
    "ProviderConfig",
    # Base
    "BaseProvider",
    "ProviderRegistry",
    # Data contracts
    "GenerateRequest",
    "GenerateResponse",
    "StreamChunk",
    "EmbeddingRequest",
    "EmbeddingResponse",
    # Exceptions
    "AtlasError",
    "ProviderNotFoundError",
    "ProviderError",
    "ConnectionError",
    "GenerationError",
    "StreamingError",
    "EmbeddingError",
    "NotSupportedError",
    "ConfigurationError",
]

__version__ = "0.1.0"
