"""
Abstract base class for all Atlas LLM providers.

Every provider adapter must inherit from ``BaseProvider`` and implement
the three async methods: ``generate``, ``stream``, and ``embeddings``.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator

from atlas.core.models import (
    EmbeddingRequest,
    EmbeddingResponse,
    GenerateRequest,
    GenerateResponse,
    StreamChunk,
)


class BaseProvider(ABC):
    """
    Abstract base for all inference providers.

    Subclasses must implement:
      - ``generate()``  – full-text generation
      - ``stream()``    – token-level streaming generation
      - ``embeddings()`` – vector embeddings
    """

    @abstractmethod
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """
        Generate a complete text response.

        Args:
            request: Normalized generation request.

        Returns:
            Normalized generation response.
        """
        ...

    @abstractmethod
    async def stream(self, request: GenerateRequest) -> AsyncIterator[StreamChunk]:
        """
        Stream text tokens as they are generated.

        Args:
            request: Normalized generation request.

        Yields:
            StreamChunk for each generated token/piece.
        """
        ...  # pragma: no cover
        # Make this an async generator so subclasses can use `yield`
        yield  # type: ignore[misc]  # noqa: E501

    @abstractmethod
    async def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Generate vector embeddings for the given input.

        Args:
            request: Normalized embedding request.

        Returns:
            Normalized embedding response.
        """
        ...
