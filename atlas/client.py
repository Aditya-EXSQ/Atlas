"""
LLMClient – the public SDK entry-point for the Atlas framework.

Example::

    from atlas import LLMClient

    client = LLMClient(provider="ollama", model="llama3")
    response = await client.generate("Hello!")
    print(response.text)

    async for chunk in client.stream("Explain transformers"):
        print(chunk.text, end="")
"""

from typing import AsyncIterator, List, Optional, Union

# Ensure built-in providers are registered on first import.
import atlas.providers  # noqa: F401
from atlas.config import ProviderConfig
from atlas.core.base import BaseProvider
from atlas.core.models import (
    EmbeddingRequest,
    EmbeddingResponse,
    GenerateRequest,
    GenerateResponse,
    StreamChunk,
)
from atlas.core.registry import ProviderRegistry


class LLMClient:
    """
    High-level, provider-agnostic LLM client.

    Args:
        provider: Registered provider name (e.g. ``"ollama"``, ``"vllm"``,
            ``"huggingface"``).
        model: Model identifier understood by the chosen provider.
        base_url: Optional base URL override for the provider backend.
        api_key: Optional API key / token.
        timeout: Request timeout in seconds.
        config: An optional ``ProviderConfig`` object.  If given, other
            keyword arguments are ignored.
        **kwargs: Extra provider-specific options.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        config: ProviderConfig | None = None,
        **kwargs,
    ):
        if config is not None:
            provider = config.provider
            model = config.model
            base_url = config.base_url or base_url
            api_key = config.api_key or api_key
            timeout = config.timeout
            kwargs = {**config.extra, **kwargs}

        if not provider or not model:
            raise ValueError("Both 'provider' and 'model' are required.")

        provider_cls = ProviderRegistry.get(provider)

        init_kwargs = {**kwargs}
        if base_url is not None:
            init_kwargs["base_url"] = base_url
        if api_key is not None:
            init_kwargs["api_key"] = api_key
        init_kwargs["timeout"] = timeout

        self._provider: BaseProvider = provider_cls(model=model, **init_kwargs)
        self._provider_name = provider
        self._model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
    ) -> GenerateResponse:
        """Generate a complete text response."""
        request = GenerateRequest(
            prompt=prompt,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
        )
        return await self._provider.generate(request)

    async def stream(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream text tokens as they are generated."""
        request = GenerateRequest(
            prompt=prompt,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
        )
        async for chunk in self._provider.stream(request):
            yield chunk

    async def embeddings(
        self,
        input: Union[str, List[str]],
        *,
        model: Optional[str] = None,
    ) -> EmbeddingResponse:
        """Generate vector embeddings for the given input."""
        request = EmbeddingRequest(input=input, model=model)
        return await self._provider.embeddings(request)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def list_providers() -> List[str]:
        """Return all registered provider names."""
        return ProviderRegistry.list_providers()

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model

    def __repr__(self) -> str:
        return (
            f"LLMClient(provider={self._provider_name!r}, "
            f"model={self._model!r})"
        )
