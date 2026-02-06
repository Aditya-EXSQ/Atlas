"""
vLLM provider adapter for Atlas.

Communicates with a vLLM server through its OpenAI-compatible API.
Supports text generation, streaming, and embeddings (when available).
"""

from typing import AsyncIterator

from atlas.core.base import BaseProvider
from atlas.core.exceptions import (
    ConnectionError,
    GenerationError,
    NotSupportedError,
)
from atlas.core.models import (
    EmbeddingRequest,
    EmbeddingResponse,
    GenerateRequest,
    GenerateResponse,
    StreamChunk,
)
from atlas.core.registry import ProviderRegistry

_PROVIDER_NAME = "vllm"

try:
    from openai import AsyncOpenAI

    _OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OPENAI_AVAILABLE = False


@ProviderRegistry.register(_PROVIDER_NAME)
class VLLMProvider(BaseProvider):
    """
    vLLM inference provider (OpenAI-compatible API).

    Requires a running vLLM server (``vllm serve <model>``).

    Args:
        model: Model name as served by vLLM.
        base_url: OpenAI-compatible API base URL.
        api_key: API key (use ``"EMPTY"`` for local servers).
        timeout: Request timeout in seconds.
        **kwargs: Extra generation defaults (temperature, top_p, …).
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        timeout: float = 120.0,
        **kwargs,
    ):
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "The 'openai' package is required for the vLLM provider. "
                "Install it with: pip install openai"
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.extra = kwargs
        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        try:
            resp = await self._client.chat.completions.create(
                model=request.model or self.model,
                messages=[{"role": "user", "content": request.prompt}],
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=request.stop,
                stream=False,
            )
        except Exception as exc:
            if "connect" in str(exc).lower():
                raise ConnectionError(
                    _PROVIDER_NAME, self.base_url, str(exc)
                ) from exc
            raise GenerationError(_PROVIDER_NAME, str(exc)) from exc

        choice = resp.choices[0]
        usage = None
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }

        return GenerateResponse(
            text=choice.message.content or "",
            model=resp.model,
            provider=_PROVIDER_NAME,
            usage=usage,
            finish_reason=choice.finish_reason,
        )

    # ------------------------------------------------------------------
    # stream
    # ------------------------------------------------------------------
    async def stream(self, request: GenerateRequest) -> AsyncIterator[StreamChunk]:
        try:
            resp = await self._client.chat.completions.create(
                model=request.model or self.model,
                messages=[{"role": "user", "content": request.prompt}],
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=request.stop,
                stream=True,
            )
        except Exception as exc:
            if "connect" in str(exc).lower():
                raise ConnectionError(
                    _PROVIDER_NAME, self.base_url, str(exc)
                ) from exc
            raise GenerationError(_PROVIDER_NAME, str(exc)) from exc

        async for chunk in resp:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield StreamChunk(
                        text=delta.content,
                        model=chunk.model,
                        provider=_PROVIDER_NAME,
                        finish_reason=chunk.choices[0].finish_reason,
                    )

    # ------------------------------------------------------------------
    # embeddings
    # ------------------------------------------------------------------
    async def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        raise NotSupportedError(
            _PROVIDER_NAME,
            "embeddings (vLLM servers typically do not expose an embeddings endpoint; "
            "use a dedicated embedding provider instead)",
        )
