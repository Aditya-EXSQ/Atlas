"""
Ollama provider adapter for Atlas.

Connects to a local or remote Ollama server via its REST API.
Supports text generation, streaming, and embeddings.
"""

import json
from typing import AsyncIterator

import httpx

from atlas.core.base import BaseProvider
from atlas.core.exceptions import ConnectionError, EmbeddingError, GenerationError
from atlas.core.models import (
    EmbeddingRequest,
    EmbeddingResponse,
    GenerateRequest,
    GenerateResponse,
    StreamChunk,
)
from atlas.core.registry import ProviderRegistry

_PROVIDER_NAME = "ollama"


@ProviderRegistry.register(_PROVIDER_NAME)
class OllamaProvider(BaseProvider):
    """
    Ollama inference provider.

    Requires a running Ollama server (``ollama serve``).

    Args:
        model: Model name (e.g. ``"llama3"``, ``"mistral:7b-instruct"``).
        base_url: Ollama server URL.  Defaults to ``http://localhost:11434``.
        timeout: Request timeout in seconds.
        **kwargs: Extra options forwarded to the Ollama API.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        **kwargs,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.extra = kwargs

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        payload = {
            "model": request.model or self.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "num_predict": request.max_tokens,
            },
        }
        if request.stop:
            payload["options"]["stop"] = request.stop

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/api/chat", json=payload
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.ConnectError as exc:
            raise ConnectionError(_PROVIDER_NAME, self.base_url, str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            raise GenerationError(_PROVIDER_NAME, str(exc)) from exc

        return GenerateResponse(
            text=data["message"]["content"],
            model=request.model or self.model,
            provider=_PROVIDER_NAME,
            finish_reason=data.get("done_reason"),
        )

    # ------------------------------------------------------------------
    # stream
    # ------------------------------------------------------------------
    async def stream(self, request: GenerateRequest) -> AsyncIterator[StreamChunk]:
        payload = {
            "model": request.model or self.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "stream": True,
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "num_predict": request.max_tokens,
            },
        }
        if request.stop:
            payload["options"]["stop"] = request.stop

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/api/chat", json=payload
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield StreamChunk(
                                text=content,
                                model=request.model or self.model,
                                provider=_PROVIDER_NAME,
                                finish_reason=(
                                    data.get("done_reason")
                                    if data.get("done")
                                    else None
                                ),
                            )
                        if data.get("done"):
                            break
        except httpx.ConnectError as exc:
            raise ConnectionError(_PROVIDER_NAME, self.base_url, str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            raise GenerationError(_PROVIDER_NAME, str(exc)) from exc

    # ------------------------------------------------------------------
    # embeddings
    # ------------------------------------------------------------------
    async def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        texts = (
            request.input if isinstance(request.input, list) else [request.input]
        )
        all_embeddings = []

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                for text in texts:
                    resp = await client.post(
                        f"{self.base_url}/api/embeddings",
                        json={
                            "model": request.model or self.model,
                            "prompt": text,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    all_embeddings.append(data["embedding"])
        except httpx.ConnectError as exc:
            raise ConnectionError(_PROVIDER_NAME, self.base_url, str(exc)) from exc
        except (httpx.HTTPStatusError, KeyError) as exc:
            raise EmbeddingError(_PROVIDER_NAME, str(exc)) from exc

        return EmbeddingResponse(
            embeddings=all_embeddings,
            model=request.model or self.model,
            provider=_PROVIDER_NAME,
        )
