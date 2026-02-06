"""
Hugging Face provider adapter for Atlas.

Supports both the hosted Hugging Face Inference API and local
``transformers`` pipeline inference.
"""

from typing import AsyncIterator

import httpx

from atlas.core.base import BaseProvider
from atlas.core.exceptions import (
    ConnectionError,
    EmbeddingError,
    GenerationError,
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

_PROVIDER_NAME = "huggingface"
_HF_API_BASE = "https://api-inference.huggingface.co"


@ProviderRegistry.register(_PROVIDER_NAME)
class HuggingFaceProvider(BaseProvider):
    """
    Hugging Face Inference API provider.

    Uses the hosted HF Inference API (``api-inference.huggingface.co``).
    Requires a valid Hugging Face API token for gated / private models.

    Args:
        model: Model ID on Hugging Face Hub (e.g. ``"mistralai/Mistral-7B-Instruct-v0.2"``).
        api_token: HF API token.  Can also be set via ``HF_TOKEN`` env var.
        base_url: Override for the Inference API base URL.
        timeout: Request timeout in seconds.
        **kwargs: Extra parameters forwarded to the API.
    """

    def __init__(
        self,
        model: str,
        api_token: str = "",
        base_url: str = _HF_API_BASE,
        timeout: float = 120.0,
        **kwargs,
    ):
        import os

        self.model = model
        self.api_token = api_token or os.environ.get("HF_TOKEN", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.extra = kwargs
        self._headers = {}
        if self.api_token:
            self._headers["Authorization"] = f"Bearer {self.api_token}"

    def _model_url(self, model: str | None = None) -> str:
        return f"{self.base_url}/models/{model or self.model}"

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        payload = {
            "inputs": request.prompt,
            "parameters": {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_new_tokens": request.max_tokens,
                "return_full_text": False,
            },
        }
        if request.stop:
            payload["parameters"]["stop_sequences"] = request.stop

        url = self._model_url(request.model)
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    url, json=payload, headers=self._headers
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.ConnectError as exc:
            raise ConnectionError(_PROVIDER_NAME, url, str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            raise GenerationError(_PROVIDER_NAME, str(exc)) from exc

        # HF Inference API returns a list of generated texts
        if isinstance(data, list) and data:
            text = data[0].get("generated_text", "")
        elif isinstance(data, dict):
            text = data.get("generated_text", "")
        else:
            text = str(data)

        return GenerateResponse(
            text=text,
            model=request.model or self.model,
            provider=_PROVIDER_NAME,
        )

    # ------------------------------------------------------------------
    # stream
    # ------------------------------------------------------------------
    async def stream(self, request: GenerateRequest) -> AsyncIterator[StreamChunk]:
        payload = {
            "inputs": request.prompt,
            "parameters": {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_new_tokens": request.max_tokens,
                "return_full_text": False,
            },
            "stream": True,
        }
        if request.stop:
            payload["parameters"]["stop_sequences"] = request.stop

        url = self._model_url(request.model)
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST", url, json=payload, headers=self._headers
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        import json

                        try:
                            data = json.loads(line[len("data:"):].strip())
                        except json.JSONDecodeError:
                            continue
                        token = data.get("token", {}).get("text", "")
                        if token:
                            yield StreamChunk(
                                text=token,
                                model=request.model or self.model,
                                provider=_PROVIDER_NAME,
                            )
        except httpx.ConnectError as exc:
            raise ConnectionError(_PROVIDER_NAME, url, str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            raise StreamingError(_PROVIDER_NAME, str(exc)) from exc

    # ------------------------------------------------------------------
    # embeddings
    # ------------------------------------------------------------------
    async def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        texts = (
            request.input if isinstance(request.input, list) else [request.input]
        )
        payload = {"inputs": texts}

        url = self._model_url(request.model)
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    url, json=payload, headers=self._headers
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.ConnectError as exc:
            raise ConnectionError(_PROVIDER_NAME, url, str(exc)) from exc
        except (httpx.HTTPStatusError, KeyError) as exc:
            raise EmbeddingError(_PROVIDER_NAME, str(exc)) from exc

        # HF returns list of embeddings directly
        if isinstance(data, list):
            embeddings_list = data
        else:
            raise EmbeddingError(
                _PROVIDER_NAME, f"Unexpected response format: {type(data)}"
            )

        return EmbeddingResponse(
            embeddings=embeddings_list,
            model=request.model or self.model,
            provider=_PROVIDER_NAME,
        )
