"""
Optional FastAPI HTTP server for Atlas.

Exposes a unified REST API and OpenAI-compatible endpoints that proxy
requests to the configured provider.  The SDK (``LLMClient``) works
independently of this server.

Run with::

    uvicorn atlas.server.app:create_app --factory --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import os
import time
from typing import AsyncIterator, List, Optional

from atlas.client import LLMClient

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FASTAPI_AVAILABLE = False

# -----------------------------------------------------------------------
# Pydantic request / response models (OpenAI-compatible where possible)
# -----------------------------------------------------------------------

if _FASTAPI_AVAILABLE:

    class GenerateRequestBody(BaseModel):
        prompt: str
        model: Optional[str] = None
        temperature: float = 0.7
        top_p: float = 0.9
        max_tokens: int = 512
        stop: Optional[List[str]] = None
        stream: bool = False

    class GenerateResponseBody(BaseModel):
        text: str
        model: str
        provider: str
        usage: Optional[dict] = None
        finish_reason: Optional[str] = None

    # -- OpenAI-compatible models ------------------------------------

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatCompletionRequest(BaseModel):
        model: Optional[str] = None
        messages: List[ChatMessage]
        temperature: float = 0.7
        top_p: float = 0.9
        max_tokens: int = 512
        stop: Optional[List[str]] = None
        stream: bool = False

    class ChatChoiceDelta(BaseModel):
        role: Optional[str] = None
        content: Optional[str] = None

    class ChatChoice(BaseModel):
        index: int = 0
        message: Optional[ChatMessage] = None
        delta: Optional[ChatChoiceDelta] = None
        finish_reason: Optional[str] = None

    class ChatCompletionResponse(BaseModel):
        id: str = ""
        object: str = "chat.completion"
        created: int = Field(default_factory=lambda: int(time.time()))
        model: str = ""
        choices: List[ChatChoice] = []
        usage: Optional[dict] = None

    class EmbeddingRequestBody(BaseModel):
        input: str | list[str]
        model: Optional[str] = None

    class EmbeddingData(BaseModel):
        object: str = "embedding"
        index: int = 0
        embedding: List[float]

    class EmbeddingResponseBody(BaseModel):
        object: str = "list"
        data: List[EmbeddingData]
        model: str = ""
        usage: Optional[dict] = None


# -----------------------------------------------------------------------
# Application factory
# -----------------------------------------------------------------------


def create_app(
    provider: str | None = None,
    model: str | None = None,
    **client_kwargs,
) -> "FastAPI":
    """
    Create and return a FastAPI application.

    Provider and model can be passed directly or read from
    ``ATLAS_PROVIDER`` / ``ATLAS_MODEL`` environment variables.
    """
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is required for the Atlas server. "
            "Install it with: pip install fastapi uvicorn"
        )

    provider = provider or os.environ.get("ATLAS_PROVIDER", "")
    model = model or os.environ.get("ATLAS_MODEL", "")

    if not provider or not model:
        raise ValueError(
            "Set ATLAS_PROVIDER and ATLAS_MODEL environment variables, "
            "or pass them to create_app()."
        )

    client = LLMClient(provider=provider, model=model, **client_kwargs)

    app = FastAPI(
        title="Atlas LLM Gateway",
        description="Provider-agnostic LLM inference API",
        version="0.1.0",
    )

    # ----- Atlas unified endpoints -----------------------------------

    @app.post("/v1/generate", response_model=GenerateResponseBody)
    async def generate(body: GenerateRequestBody):
        if body.stream:
            return StreamingResponse(
                _stream_generate(client, body), media_type="text/event-stream"
            )
        try:
            resp = await client.generate(
                prompt=body.prompt,
                model=body.model,
                temperature=body.temperature,
                top_p=body.top_p,
                max_tokens=body.max_tokens,
                stop=body.stop,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        return GenerateResponseBody(
            text=resp.text,
            model=resp.model,
            provider=resp.provider,
            usage=resp.usage,
            finish_reason=resp.finish_reason,
        )

    # ----- OpenAI-compatible endpoints --------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest):
        # Concatenate messages into a single prompt
        prompt = "\n".join(
            f"{m.role}: {m.content}" for m in body.messages
        )

        if body.stream:
            return StreamingResponse(
                _stream_chat(client, body, prompt),
                media_type="text/event-stream",
            )

        try:
            resp = await client.generate(
                prompt=prompt,
                model=body.model,
                temperature=body.temperature,
                top_p=body.top_p,
                max_tokens=body.max_tokens,
                stop=body.stop,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

        return ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            model=resp.model,
            choices=[
                ChatChoice(
                    message=ChatMessage(role="assistant", content=resp.text),
                    finish_reason=resp.finish_reason or "stop",
                )
            ],
            usage=resp.usage,
        )

    @app.post("/v1/embeddings")
    async def embeddings_endpoint(body: EmbeddingRequestBody):
        try:
            resp = await client.embeddings(input=body.input, model=body.model)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        data = [
            EmbeddingData(index=i, embedding=emb)
            for i, emb in enumerate(resp.embeddings)
        ]
        return EmbeddingResponseBody(
            data=data,
            model=resp.model,
            usage=resp.usage,
        )

    @app.get("/v1/providers")
    async def list_providers():
        return {"providers": LLMClient.list_providers()}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


# -----------------------------------------------------------------------
# Streaming helpers
# -----------------------------------------------------------------------


async def _stream_generate(
    client: LLMClient, body: "GenerateRequestBody"
) -> AsyncIterator[str]:
    import json

    async for chunk in client.stream(
        prompt=body.prompt,
        model=body.model,
        temperature=body.temperature,
        top_p=body.top_p,
        max_tokens=body.max_tokens,
        stop=body.stop,
    ):
        yield f"data: {json.dumps({'text': chunk.text})}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_chat(
    client: LLMClient, body: "ChatCompletionRequest", prompt: str
) -> AsyncIterator[str]:
    import json

    async for chunk in client.stream(
        prompt=prompt,
        model=body.model,
        temperature=body.temperature,
        top_p=body.top_p,
        max_tokens=body.max_tokens,
        stop=body.stop,
    ):
        data = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "model": chunk.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk.text},
                    "finish_reason": chunk.finish_reason,
                }
            ],
        }
        yield f"data: {json.dumps(data)}\n\n"
    yield "data: [DONE]\n\n"
