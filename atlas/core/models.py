"""
Normalized data contracts for the Atlas LLM inference framework.

These dataclasses define the unified request/response shapes that all
providers must accept and return, ensuring a consistent API regardless
of the underlying backend.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class GenerateRequest:
    """Normalized request for text generation."""

    prompt: str
    model: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    stop: Optional[List[str]] = None
    extra: Dict = field(default_factory=dict)


@dataclass
class GenerateResponse:
    """Normalized response from text generation."""

    text: str
    model: str
    provider: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None


@dataclass
class StreamChunk:
    """A single chunk from a streaming generation response."""

    text: str
    model: str
    provider: str
    finish_reason: Optional[str] = None


@dataclass
class EmbeddingRequest:
    """Normalized request for embedding generation."""

    input: Union[str, List[str]]
    model: Optional[str] = None
    extra: Dict = field(default_factory=dict)


@dataclass
class EmbeddingResponse:
    """Normalized response from embedding generation."""

    embeddings: List[List[float]]
    model: str
    provider: str
    usage: Optional[Dict[str, int]] = None
