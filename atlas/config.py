"""
Configuration handling for Atlas.

Supports construction from dictionaries, environment variables,
and YAML files so that no configuration is hardcoded.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ProviderConfig:
    """Configuration for a single provider instance."""

    provider: str
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: float = 120.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProviderConfig":
        """Create a ``ProviderConfig`` from a plain dictionary."""
        return cls(
            provider=data["provider"],
            model=data["model"],
            base_url=data.get("base_url"),
            api_key=data.get("api_key"),
            timeout=data.get("timeout", 120.0),
            extra={
                k: v
                for k, v in data.items()
                if k not in {"provider", "model", "base_url", "api_key", "timeout"}
            },
        )

    @classmethod
    def from_env(
        cls,
        prefix: str = "ATLAS",
    ) -> "ProviderConfig":
        """
        Build configuration from environment variables.

        Reads::

            {PREFIX}_PROVIDER   – provider name  (required)
            {PREFIX}_MODEL      – model name     (required)
            {PREFIX}_BASE_URL   – base URL
            {PREFIX}_API_KEY    – API key
            {PREFIX}_TIMEOUT    – timeout in seconds
        """
        provider = os.environ.get(f"{prefix}_PROVIDER", "")
        model = os.environ.get(f"{prefix}_MODEL", "")
        if not provider or not model:
            raise ValueError(
                f"Environment variables {prefix}_PROVIDER and {prefix}_MODEL "
                f"must be set."
            )
        return cls(
            provider=provider,
            model=model,
            base_url=os.environ.get(f"{prefix}_BASE_URL"),
            api_key=os.environ.get(f"{prefix}_API_KEY"),
            timeout=float(os.environ.get(f"{prefix}_TIMEOUT", "120")),
        )
