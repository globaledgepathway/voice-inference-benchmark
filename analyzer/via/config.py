"""Load a benchmark config (YAML) with ${ENV_VAR} expansion."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


@dataclass
class Endpoint:
    """An OpenAI-compatible endpoint (vLLM, Kokoro-FastAPI, OpenAI, Azure AI Foundry)."""

    base_url: str
    model: str
    api_key: str = "EMPTY"
    auth_header: str = "Authorization"  # Azure key auth uses "api-key"
    extra: dict = field(default_factory=dict)

    @property
    def headers(self) -> dict:
        if self.auth_header.lower() == "authorization":
            return {"Authorization": f"Bearer {self.api_key}"}
        return {self.auth_header: self.api_key}

    def url(self, path: str) -> str:
        return self.base_url.rstrip("/") + "/" + path.lstrip("/")


@dataclass
class Config:
    name: str
    stt: Endpoint
    llm: Endpoint
    tts: Endpoint
    load: dict
    cost: dict
    sla: dict
    raw: dict

    @property
    def system_prompt(self) -> str:
        return self.llm.extra.get(
            "system_prompt",
            "You are a friendly voice assistant. Reply in 1-3 short spoken sentences.",
        )


def _endpoint(d: dict) -> Endpoint:
    known = {"base_url", "model", "api_key", "auth_header"}
    return Endpoint(
        base_url=d["base_url"],
        model=d["model"],
        api_key=d.get("api_key", "EMPTY"),
        auth_header=d.get("auth_header", "Authorization"),
        extra={k: v for k, v in d.items() if k not in known},
    )


def load_config(path: str | Path) -> Config:
    raw = _expand(yaml.safe_load(Path(path).read_text()))
    return Config(
        name=raw["name"],
        stt=_endpoint(raw["stt"]),
        llm=_endpoint(raw["llm"]),
        tts=_endpoint(raw["tts"]),
        load={
            "concurrency": [1, 2, 4, 8, 16, 32, 64, 100],
            "turns_per_session": 5,
            "pacing": "realtime",  # realtime | max
            "streaming": True,
            "timeout_s": 120,
            **raw.get("load", {}),
        },
        cost=raw.get("cost", {}),
        sla={"time_to_first_audio_p95_ms": 1000, "max_error_rate": 0.01, **raw.get("sla", {})},
        raw=raw,
    )
