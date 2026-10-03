"""Thin async clients for the three OpenAI-compatible stages, instrumented for timing."""

from __future__ import annotations

import json
import time
from typing import AsyncIterator

import httpx

from .config import Endpoint

now = time.perf_counter


async def transcribe(client: httpx.AsyncClient, ep: Endpoint, wav: bytes) -> str:
    """POST /audio/transcriptions (vLLM Whisper, OpenAI, Azure)."""
    data = {"model": ep.model, "response_format": "json"}
    if ep.extra.get("language"):
        data["language"] = ep.extra["language"]
    r = await client.post(
        ep.url("audio/transcriptions"),
        headers=ep.headers,
        data=data,
        files={"file": ("audio.wav", wav, "audio/wav")},
    )
    r.raise_for_status()
    return r.json().get("text", "").strip()


async def chat_stream(
    client: httpx.AsyncClient, ep: Endpoint, messages: list[dict]
) -> AsyncIterator[tuple[str, object]]:
    """Stream /chat/completions. Yields ("token", text) and finally ("usage", dict|None)."""
    payload = {
        "model": ep.model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": ep.extra.get("max_tokens", 150),
        "temperature": ep.extra.get("temperature", 0.7),
    }
    usage = None
    async with client.stream(
        "POST", ep.url("chat/completions"), headers=ep.headers, json=payload
    ) as r:
        if r.status_code >= 400:
            await r.aread()
            r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices") or []:
                text = (choice.get("delta") or {}).get("content")
                if text:
                    yield "token", text
    yield "usage", usage


async def synthesize(client: httpx.AsyncClient, ep: Endpoint, text: str) -> dict:
    """POST /audio/speech and stream the audio. Returns absolute first-byte/done times."""
    payload = {
        "model": ep.model,
        "input": text,
        "voice": ep.extra.get("voice", "alloy"),
        "response_format": ep.extra.get("response_format", "wav"),
    }
    t_start = now()
    t_first = None
    n = 0
    async with client.stream(
        "POST", ep.url("audio/speech"), headers=ep.headers, json=payload
    ) as r:
        if r.status_code >= 400:
            await r.aread()
            r.raise_for_status()
        async for chunk in r.aiter_bytes():
            if chunk and t_first is None:
                t_first = now()
            n += len(chunk)
    t_done = now()
    return {"t_start": t_start, "t_first": t_first or t_done, "t_done": t_done, "bytes": n}


async def synthesize_to_bytes(client: httpx.AsyncClient, ep: Endpoint, text: str) -> bytes:
    payload = {
        "model": ep.model,
        "input": text,
        "voice": ep.extra.get("voice", "alloy"),
        "response_format": "wav",
    }
    r = await client.post(ep.url("audio/speech"), headers=ep.headers, json=payload)
    r.raise_for_status()
    return r.content
