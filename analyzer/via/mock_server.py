"""Fake OpenAI-compatible STT/LLM/TTS server for dry runs (no GPU, no API keys).

Latency grows with in-flight requests to imitate GPU contention, so a sweep against it
produces realistic-looking curves. Numbers are synthetic: never publish them.

    python -m via.mock_server --port 9000
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import random
import struct
import time

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

INFLIGHT = {"stt": 0, "llm": 0, "tts": 0}
CAPACITY = {"stt": 16, "llm": 32, "tts": 12}  # contention knee for each stage
REPLIES = [
    "Sure, I can help with that. Your order shipped yesterday and should arrive on Friday.",
    "Good question. The premium plan includes priority support and unlimited projects.",
    "I'm sorry about that. I've reset your password, so check your email for the link.",
    "Yes, we're open until nine tonight. Is there anything else I can help you with?",
]


def _slow(stage: str, base_s: float) -> float:
    load = INFLIGHT[stage] / CAPACITY[stage]
    return base_s * (1 + load ** 2) * random.uniform(0.85, 1.15)


def _wav(seconds: float, rate: int = 16000) -> bytes:
    n = int(seconds * rate)
    buf = io.BytesIO()
    data = b"\x00\x00" * n
    buf.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16))
    buf.write(b"data" + struct.pack("<I", len(data)) + data)
    return buf.getvalue()


async def transcriptions(request: Request):
    INFLIGHT["stt"] += 1
    try:
        await request.form()
        await asyncio.sleep(_slow("stt", 0.12))
        return JSONResponse({"text": "Hi, can you check the status of my order please?"})
    finally:
        INFLIGHT["stt"] -= 1


async def chat(request: Request):
    body = await request.json()
    reply = random.choice(REPLIES)
    words = reply.split(" ")

    async def gen():
        INFLIGHT["llm"] += 1
        try:
            await asyncio.sleep(_slow("llm", 0.08))  # prefill / TTFT
            for i, w in enumerate(words):
                tok = w if i == 0 else " " + w
                chunk = {"choices": [{"index": 0, "delta": {"content": tok}}]}
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(_slow("llm", 0.012))  # per-token decode
            usage = {"prompt_tokens": 40 + len(json.dumps(body["messages"])) // 4,
                     "completion_tokens": len(words)}
            yield f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            INFLIGHT["llm"] -= 1

    return StreamingResponse(gen(), media_type="text/event-stream")


async def speech(request: Request):
    body = await request.json()
    secs = max(0.5, len(body["input"].split()) / 2.5)
    audio = _wav(secs)

    async def gen():
        INFLIGHT["tts"] += 1
        try:
            await asyncio.sleep(_slow("tts", 0.10))
            for i in range(0, len(audio), 16000):
                yield audio[i:i + 16000]
                await asyncio.sleep(0.002)
        finally:
            INFLIGHT["tts"] -= 1

    return StreamingResponse(gen(), media_type="audio/wav")


app = Starlette(routes=[
    Route("/v1/audio/transcriptions", transcriptions, methods=["POST"]),
    Route("/v1/chat/completions", chat, methods=["POST"]),
    Route("/v1/audio/speech", speech, methods=["POST"]),
])


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9000)
    random.seed(int(time.time()))
    uvicorn.run(app, host="0.0.0.0", port=ap.parse_args().port, log_level="warning")
