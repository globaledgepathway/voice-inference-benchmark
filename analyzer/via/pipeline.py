"""One voice-agent turn: user audio -> STT -> streaming LLM -> TTS, with per-stage timing.

Timeline of a streaming turn (all times relative to t0 = user finished speaking):

    t0 ──STT──▶ t_stt ──LLM TTFT──▶ first token ──…──▶ first sentence ──TTS TTFB──▶ FIRST AUDIO
                                                     (TTS starts here, while LLM keeps generating)

`time_to_first_audio_ms` is the number a caller feels: silence after they stop talking.
In non-streaming mode TTS waits for the complete LLM response.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import asdict, dataclass

import httpx

from .clients import chat_stream, now, synthesize, transcribe
from .config import Config

# A sentence boundary followed by whitespace, after at least a few words.
_SENTENCE_END = re.compile(r"[.!?;:](\s|$)")
MIN_FIRST_CHUNK_CHARS = 20
SPEAKING_RATE_WPS = 2.5  # ~150 words/min, used to estimate agent playback time


@dataclass
class TurnResult:
    session: int
    turn: int
    concurrency: int
    ok: bool
    error: str = ""
    user_audio_s: float = 0.0
    agent_audio_s: float = 0.0
    stt_ms: float = 0.0
    llm_ttft_ms: float = 0.0
    llm_total_ms: float = 0.0
    tts_ttfb_ms: float = 0.0
    time_to_first_audio_ms: float = 0.0
    turn_total_ms: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    output_chars: int = 0
    decode_tps: float = 0.0
    transcript: str = ""
    response: str = ""

    def row(self) -> dict:
        return asdict(self)


def first_chunk_end(text: str) -> int | None:
    """Index where a speakable first sentence ends, or None if there isn't one yet."""
    for m in _SENTENCE_END.finditer(text):
        if m.end() >= MIN_FIRST_CHUNK_CHARS:
            return m.end()
    return None


async def run_turn(
    client: httpx.AsyncClient,
    cfg: Config,
    wav: bytes,
    history: list[dict],
    *,
    streaming: bool,
    result: TurnResult,
) -> TurnResult:
    t0 = now()

    # 1. Speech-to-text
    transcript = await transcribe(client, cfg.stt, wav)
    t_stt = now()

    # 2. LLM, streamed
    messages = [{"role": "system", "content": cfg.system_prompt}, *history,
                {"role": "user", "content": transcript or "(silence)"}]
    t_first_token = None
    text = ""
    chunks = 0
    usage = None
    first_tts: asyncio.Task | None = None
    first_end = 0

    async for kind, val in chat_stream(client, cfg.llm, messages):
        if kind == "usage":
            usage = val
            continue
        if t_first_token is None:
            t_first_token = now()
        text += val
        chunks += 1
        if streaming and first_tts is None:
            end = first_chunk_end(text)
            if end:
                first_end = end
                first_tts = asyncio.create_task(synthesize(client, cfg.tts, text[:end].strip()))
    t_llm = now()
    t_first_token = t_first_token or t_llm

    # 3. Text-to-speech. Streaming: first sentence already in flight; synthesize the rest too
    # so the TTS server sees realistic load (not on the first-audio critical path).
    if first_tts is None:
        first_end = len(text)
        speak = text.strip() or "Sorry, I didn't catch that."
        first_tts = asyncio.create_task(synthesize(client, cfg.tts, speak))
    first = await first_tts
    rest = text[first_end:]
    if rest.strip():
        await synthesize(client, cfg.tts, rest.strip())
    t_end = now()

    out_tokens = (usage or {}).get("completion_tokens") or chunks
    decode_s = max(t_llm - t_first_token, 1e-6)
    result.ok = True
    result.transcript = transcript
    result.response = text
    result.stt_ms = (t_stt - t0) * 1000
    result.llm_ttft_ms = (t_first_token - t_stt) * 1000
    result.llm_total_ms = (t_llm - t_stt) * 1000
    result.tts_ttfb_ms = (first["t_first"] - first["t_start"]) * 1000
    result.time_to_first_audio_ms = (first["t_first"] - t0) * 1000
    result.turn_total_ms = (t_end - t0) * 1000
    result.prompt_tokens = (usage or {}).get("prompt_tokens") or 0
    result.output_tokens = out_tokens
    result.output_chars = len(text)
    result.decode_tps = (out_tokens - 1) / decode_s if out_tokens > 1 else 0.0
    result.agent_audio_s = len(text.split()) / SPEAKING_RATE_WPS
    return result
