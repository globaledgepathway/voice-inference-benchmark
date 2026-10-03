"""Build the caller test set: one WAV per line of data/prompts.txt, via the configured TTS.

Using TTS for the caller keeps the test set reproducible and license-clean. Swap in real
recordings (16 kHz mono WAV) in data/audio/ for a more realistic STT workload.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from .clients import synthesize_to_bytes
from .config import Config


async def make_clips(cfg: Config, prompts_path: str, audio_dir: str, voice: str | None) -> None:
    out = Path(audio_dir)
    out.mkdir(parents=True, exist_ok=True)
    prompts = [l.strip() for l in Path(prompts_path).read_text().splitlines()
               if l.strip() and not l.startswith("#")]
    if voice:
        cfg.tts.extra["voice"] = voice
    async with httpx.AsyncClient(timeout=120) as client:
        for i, text in enumerate(prompts):
            wav = await synthesize_to_bytes(client, cfg.tts, text)
            (out / f"{i:03d}.wav").write_bytes(wav)
    print(f"Wrote {len(prompts)} clips to {out}")
