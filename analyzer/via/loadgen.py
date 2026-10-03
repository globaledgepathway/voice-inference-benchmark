"""Concurrency sweep: N simulated callers, each holding a multi-turn conversation.

Pacing
------
realtime : each caller "speaks" for the length of its audio clip, then "listens" for the
           estimated length of the agent's reply before the next turn. This is how a real
           voice agent loads a GPU, and is what cost per conversation-minute assumes.
max      : no pauses. Stress test to find raw throughput ceilings.
"""

from __future__ import annotations

import asyncio
import csv
import json
import random
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import Config
from .metrics import summarize
from .pipeline import TurnResult, run_turn

HISTORY_MESSAGES = 6  # keep the last 3 exchanges as context


@dataclass
class Clip:
    path: Path
    wav: bytes
    duration_s: float


def load_clips(audio_dir: str | Path) -> list[Clip]:
    clips = []
    for p in sorted(Path(audio_dir).glob("*.wav")):
        with wave.open(str(p)) as w:
            dur = w.getnframes() / float(w.getframerate())
        clips.append(Clip(p, p.read_bytes(), dur))
    if not clips:
        raise SystemExit(f"No .wav files in {audio_dir}. Run scripts/make_test_audio.py first.")
    return clips


async def _session(sid: int, client, cfg: Config, clips: list[Clip], concurrency: int,
                   rows: list[dict]) -> None:
    """Run one caller's multi-turn conversation, appending one row per turn."""
    rng = random.Random(sid)
    turns = cfg.load["turns_per_session"]
    realtime = cfg.load["pacing"] == "realtime"
    history: list[dict] = []
    # Stagger call starts so callers don't all speak in lock-step.
    if realtime:
        await asyncio.sleep(rng.uniform(0, 2.0))
    for t in range(turns):
        clip = clips[(sid * turns + t) % len(clips)]
        if realtime:
            await asyncio.sleep(clip.duration_s)  # caller is talking
        res = TurnResult(session=sid, turn=t, concurrency=concurrency, ok=False,
                         user_audio_s=clip.duration_s)
        try:
            await run_turn(client, cfg, clip.wav, history,
                           streaming=cfg.load["streaming"], result=res)
            history += [{"role": "user", "content": res.transcript},
                        {"role": "assistant", "content": res.response}]
            history = history[-HISTORY_MESSAGES:]
        except Exception as e:  # noqa: BLE001 — record and keep the caller going
            res.error = f"{type(e).__name__}: {e}"[:300]
        rows.append(res.row())
        if realtime and res.ok:
            await asyncio.sleep(res.agent_audio_s)  # caller listens to the reply


async def run_level(cfg: Config, clips: list[Clip], concurrency: int) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    limits = httpx.Limits(max_connections=concurrency * 4, max_keepalive_connections=concurrency * 4)
    timeout = httpx.Timeout(cfg.load["timeout_s"])
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        t0 = time.perf_counter()
        await asyncio.gather(*[
            _session(i, client, cfg, clips, concurrency, rows) for i in range(concurrency)
        ])
        wall = time.perf_counter() - t0
    # Conversation time = audio actually exchanged (caller speech + agent speech), so cost per
    # conversation-minute means the same thing in realtime and max pacing.
    talk_s = sum(r["user_audio_s"] + r["agent_audio_s"] for r in rows if r["ok"])
    summary = summarize(rows, concurrency=concurrency, wall_s=wall,
                        conversation_s=talk_s, cost_cfg=cfg.cost, sla=cfg.sla)
    return rows, summary


async def warmup(cfg: Config, clips: list[Clip]) -> None:
    async with httpx.AsyncClient(timeout=cfg.load["timeout_s"]) as client:
        res = TurnResult(session=-1, turn=0, concurrency=0, ok=False)
        await run_turn(client, cfg, clips[0].wav, [], streaming=True, result=res)


async def sweep(cfg: Config, audio_dir: str, out_dir: str, levels: list[int] | None = None) -> Path:
    clips = load_clips(audio_dir)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(out_dir) / f"{cfg.name}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(
        {**cfg.raw, "stt": {**cfg.raw["stt"], "api_key": "***"},
         "llm": {**cfg.raw["llm"], "api_key": "***"},
         "tts": {**cfg.raw["tts"], "api_key": "***"}}, indent=2))

    print(f"Warming up {cfg.name} ...")
    await warmup(cfg, clips)

    summaries = []
    for c in levels or cfg.load["concurrency"]:
        print(f"  concurrency={c:>3} ...", end="", flush=True)
        rows, s = await run_level(cfg, clips, c)
        with open(run_dir / f"turns_c{c}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        summaries.append(s)
        print(f" p95 first-audio {s['time_to_first_audio_ms_p95']:>7.0f} ms | "
              f"errors {s['errors']:>3} | ${s['cost_per_conversation_min_usd']}/conv-min | "
              f"{'OK' if s['meets_sla'] else 'SLA MISS'}")
        if s["error_rate"] > 0.5:
            print("  >50% errors, stopping sweep early.")
            break

    with open(run_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)
    print(f"Results: {run_dir}")
    return run_dir
