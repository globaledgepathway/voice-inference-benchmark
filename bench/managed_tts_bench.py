#!/usr/bin/env python3
"""Managed text-to-speech latency: time to first audio byte, streaming.

Providers (any you have keys for):
  azure   Azure AI Speech REST (env AZURE_SPEECH_KEY, AZURE_SPEECH_REGION; voice --azure-voice)
  openai  OpenAI /v1/audio/speech (env OPENAI_API_KEY; model --openai-model)

Same first-sentence texts as a voice agent would speak. Reports, per provider and
concurrency: time to first audio byte (what the caller waits for), time to full audio,
errors. Standard library only.

  python bench/managed_tts_bench.py --providers azure openai --concurrency 1 4 8 --requests 32
"""
import argparse
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from xml.sax.saxutils import escape

SENTENCES = [
    "Sure, I can help you with that booking today.",
    "Your flight to Miami leaves at nine fifteen tomorrow morning, and I have a window seat available for you.",
    "I have moved your appointment to Thursday at three in the afternoon.",
    "Thanks for your patience, your refund was processed on Monday and should show up within five business days.",
    "I understand the price feels high, so let me walk you through what is included in the premium plan.",
    "Got it, I have updated the delivery address to the one on Maple Street.",
    "The earliest dental cleaning we have next week is Tuesday at eight thirty in the morning.",
    "Your order shipped yesterday and the tracking number has been sent to your email.",
]


def pct(v, p):
    s = sorted(v)
    if not s:
        return None
    k = (len(s) - 1) * p / 100
    lo = int(k); hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def azure_req(text, a):
    region = os.environ["AZURE_SPEECH_REGION"]
    ssml = (f"<speak version='1.0' xml:lang='en-US'><voice name='{a.azure_voice}'>"
            f"{escape(text)}</voice></speak>")
    return urllib.request.Request(
        f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1",
        data=ssml.encode(), method="POST",
        headers={"Ocp-Apim-Subscription-Key": os.environ["AZURE_SPEECH_KEY"],
                 "Content-Type": "application/ssml+xml",
                 "X-Microsoft-OutputFormat": "raw-24khz-16bit-mono-pcm",
                 "User-Agent": "voice-inference-benchmark"})


def openai_req(text, a):
    body = {"model": a.openai_model, "voice": a.openai_voice, "input": text, "response_format": "pcm"}
    return urllib.request.Request(
        "https://api.openai.com/v1/audio/speech", data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                 "Content-Type": "application/json"})


BUILD = {"azure": azure_req, "openai": openai_req}


def one(provider, text, a):
    req = BUILD[provider](text, a)
    t0 = time.perf_counter(); first = None; n = 0
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            while True:
                chunk = r.read1(4096) if hasattr(r, "read1") else r.read(4096)
                if not chunk:
                    break
                if first is None:
                    first = time.perf_counter()
                n += len(chunk)
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}
    end = time.perf_counter()
    return {"ok": first is not None, "first_byte_s": (first - t0) if first else None,
            "total_s": end - t0, "audio_s": n / 2 / 24000}  # 16-bit mono 24 kHz PCM


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--providers", nargs="+", default=["azure", "openai"], choices=list(BUILD))
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--azure-voice", default="en-US-AvaMultilingualNeural")
    ap.add_argument("--openai-model", default="gpt-4o-mini-tts")
    ap.add_argument("--openai-voice", default="alloy")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    need = {"azure": ["AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION"], "openai": ["OPENAI_API_KEY"]}
    providers = [p for p in a.providers if all(os.environ.get(k) for k in need[p])]
    skipped = set(a.providers) - set(providers)
    if skipped:
        print("skipping (no key in env):", ", ".join(sorted(skipped)))
    rows = []
    print("provider  conc  ok/err  first-audio p50/p95 ms  full-audio p50 ms  audio s p50")
    for p in providers:
        for t in SENTENCES[:2]:  # warm-up: DNS, TLS, auth
            one(p, t, a)
        for c in a.concurrency:
            jobs = [SENTENCES[i % len(SENTENCES)] for i in range(a.requests)]
            with ThreadPoolExecutor(c) as ex:
                res = list(ex.map(lambda t: one(p, t, a), jobs))
            ok = [r for r in res if r["ok"]]
            fb = [r["first_byte_s"] * 1000 for r in ok]
            row = {"provider": p, "concurrency": c, "requests": len(res), "errors": len(res) - len(ok),
                   "first_audio_p50_ms": round(pct(fb, 50)) if fb else None,
                   "first_audio_p95_ms": round(pct(fb, 95)) if fb else None,
                   "full_audio_p50_ms": round(pct([r["total_s"] * 1000 for r in ok], 50)) if ok else None,
                   "audio_s_p50": round(pct([r["audio_s"] for r in ok], 50), 2) if ok else None}
            rows.append(row)
            print(f"{p:<8} {c:>5}  {len(ok)}/{row['errors']}  {row['first_audio_p50_ms']}/{row['first_audio_p95_ms']}"
                  f"  {row['full_audio_p50_ms']}  {row['audio_s_p50']}", flush=True)
            if row["errors"]:
                print("   sample error:", next(r for r in res if not r["ok"])["error"])
    os.makedirs(a.out, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(a.out, f"managed_tts_{stamp}_summary.json")
    json.dump({"args": vars(a), "runs": rows}, open(path, "w"), indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()
