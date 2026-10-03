#!/usr/bin/env python3
"""LLM latency/throughput benchmark for voice-agent workloads.

Works against any OpenAI-compatible /chat/completions endpoint:
  - vLLM on AMD MI300X (ROCm)
  - vLLM on NVIDIA H100 (CUDA)
  - Azure AI Foundry managed models

Measures per request: time-to-first-token (TTFT), inter-token latency (ITL),
end-to-end latency, output tokens/sec. Sweeps concurrency levels.
Standard library only, so it runs on any GPU box without extra installs.

Usage:
  python bench/llm_bench.py --config configs/targets.json --target mi300x_vllm \
      --concurrency 1 4 16 32 --requests 64
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone


def load_prompts(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def pct(values, p):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def build_request(target, messages, max_tokens, extra=None):
    url = target["base_url"].rstrip("/") + "/chat/completions"
    if target.get("api_version"):
        url += f"?api-version={target['api_version']}"
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(target.get("api_key_env", ""), "")
    if key:
        if target.get("auth_header") == "api-key":
            headers["api-key"] = key
        else:
            headers["Authorization"] = f"Bearer {key}"
    body = {
        "model": target["model"],
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    body.update(extra or {})  # e.g. {"priority": 0} for vLLM --scheduling-policy priority
    return urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")


def run_one(target, prompt, max_tokens, timeout, extra=None):
    req = build_request(target, prompt["messages"], max_tokens, extra)
    t0 = time.perf_counter()
    first = None
    token_times = []
    usage = {}
    text_parts = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices", []):
                    delta = (ch.get("delta") or {}).get("content")
                    if delta:
                        now = time.perf_counter()
                        if first is None:
                            first = now
                        token_times.append(now)
                        text_parts.append(delta)
    except Exception as e:  # record failures instead of crashing the sweep
        return {"ok": False, "error": str(e)[:300], "id": prompt.get("id")}

    end = time.perf_counter()
    out_tokens = usage.get("completion_tokens") or len(token_times)
    itls = [b - a for a, b in zip(token_times, token_times[1:])]
    decode_time = (end - first) if first else 0
    return {
        "ok": first is not None,
        "id": prompt.get("id"),
        "ttft_s": (first - t0) if first else None,
        "e2e_s": end - t0,
        "itl_mean_s": statistics.mean(itls) if itls else None,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": out_tokens,
        "decode_tok_per_s": (out_tokens / decode_time) if decode_time > 0 else None,
    }


def sweep(target, prompts, concurrency, n_requests, max_tokens, timeout):
    jobs = [prompts[i % len(prompts)] for i in range(n_requests)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(lambda p: run_one(target, p, max_tokens, timeout), jobs))
    wall = time.perf_counter() - t0
    ok = [r for r in results if r["ok"]]
    ttft = [r["ttft_s"] for r in ok]
    e2e = [r["e2e_s"] for r in ok]
    itl = [r["itl_mean_s"] for r in ok if r["itl_mean_s"] is not None]
    out_tok = sum(r["completion_tokens"] or 0 for r in ok)
    in_tok = sum(r["prompt_tokens"] or 0 for r in ok)
    summary = {
        "target": target["name"],
        "hardware": target.get("hardware", ""),
        "model": target["model"],
        "concurrency": concurrency,
        "requests": n_requests,
        "success": len(ok),
        "errors": len(results) - len(ok),
        "wall_s": round(wall, 3),
        "ttft_p50_ms": round(pct(ttft, 50) * 1000, 1) if ttft else None,
        "ttft_p95_ms": round(pct(ttft, 95) * 1000, 1) if ttft else None,
        "itl_p50_ms": round(pct(itl, 50) * 1000, 2) if itl else None,
        "e2e_p50_s": round(pct(e2e, 50), 3) if e2e else None,
        "e2e_p95_s": round(pct(e2e, 95), 3) if e2e else None,
        "total_input_tokens": in_tok,
        "total_output_tokens": out_tok,
        "agg_output_tok_per_s": round(out_tok / wall, 1) if wall else None,
        "req_per_s": round(len(ok) / wall, 3) if wall else None,
    }
    return summary, results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/targets.json")
    ap.add_argument("--target", required=True, help="target name from the config")
    ap.add_argument("--prompts", default="prompts/voice_turns.jsonl")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--requests", type=int, default=32, help="requests per concurrency level")
    ap.add_argument("--max-tokens", type=int, default=150, help="voice replies are short; 150 ~ 20s of speech")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    target = next((t for t in cfg["targets"] if t["name"] == args.target), None)
    if not target:
        sys.exit(f"target '{args.target}' not in {args.config}")
    prompts = load_prompts(args.prompts)

    print(f"warming up {target['name']} ({args.warmup} requests)...")
    for p in prompts[: args.warmup]:
        r = run_one(target, p, args.max_tokens, args.timeout)
        if not r["ok"]:
            sys.exit(f"warmup failed: {r.get('error')}")

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summaries = []
    for c in args.concurrency:
        print(f"concurrency={c} ...", flush=True)
        s, raw = sweep(target, prompts, c, args.requests, args.max_tokens, args.timeout)
        summaries.append(s)
        print(json.dumps(s, indent=2))
        with open(os.path.join(args.out, f"{target['name']}_c{c}_{stamp}_raw.jsonl"), "w") as f:
            for r in raw:
                f.write(json.dumps(r) + "\n")

    path = os.path.join(args.out, f"{target['name']}_{stamp}_summary.json")
    with open(path, "w") as f:
        json.dump({"target": target, "max_tokens": args.max_tokens, "runs": summaries}, f, indent=2)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
