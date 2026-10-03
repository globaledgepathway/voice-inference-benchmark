#!/usr/bin/env python3
"""Noisy-neighbour benchmark: two tenants sharing one GPU / one inference server.

  Tenant A ("voice")  interactive voice turns at a fixed concurrency, latency-sensitive
  Tenant B ("batch")  long-prompt, long-output jobs that keep the GPU busy

Three phases, same tenant-A load each time:
  1. alone       tenant A only (baseline)
  2. shared      tenant A + tenant B at the same priority
  3. priority    tenant A + tenant B, A sent with higher priority
                 (only with --priority; start vLLM with --scheduling-policy priority)

Reports, per phase: tenant A TTFT p50/p95, ITL p50, E2E p95, SLA pass/fail;
tenant B throughput; and a chargeback split of the GPU cost for the shared window,
allocated by each tenant's share of processed tokens (input + output).

Standard library only. Usage:
  python bench/noisy_neighbor.py --target mi300x_vllm --voice-conc 8 --batch-conc 32 \
      --voice-requests 96 --priority
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_bench import load_prompts, pct, run_one  # noqa: E402


def batch_prompt(words):
    """A long 'summarise this call log' job, roughly `words` words of input."""
    line = ("Agent: thanks for calling, how can I help. Caller: I need to change my booking "
            "to next Tuesday and check whether the refund for the old ticket went through. ")
    body = (line * (words // len(line.split()) + 1)).split()[:words]
    return {"id": "batch", "messages": [
        {"role": "system", "content": "You are an analyst. Write a detailed report."},
        {"role": "user", "content": "Summarise this call log, list every issue raised, and "
                                    "suggest follow-ups:\n" + " ".join(body)}]}


class BatchTenant:
    """Keeps `conc` long requests in flight until stopped."""

    def __init__(self, target, conc, words, max_tokens, timeout, extra):
        self.args = (target, batch_prompt(words), max_tokens, timeout, extra)
        self.conc, self.stop, self.lock = conc, threading.Event(), threading.Lock()
        self.results, self.threads = [], []

    def _loop(self):
        while not self.stop.is_set():
            r = run_one(*self.args)
            with self.lock:
                self.results.append(r)

    def start(self):
        self.t0 = time.perf_counter()
        self.threads = [threading.Thread(target=self._loop, daemon=True) for _ in range(self.conc)]
        for t in self.threads:
            t.start()

    def finish(self):
        self.stop.set()
        wall = time.perf_counter() - self.t0
        for t in self.threads:
            t.join(timeout=300)
        ok = [r for r in self.results if r["ok"]]
        out_tok = sum(r["completion_tokens"] or 0 for r in ok)
        in_tok = sum(r["prompt_tokens"] or 0 for r in ok)
        return {"requests": len(self.results), "errors": len(self.results) - len(ok),
                "input_tokens": in_tok, "output_tokens": out_tok,
                "output_tok_per_s": round(out_tok / wall, 1) if wall else None}


def voice_phase(target, prompts, conc, n, max_tokens, timeout, extra):
    jobs = [prompts[i % len(prompts)] for i in range(n)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(conc) as ex:
        res = list(ex.map(lambda p: run_one(target, p, max_tokens, timeout, extra), jobs))
    wall = time.perf_counter() - t0
    ok = [r for r in res if r["ok"]]
    ms = lambda k, p: round(pct([r[k] for r in ok if r[k] is not None], p) * 1000, 1) if ok else None
    return {"requests": n, "errors": n - len(ok), "wall_s": round(wall, 2),
            "ttft_p50_ms": ms("ttft_s", 50), "ttft_p95_ms": ms("ttft_s", 95),
            "itl_p50_ms": ms("itl_mean_s", 50), "e2e_p95_ms": ms("e2e_s", 95),
            "input_tokens": sum(r["prompt_tokens"] or 0 for r in ok),
            "output_tokens": sum(r["completion_tokens"] or 0 for r in ok)}


def run_phase(name, a, target, prompts, batch_extra, voice_extra):
    print(f"phase: {name} ...", flush=True)
    batch = None
    if name != "alone":
        batch = BatchTenant(target, a.batch_conc, a.batch_input_words, a.batch_max_tokens,
                            a.timeout, batch_extra)
        batch.start()
        time.sleep(a.ramp_s)  # let the batch tenant fill the GPU first
    t0 = time.perf_counter()
    voice = voice_phase(target, prompts, a.voice_conc, a.voice_requests, a.max_tokens,
                        a.timeout, voice_extra)
    window = time.perf_counter() - t0
    out = {"phase": name, "voice": voice, "window_s": round(window, 2)}
    if batch:
        out["batch"] = batch.finish()
    return out


def chargeback(phase, gpu_hourly):
    """Split the GPU cost of the measured window between tenants by processed tokens."""
    v, b = phase["voice"], phase.get("batch")
    vt = v["input_tokens"] + v["output_tokens"]
    bt = (b["input_tokens"] + b["output_tokens"]) if b else 0
    cost = gpu_hourly * phase["window_s"] / 3600
    total = vt + bt or 1
    return {"window_cost_usd": round(cost, 5),
            "voice_share": round(vt / total, 3), "voice_usd": round(cost * vt / total, 5),
            "batch_share": round(bt / total, 3), "batch_usd": round(cost * bt / total, 5)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/targets.json")
    ap.add_argument("--target", required=True)
    ap.add_argument("--prompts", default="prompts/voice_turns.jsonl")
    ap.add_argument("--voice-conc", type=int, default=8)
    ap.add_argument("--voice-requests", type=int, default=96)
    ap.add_argument("--max-tokens", type=int, default=150)
    ap.add_argument("--batch-conc", type=int, default=32)
    ap.add_argument("--batch-input-words", type=int, default=1500)
    ap.add_argument("--batch-max-tokens", type=int, default=512)
    ap.add_argument("--ramp-s", type=float, default=10)
    ap.add_argument("--priority", action="store_true",
                    help="add a phase with voice at higher priority (vLLM --scheduling-policy priority)")
    ap.add_argument("--ttft-sla-ms", type=float, default=500)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    target = next((t for t in cfg["targets"] if t["name"] == a.target), None)
    if not target:
        sys.exit(f"target '{a.target}' not in {a.config}")
    prompts = load_prompts(a.prompts)
    gpu_hourly = target.get("pricing", {}).get("gpu_hourly_usd", 0.0)

    for p in prompts[:3]:  # warm-up
        if not run_one(target, p, a.max_tokens, a.timeout)["ok"]:
            sys.exit("warm-up failed; is the server up?")

    phases = [run_phase("alone", a, target, prompts, None, None),
              run_phase("shared", a, target, prompts, None, None)]
    if a.priority:  # vLLM: lower value = served first
        phases.append(run_phase("priority", a, target, prompts, {"priority": 10}, {"priority": 0}))
    for ph in phases:
        if ph["phase"] != "alone":
            ph["chargeback"] = chargeback(ph, gpu_hourly)

    base = phases[0]["voice"]["ttft_p95_ms"] or 1
    print(f"\n## Noisy neighbour: {target['name']}  (voice conc {a.voice_conc}, "
          f"batch conc {a.batch_conc} x ~{a.batch_input_words} words in / {a.batch_max_tokens} out)\n")
    print("| phase | voice TTFT p50 ms | voice TTFT p95 ms | vs alone | ITL p50 ms | E2E p95 ms "
          f"| SLA {a.ttft_sla_ms:.0f} ms | batch out tok/s | voice $ / batch $ |")
    print("|---|---|---|---|---|---|---|---|---|")
    for ph in phases:
        v, b, c = ph["voice"], ph.get("batch"), ph.get("chargeback")
        sla = "pass" if v["ttft_p95_ms"] is not None and v["ttft_p95_ms"] <= a.ttft_sla_ms and not v["errors"] else "FAIL"
        bill = f"{c['voice_usd']:.5f} / {c['batch_usd']:.5f}" if c else "-"
        batch_tps = b["output_tok_per_s"] if b else "-"
        slowdown = (v["ttft_p95_ms"] or 0) / base
        print(f"| {ph['phase']} | {v['ttft_p50_ms']} | {v['ttft_p95_ms']} | {slowdown:.1f}x | "
              f"{v['itl_p50_ms']} | {v['e2e_p95_ms']} | {sla} | {batch_tps} | {bill} |")

    os.makedirs(a.out, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(a.out, f"noisy_neighbor_{target['name']}_{stamp}.json")
    json.dump({"target": target, "args": vars(a), "phases": phases}, open(path, "w"), indent=2)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
