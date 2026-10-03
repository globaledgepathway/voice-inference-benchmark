"""Aggregate turn results into per-concurrency summaries, including cost per conversation-minute."""

from __future__ import annotations

import math

LATENCY_FIELDS = ["stt_ms", "llm_ttft_ms", "llm_total_ms", "tts_ttfb_ms",
                  "time_to_first_audio_ms", "turn_total_ms"]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def managed_cost_usd(rows: list[dict], pricing: dict) -> float:
    """Pay-per-use cost for managed APIs (prices are inputs in the config)."""
    total = 0.0
    for r in rows:
        total += r["user_audio_s"] / 60 * pricing.get("stt_usd_per_min", 0)
        total += r["prompt_tokens"] / 1e6 * pricing.get("llm_input_usd_per_1m_tokens", 0)
        total += r["output_tokens"] / 1e6 * pricing.get("llm_output_usd_per_1m_tokens", 0)
        total += r["output_chars"] / 1e6 * pricing.get("tts_usd_per_1m_chars", 0)
    return total


def summarize(rows: list[dict], *, concurrency: int, wall_s: float,
              conversation_s: float, cost_cfg: dict, sla: dict) -> dict:
    ok = [r for r in rows if r["ok"]]
    out: dict = {
        "concurrency": concurrency,
        "turns": len(rows),
        "errors": len(rows) - len(ok),
        "error_rate": (len(rows) - len(ok)) / len(rows) if rows else 0.0,
        "wall_s": round(wall_s, 2),
        "conversation_min": round(conversation_s / 60, 3),
    }
    for f in LATENCY_FIELDS:
        vals = [r[f] for r in ok]
        out[f"{f}_p50"] = round(percentile(vals, 50), 1)
        out[f"{f}_p95"] = round(percentile(vals, 95), 1)
    out["decode_tps_per_session_p50"] = round(percentile([r["decode_tps"] for r in ok], 50), 1)
    out["output_tokens_per_s_total"] = round(sum(r["output_tokens"] for r in ok) / wall_s, 1) if wall_s else 0

    # Cost per conversation-minute
    mode = cost_cfg.get("mode", "gpu_hourly")
    if mode == "gpu_hourly":
        hourly = cost_cfg.get("gpu_hourly_usd", 0) * cost_cfg.get("gpu_count", 1)
        cost = hourly * wall_s / 3600
    else:  # per_use (managed APIs)
        cost = managed_cost_usd(ok, cost_cfg.get("pricing", {}))
    out["run_cost_usd"] = round(cost, 4)
    out["cost_per_conversation_min_usd"] = (
        round(cost / (conversation_s / 60), 5) if conversation_s else float("nan")
    )
    out["sla_ms"] = sla["time_to_first_audio_p95_ms"]
    out["meets_sla"] = bool(
        ok
        and out["time_to_first_audio_ms_p95"] <= sla["time_to_first_audio_p95_ms"]
        and out["error_rate"] <= sla["max_error_rate"]
    )
    return out
