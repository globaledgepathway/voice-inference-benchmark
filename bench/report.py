#!/usr/bin/env python3
"""Turn benchmark summaries into a cost + latency comparison table.

Self-hosted GPUs (MI300X, H100): cost per 1M output tokens is
    gpu_hourly_usd / (aggregate output tok/s * 3600) * 1e6
taken at the highest concurrency that still meets the voice latency SLA
(TTFT p95 under --ttft-sla-ms), since that is the throughput you can sell.

Managed APIs (Azure AI Foundry): cost per token comes straight from the
price list in the config.

Cost per voice turn uses the average input/output tokens measured in the run.

Usage:
  python bench/report.py --config configs/targets.json --results results --ttft-sla-ms 500
"""
import argparse
import glob
import json
import os


def latest_summaries(results_dir):
    """Merge every llm_bench sweep per target. Files sort by timestamp, so a later
    re-run at the same concurrency replaces the earlier one."""
    by_target = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*_summary.json"))):
        data = json.load(open(path))
        if "target" not in data:  # voice-loop / TTS summaries, not llm_bench sweeps
            continue
        merged = by_target.setdefault(data["target"]["name"], {"target": data["target"], "runs": {}})
        merged["target"] = data["target"]
        for r in data["runs"]:
            merged["runs"][r["concurrency"]] = r
    for d in by_target.values():
        d["runs"] = [d["runs"][c] for c in sorted(d["runs"])]
    return by_target


def pick_run(runs, sla_ms):
    ok = [r for r in runs if r["ttft_p95_ms"] is not None and r["ttft_p95_ms"] <= sla_ms and r["errors"] == 0]
    if not ok:
        return None
    return max(ok, key=lambda r: r["agg_output_tok_per_s"] or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/targets.json")
    ap.add_argument("--results", default="results")
    ap.add_argument("--ttft-sla-ms", type=float, default=500)
    ap.add_argument("--out", default="results/REPORT.md")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    pricing = {t["name"]: t.get("pricing", {}) for t in cfg["targets"]}
    data = latest_summaries(args.results)
    if not data:
        raise SystemExit(f"no *_summary.json files in {args.results}")

    rows = []
    for name, d in data.items():
        runs = d["runs"]
        best = pick_run(runs, args.ttft_sla_ms)
        p = pricing.get(name, {})
        low = min(runs, key=lambda r: r["concurrency"])
        row = {
            "target": name,
            "hardware": d["target"].get("hardware", ""),
            "model": d["target"]["model"],
            "ttft_p50_ms@c=low": low["ttft_p50_ms"],
            "e2e_p50_s@c=low": low["e2e_p50_s"],
            "sla_concurrency": best["concurrency"] if best else "—",
            "tok_per_s@sla": best["agg_output_tok_per_s"] if best else "—",
            "usd_per_1M_out": "—",
            "usd_per_turn": "—",
        }
        ref = best or low
        turns = max(ref["success"], 1)
        avg_in = ref["total_input_tokens"] / turns
        avg_out = ref["total_output_tokens"] / turns

        if p.get("gpu_hourly_usd") and best and best["agg_output_tok_per_s"]:
            per_tok = p["gpu_hourly_usd"] / (best["agg_output_tok_per_s"] * 3600)
            row["usd_per_1M_out"] = round(per_tok * 1e6, 3)
            row["usd_per_turn"] = f"{per_tok * avg_out:.6f}"
        elif p.get("input_usd_per_1M") is not None and p.get("output_usd_per_1M") is not None:
            row["usd_per_1M_out"] = p["output_usd_per_1M"]
            row["usd_per_turn"] = f"{(avg_in * p['input_usd_per_1M'] + avg_out * p['output_usd_per_1M']) / 1e6:.6f}"
        rows.append(row)

    cols = list(rows[0].keys())
    lines = [
        "# Voice inference benchmark: results",
        "",
        f"Voice SLA: TTFT p95 ≤ {args.ttft_sla_ms:.0f} ms. Throughput and self-hosted cost are taken "
        "at the highest concurrency that meets it.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    lines += ["", "## Per-concurrency detail", ""]
    for name, d in data.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| concurrency | TTFT p50 ms | TTFT p95 ms | ITL p50 ms | E2E p95 s | out tok/s | errors |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in d["runs"]:
            lines.append(
                f"| {r['concurrency']} | {r['ttft_p50_ms']} | {r['ttft_p95_ms']} | {r['itl_p50_ms']} | "
                f"{r['e2e_p95_s']} | {r['agg_output_tok_per_s']} | {r['errors']} |"
            )
        lines.append("")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
