"""Compare sweep runs: a markdown table plus latency and cost charts."""

from __future__ import annotations

import csv
from pathlib import Path


def _load(run: Path) -> tuple[str, list[dict]]:
    with open(run / "summary.csv") as f:
        rows = [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(f)]
    return run.name.rsplit("-", 2)[0], rows


def _num(v: str):
    if v in ("True", "False"):
        return v == "True"
    try:
        return float(v)
    except ValueError:
        return v


def build_report(runs: list[str], out: str) -> None:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = [_load(Path(r)) for r in runs if (Path(r) / "summary.csv").exists()]
    if not data:
        raise SystemExit("No summary.csv found in the given run folders.")

    lines = ["# Voice inference benchmark report", "",
             "| Config | Max concurrency within SLA | p95 first audio @ that level (ms) "
             "| p50 TTFT (ms) | Cost / conversation-min (USD) |",
             "|---|---:|---:|---:|---:|"]
    for name, rows in data:
        ok = [r for r in rows if r["meets_sla"]]
        best = max(ok, key=lambda r: r["concurrency"]) if ok else None
        if best:
            lines.append(f"| {name} | {int(best['concurrency'])} | "
                         f"{best['time_to_first_audio_ms_p95']:.0f} | {best['llm_ttft_ms_p50']:.0f} | "
                         f"{best['cost_per_conversation_min_usd']:.4f} |")
        else:
            lines.append(f"| {name} | none | - | - | - |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; wrote table only.")
        print(out_dir / "report.md")
        return

    charts = [
        ("time_to_first_audio_ms_p95", "p95 time to first audio (ms)", "latency.png"),
        ("cost_per_conversation_min_usd", "Cost per conversation-minute (USD)", "cost.png"),
        ("output_tokens_per_s_total", "Total output tokens/s", "throughput.png"),
    ]
    for field, label, fname in charts:
        fig, ax = plt.subplots(figsize=(7, 4))
        for name, rows in data:
            ax.plot([r["concurrency"] for r in rows], [r[field] for r in rows], marker="o", label=name)
        if field.startswith("time_to_first_audio"):
            ax.axhline(y=data[0][1][0].get("sla_ms", 1000), linestyle="--", linewidth=1,
                       color="gray", label="SLA")
        ax.set_xscale("log")
        levels = sorted({int(r["concurrency"]) for _, rows in data for r in rows})
        ax.set_xticks(levels)
        ax.set_xticklabels([str(c) for c in levels])
        ax.minorticks_off()
        ax.set_xlabel("Concurrent conversations")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)
    print(f"Report written to {out_dir}")
