"""Command line entry point.

    python -m via.cli sweep  --config configs/mi300x-llama3-8b.yaml
    python -m via.cli sweep  --config configs/mock.yaml --levels 1 4 16
    python -m via.cli audio  --config configs/mock.yaml        # build test clips
    python -m via.cli report results/*/                         # compare runs
"""

from __future__ import annotations

import argparse
import asyncio

from .config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(prog="via", description="Voice AI inference cost & latency analyzer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sweep", help="run a concurrency sweep against one config")
    s.add_argument("--config", required=True)
    s.add_argument("--audio-dir", default="data/audio")
    s.add_argument("--out", default="results")
    s.add_argument("--levels", type=int, nargs="*", help="override concurrency levels")
    s.add_argument("--turns", type=int, help="override turns per session")
    s.add_argument("--pacing", choices=["realtime", "max"])
    s.add_argument("--no-streaming", action="store_true", help="TTS waits for the full LLM reply")
    s.add_argument("--tag", help="suffix for the run name, e.g. fp8 or nocache")

    a = sub.add_parser("audio", help="synthesize test caller clips from data/prompts.txt")
    a.add_argument("--config", required=True)
    a.add_argument("--prompts", default="data/prompts.txt")
    a.add_argument("--audio-dir", default="data/audio")
    a.add_argument("--voice", help="caller voice (use a different voice than the agent)")

    r = sub.add_parser("report", help="compare one or more result folders")
    r.add_argument("runs", nargs="+")
    r.add_argument("--out", default="results/report")

    args = ap.parse_args()

    if args.cmd == "sweep":
        cfg = load_config(args.config)
        if args.turns:
            cfg.load["turns_per_session"] = args.turns
        if args.pacing:
            cfg.load["pacing"] = args.pacing
        if cfg.load["pacing"] == "max":
            cfg.name += "-maxload"  # throughput ceiling, not comparable to realtime cost
        if args.no_streaming:
            cfg.load["streaming"] = False
            cfg.name += "-nostream"
        if args.tag:
            cfg.name += f"-{args.tag}"
        from .loadgen import sweep
        asyncio.run(sweep(cfg, args.audio_dir, args.out, args.levels))
    elif args.cmd == "audio":
        from .audio import make_clips
        asyncio.run(make_clips(load_config(args.config), args.prompts, args.audio_dir, args.voice))
    elif args.cmd == "report":
        from .report import build_report
        build_report(args.runs, args.out)


if __name__ == "__main__":
    main()
