#!/usr/bin/env bash
# Run the same sweep against every target, then build the report.
set -euo pipefail
cd "$(dirname "$0")/.."
CONC="${CONC:-1 4 8 16 32 64}"
REQ="${REQ:-64}"
for t in ${TARGETS:-mi300x_vllm h100_vllm azure_foundry}; do
  python3 bench/llm_bench.py --target "$t" --concurrency $CONC --requests "$REQ"
done
python3 bench/report.py --ttft-sla-ms "${SLA_MS:-500}"
