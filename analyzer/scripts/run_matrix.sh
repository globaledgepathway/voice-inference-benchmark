#!/usr/bin/env bash
# The full MI300X experiment matrix. Run on the AMD Developer Cloud box, from the repo root.
# Each block restarts the servers with different settings, then sweeps concurrency.
# Budget: roughly 25-40 min per block with the default 8 concurrency levels x 5 turns.
set -euo pipefail
GPU=${GPU:-rocm}            # rocm | cuda
SERVE="scripts/serve_${GPU}.sh"
CFG8=configs/mi300x-llama3-8b.yaml
[[ "$GPU" == "cuda" ]] && CFG8=configs/h100-llama3-8b.yaml

# 1. Baseline: 8B, BF16, streaming, prefix caching on
$SERVE
python -m via.cli sweep --config "$CFG8"

# 2. Streaming off (TTS waits for the whole reply). Same servers.
python -m via.cli sweep --config "$CFG8" --no-streaming

# 3. Raw throughput ceiling (no conversational pauses)
python -m via.cli sweep --config "$CFG8" --pacing max

# 4. Prefix caching off
PREFIX_CACHE=0 $SERVE
python -m via.cli sweep --config "$CFG8" --tag nocache --levels 16 32 64 100

# 5. FP8 weights + FP8 KV cache, 8B
QUANT=fp8 KV_FP8=1 $SERVE
python -m via.cli sweep --config "$CFG8" --tag fp8

# 6. 70B FP8 on a single GPU (MI300X only: 192 GB)
if [[ "$GPU" == "rocm" ]]; then
  MODEL=RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8 LLM_MEM=0.80 STT_MEM=0.08 $SERVE
  python -m via.cli sweep --config configs/mi300x-llama3-70b-fp8.yaml
fi

python -m via.cli report results/*/
