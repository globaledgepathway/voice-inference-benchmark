#!/usr/bin/env bash
# Start vLLM on an AMD MI300X (ROCm) host. Run on the GPU box.
# Requires: docker with ROCm devices, HF_TOKEN for gated Llama weights.
set -euo pipefail
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
IMAGE="${IMAGE:-rocm/vllm:latest}"
docker run --rm -it --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined \
  -e HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  "$IMAGE" \
  vllm serve "$MODEL" --host 0.0.0.0 --port 8000 \
    --max-model-len 8192 --gpu-memory-utilization 0.9 \
    ${VLLM_API_KEY:+--api-key "$VLLM_API_KEY"}
