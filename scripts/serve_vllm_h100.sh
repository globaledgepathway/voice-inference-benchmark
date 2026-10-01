#!/usr/bin/env bash
# Start vLLM on an NVIDIA H100 (CUDA) host. Run on the GPU box.
# Requires: docker + NVIDIA Container Toolkit, HF_TOKEN for gated Llama weights.
set -euo pipefail
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
IMAGE="${IMAGE:-vllm/vllm-openai:latest}"
docker run --rm -it --gpus all --network=host --ipc=host \
  -e HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  "$IMAGE" \
  --model "$MODEL" --host 0.0.0.0 --port 8000 \
  --max-model-len 8192 --gpu-memory-utilization 0.9 \
  ${VLLM_API_KEY:+--api-key "$VLLM_API_KEY"}
