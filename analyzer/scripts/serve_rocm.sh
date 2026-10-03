#!/usr/bin/env bash
# Start the full voice stack on one AMD Instinct GPU (MI300X) with ROCm.
#   Whisper (vLLM)  -> :8001     LLM (vLLM) -> :8000     TTS (Kokoro, CPU) -> :8880
#
# Knobs (env vars):
#   MODEL=meta-llama/Llama-3.1-8B-Instruct   STT_MODEL=openai/whisper-large-v3-turbo
#   QUANT=fp8              -> on-the-fly FP8 weights      KV_FP8=1 -> FP8 KV cache
#   PREFIX_CACHE=0         -> disable prefix caching (on by default in vLLM V1)
#   LLM_MEM=0.75 STT_MEM=0.10   fraction of GPU memory per server
#   VLLM_IMAGE=rocm/vllm:latest  TTS_IMAGE=ghcr.io/remsky/kokoro-fastapi-cpu:latest
#   EXTRA_ARGS="..."       -> passed through to `vllm serve` for the LLM
set -euo pipefail

MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
STT_MODEL=${STT_MODEL:-openai/whisper-large-v3-turbo}
VLLM_IMAGE=${VLLM_IMAGE:-rocm/vllm:latest}
TTS_IMAGE=${TTS_IMAGE:-ghcr.io/remsky/kokoro-fastapi-cpu:latest}
LLM_MEM=${LLM_MEM:-0.75}
STT_MEM=${STT_MEM:-0.10}
: "${HF_TOKEN:?export HF_TOKEN=... (Llama weights are gated on Hugging Face)}"

LLM_ARGS="--port 8000 --gpu-memory-utilization ${LLM_MEM} --max-model-len 8192"
[[ "${QUANT:-}" == "fp8" ]] && LLM_ARGS+=" --quantization fp8"
[[ "${KV_FP8:-0}" == "1" ]] && LLM_ARGS+=" --kv-cache-dtype fp8"
[[ "${PREFIX_CACHE:-1}" == "0" ]] && LLM_ARGS+=" --no-enable-prefix-caching"
LLM_ARGS+=" ${EXTRA_ARGS:-}"

ROCM_FLAGS=(--network host --ipc host --shm-size 16g
  --device /dev/kfd --device /dev/dri --group-add video
  --security-opt seccomp=unconfined
  -e HF_TOKEN -v "${HOME}/.cache/huggingface:/root/.cache/huggingface")

docker rm -f via-stt via-llm via-tts >/dev/null 2>&1 || true

echo "Starting Whisper (${STT_MODEL}) on :8001"
docker run -d --name via-stt "${ROCM_FLAGS[@]}" "${VLLM_IMAGE}" bash -c \
  "pip install -q librosa soundfile && vllm serve ${STT_MODEL} --port 8001 --gpu-memory-utilization ${STT_MEM}"

# Start the LLM after Whisper has claimed its memory.
until curl -sf localhost:8001/v1/models >/dev/null; do sleep 5; done

echo "Starting LLM (${MODEL}) on :8000 with: ${LLM_ARGS}"
docker run -d --name via-llm "${ROCM_FLAGS[@]}" "${VLLM_IMAGE}" vllm serve "${MODEL}" ${LLM_ARGS}

echo "Starting TTS on :8880"
docker run -d --name via-tts -p 8880:8880 "${TTS_IMAGE}"

until curl -sf localhost:8000/v1/models >/dev/null; do sleep 5; done
until curl -sf localhost:8880/v1/models >/dev/null || curl -sf localhost:8880/health >/dev/null; do sleep 3; done
echo "All three servers are up. Record the stack versions:"
docker exec via-llm bash -c "python -c 'import vllm,torch;print(\"vllm\",vllm.__version__,\"torch\",torch.__version__)'; rocm-smi --showproductname | head -20" \
  | tee "results/stack-$(date +%Y%m%d-%H%M%S).txt" 2>/dev/null || true
