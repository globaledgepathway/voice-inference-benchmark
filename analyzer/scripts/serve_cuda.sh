#!/usr/bin/env bash
# Same stack as serve_rocm.sh, on one NVIDIA GPU (H100). Same knobs.
set -euo pipefail

MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
STT_MODEL=${STT_MODEL:-openai/whisper-large-v3-turbo}
VLLM_IMAGE=${VLLM_IMAGE:-vllm/vllm-openai:latest}
TTS_IMAGE=${TTS_IMAGE:-ghcr.io/remsky/kokoro-fastapi-cpu:latest}
LLM_MEM=${LLM_MEM:-0.75}
STT_MEM=${STT_MEM:-0.10}
: "${HF_TOKEN:?export HF_TOKEN=... (Llama weights are gated on Hugging Face)}"

LLM_ARGS="--port 8000 --gpu-memory-utilization ${LLM_MEM} --max-model-len 8192"
[[ "${QUANT:-}" == "fp8" ]] && LLM_ARGS+=" --quantization fp8"
[[ "${KV_FP8:-0}" == "1" ]] && LLM_ARGS+=" --kv-cache-dtype fp8"
[[ "${PREFIX_CACHE:-1}" == "0" ]] && LLM_ARGS+=" --no-enable-prefix-caching"
LLM_ARGS+=" ${EXTRA_ARGS:-}"

CUDA_FLAGS=(--gpus all --network host --ipc host
  -e HF_TOKEN -v "${HOME}/.cache/huggingface:/root/.cache/huggingface")

docker rm -f via-stt via-llm via-tts >/dev/null 2>&1 || true

docker run -d --name via-stt "${CUDA_FLAGS[@]}" --entrypoint bash "${VLLM_IMAGE}" -c \
  "pip install -q librosa soundfile && vllm serve ${STT_MODEL} --port 8001 --gpu-memory-utilization ${STT_MEM}"
until curl -sf localhost:8001/v1/models >/dev/null; do sleep 5; done

docker run -d --name via-llm "${CUDA_FLAGS[@]}" "${VLLM_IMAGE}" --model "${MODEL}" ${LLM_ARGS}
docker run -d --name via-tts -p 8880:8880 "${TTS_IMAGE}"

until curl -sf localhost:8000/v1/models >/dev/null; do sleep 5; done
until curl -sf localhost:8880/v1/models >/dev/null || curl -sf localhost:8880/health >/dev/null; do sleep 3; done
echo "All three servers are up. Record the stack versions:"
docker exec via-llm bash -c "python3 -c 'import vllm,torch;print(\"vllm\",vllm.__version__,\"torch\",torch.__version__)'; nvidia-smi --query-gpu=name,driver_version --format=csv" \
  | tee "results/stack-$(date +%Y%m%d-%H%M%S).txt" 2>/dev/null || true
