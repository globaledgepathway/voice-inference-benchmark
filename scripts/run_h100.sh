#!/usr/bin/env bash
# One-shot H100 run: the same benchmarks as the MI300X, on one NVIDIA H100 80GB.
# Needs: docker + NVIDIA container toolkit, internet, ~60 GB free disk. About 45-60 minutes.
#
#   bash scripts/run_h100.sh            (asks for the HF token at a hidden prompt)
#   H100_PRICE=2.49 bash scripts/run_h100.sh     (your $/hr, used for cost per token)
#
# Phase 1  Llama-3.1-8B alone on the GPU: llm_bench sweep 1-256 callers, noisy_neighbor, vllm bench serve
# Phase 2  Whisper + Llama + Kokoro on one GPU: voice_turn_bench (speech in -> first audio out)
# Results land in results/ with an h100 prefix, then results/REPORT.md is rebuilt.
set -uo pipefail
cd "$(dirname "$0")/.."
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
ASR="openai/whisper-large-v3-turbo"
PRICE="${H100_PRICE:-0}"

if [ -z "${HF_TOKEN:-}" ]; then
  read -rs -p "Hugging Face token (hidden, needs Llama 3.1 access): " HF_TOKEN; echo
  HF_TOKEN=$(printf '%s' "$HF_TOKEN" | tr -d '[:space:]'); export HF_TOKEN
fi
code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/$MODEL/resolve/main/config.json)
[ "$code" = 200 ] || { echo "HF token check failed (HTTP $code): 401 = bad token, 403 = accept the Llama 3.1 license"; exit 1; }

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee results_gpu.txt
python3 - "$PRICE" <<'PY'
import json, sys
c = json.load(open("configs/targets.json"))
for t in c["targets"]:
    if t["name"] == "h100_vllm":
        t["base_url"] = "http://localhost:8000/v1"
        t["pricing"]["gpu_hourly_usd"] = float(sys.argv[1])
json.dump(c, open("configs/targets.json", "w"), indent=2)
PY

docker rm -f bench >/dev/null 2>&1
docker run -d --name bench --gpus all --network host --ipc host -e HF_TOKEN \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -v "$PWD:/vib" -w /vib \
  --entrypoint sleep "$IMAGE" infinity >/dev/null || exit 1
X() { docker exec -e HF_TOKEN bench bash -c "$*"; }
X 'python3 -c "import vllm; print(\"vLLM\", vllm.__version__)"' | tee -a results_gpu.txt

wait_up() {  # port, process pattern ([x]yz form so pgrep never matches its own shell)
  for _ in $(seq 1 120); do
    X "curl -sf localhost:$1/v1/models >/dev/null" && return 0
    X "pgrep -f '$2' >/dev/null" || { echo "server on :$1 died; log:"; X "tail -30 /tmp/srv_$1.log"; return 1; }
    sleep 10
  done; echo "timeout waiting for :$1"; return 1
}
stop_all() { X "pkill -f 'vllm serve'; pkill -f tts_server" ; sleep 15; }

echo "== Phase 1: Llama-3.1-8B alone (priority scheduling on, so noisy_neighbor can use it)"
X "nohup vllm serve $MODEL --port 8000 --gpu-memory-utilization 0.9 --max-model-len 8192 \
   --scheduling-policy priority > /tmp/srv_8000.log 2>&1 &"
wait_up 8000 "[v]llm serve" || exit 1
X "python3 bench/llm_bench.py --target h100_vllm --concurrency 1 4 8 16 32 64 128 256 --requests 256" \
  > results/h100_llm_bench.log 2>&1
X "python3 bench/noisy_neighbor.py --target h100_vllm --voice-conc 8 --batch-conc 32 --priority" \
  2>&1 | tee results/h100_noisy_neighbor.log | grep -E '^\||phase|saved'
X "vllm bench serve --model $MODEL --port 8000 --dataset-name random --random-input-len 512 \
   --random-output-len 128 --num-prompts 200" > results/h100_vllm_bench_serve_random_512in_128out.txt 2>&1
grep -E 'Successful|Output token throughput|Total token throughput|Mean TTFT|Median TTFT' \
  results/h100_vllm_bench_serve_random_512in_128out.txt

echo "== Phase 2: Whisper + Llama + Kokoro on one GPU"
stop_all
X "pip install -q kokoro soundfile >/tmp/pip.log 2>&1; (apt-get update -qq && apt-get install -y -qq espeak-ng) >/dev/null 2>&1; true"
X "nohup vllm serve $ASR --port 8001 --gpu-memory-utilization 0.2 > /tmp/srv_8001.log 2>&1 &"
wait_up 8001 "[w]hisper-large" || exit 1
X "nohup vllm serve $MODEL --port 8000 --gpu-memory-utilization 0.6 --max-model-len 8192 > /tmp/srv_8000.log 2>&1 &"
wait_up 8000 "[L]lama-3.1" || exit 1
# TTS_MIOPEN=1 keeps cuDNN on: the MIOpen re-tuning issue is ROCm-only
X "TTS_MIOPEN=1 TTS_PORT=8002 nohup python3 bench/tts_server.py > /tmp/tts.log 2>&1 &"
for _ in $(seq 1 60); do X "grep -q 'TTS ready' /tmp/tts.log" && break; sleep 5; done
X "grep 'TTS ready' /tmp/tts.log" || { echo "TTS failed:"; X "tail -20 /tmp/tts.log"; exit 1; }
X "python3 bench/voice_turn_bench.py --concurrency 1 4 8 16 32 --out results/h100_voice_full_loop_summary.json" \
  2>&1 | grep -vE 'WARNING|Warning' | tail -8 | tee results/h100_voice_full_loop.log

echo "== Report"
X "python3 bench/report.py --ttft-sla-ms 500" | sed -n '1,8p'
stop_all; docker rm -f bench >/dev/null
mv results_gpu.txt results/h100_gpu_info.txt
tar czf ~/h100_results.tgz results configs/targets.json && echo "Saved ~/h100_results.tgz. Stop or destroy the GPU now."
