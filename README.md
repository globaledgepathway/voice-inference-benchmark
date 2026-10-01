# voice-inference-benchmark

What does it cost, and how fast is it, to run the LLM brain of a real-time voice agent?
This repo answers that for one open model on three kinds of infrastructure:

| Target | What it is |
|---|---|
| `mi300x_vllm` | AMD Instinct MI300X on AMD Developer Cloud, ROCm + vLLM |
| `h100_vllm` | NVIDIA H100 80GB, CUDA + vLLM |
| `azure_foundry` | Azure AI Foundry managed (serverless) endpoint, pay per token |

All three expose the same OpenAI-compatible `/chat/completions` API, so one harness drives them all with the same prompts.

## Why voice is different

In a voice call, the user hears silence until the first word is spoken. So the number that matters most is **time to first token (TTFT)**, not raw throughput. The harness reports:

- **TTFT p50/p95**: how long the caller waits before the agent starts talking
- **Inter-token latency (ITL)**: has to stay well under speech rate, or the TTS starves
- **End-to-end latency** per reply
- **Aggregate output tokens/sec** across concurrent calls
- **Cost per 1M output tokens** and **cost per voice turn**

Self-hosted cost is measured at the **highest concurrency that still meets the voice SLA** (default: TTFT p95 ≤ 500 ms). That's the number of simultaneous calls one GPU can really serve.

## Layout

```
bench/llm_bench.py        streaming benchmark client, concurrency sweep (stdlib only)
bench/report.py           builds results/REPORT.md: latency + cost comparison
configs/targets.json      endpoints, model names, pricing (fill in)
prompts/voice_turns.jsonl 12 realistic phone-call turns (booking, billing, sales objections…)
scripts/serve_vllm_mi300x.sh   start vLLM on MI300X (ROCm docker image)
scripts/serve_vllm_h100.sh     start vLLM on H100 (CUDA docker image)
scripts/run_all.sh        sweep every target, then build the report
scripts/mock_server.py    fake streaming server for a dry run without a GPU
```

## Quick start

No Python dependencies; any Python 3.9+ works.

**1. Dry run locally (no GPU):**

```bash
python3 scripts/mock_server.py --port 8000 &
# set mi300x_vllm.base_url to http://localhost:8000/v1 in configs/targets.json
python3 bench/llm_bench.py --target mi300x_vllm --concurrency 1 8 --requests 16
```

**2. Serve the model on each GPU box:**

```bash
export HF_TOKEN=...            # Llama 3.1 weights are gated on Hugging Face
./scripts/serve_vllm_mi300x.sh # on the MI300X droplet
./scripts/serve_vllm_h100.sh   # on the H100 box
```

**3. Deploy the same model in Azure AI Foundry** as a serverless endpoint, then copy its endpoint URL and key.

**4. Fill in `configs/targets.json`:** `base_url`, `model`, and pricing (`gpu_hourly_usd` for the GPUs you rent; input/output $/1M tokens from the Azure price list). Keys come from env vars:

```bash
export AZURE_AI_KEY=...
export VLLM_API_KEY=...        # only if you started vLLM with --api-key
```

**5. Run everything and build the report:**

```bash
CONC="1 4 8 16 32 64" REQ=64 SLA_MS=500 ./scripts/run_all.sh
cat results/REPORT.md
```

## Method notes

- Same model, same prompts, same `max_tokens` (150 ≈ 20 s of speech) on every target.
- 3 warm-up requests before measuring, so cold-start and graph capture don't skew TTFT.
- Each concurrency level runs `--requests` calls; failures are counted, not hidden.
- Token counts come from the server's `usage` field when available, otherwise streamed chunks.
- GPU cost assumes the GPU is fully utilized at the SLA concurrency. Real utilization is lower, so treat it as a floor.

## Results

Run it and the table lands in `results/REPORT.md`. *(Numbers to be added after the MI300X / H100 / Azure runs.)*

## Roadmap

- [ ] ASR (Whisper) and TTS stages, for full voice-turn latency: speech in → speech out
- [ ] Larger model (70B) where MI300X's 192 GB HBM fits on one GPU and H100 needs two
- [ ] FP8 quantization comparison
- [ ] Charts for TTFT vs concurrency and $/turn
