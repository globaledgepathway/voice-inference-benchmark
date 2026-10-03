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
bench/voice_turn_bench.py full voice loop: Whisper STT -> streaming LLM -> TTS, time to first audio
bench/tts_server.py       minimal Kokoro-82M TTS server (one GPU worker per process)
bench/noisy_neighbor.py   two tenants on one GPU: voice latency alone vs with a batch neighbour, chargeback
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

## Shared GPUs: the noisy-neighbour test

Platform teams rarely give one model a whole GPU. `bench/noisy_neighbor.py` measures what sharing costs a
latency-sensitive tenant, with no scheduler to install:

| Tenant | Workload | Cares about |
|---|---|---|
| A, voice | short voice turns at fixed concurrency | TTFT p95 under the voice SLA |
| B, batch | long prompts (~1,500 words in, 512 out) kept in flight | throughput |

Phases, same tenant-A load each time: **alone** (baseline), **shared** (both at equal priority), and
**priority** (voice requests sent with higher priority; start vLLM with `--scheduling-policy priority`).
For the shared phases it splits the GPU cost of the window between tenants by processed tokens: a simple
chargeback model.

```bash
# on the GPU box, vLLM started with: --scheduling-policy priority
python3 bench/noisy_neighbor.py --target h100_vllm --voice-conc 8 --batch-conc 32 --priority
```

What it shows: how far voice TTFT p95 moves when a batch job lands on the same GPU, whether request
priority restores it, and what each tenant would be billed. Those are the questions queue-based
schedulers (Kubernetes GPU operator with MIG/time-slicing, Run:ai quotas and preemption, Ray Serve
replicas) exist to answer.

## Method notes

- Same model, same prompts, same `max_tokens` (150 ≈ 20 s of speech) on every target.
- 3 warm-up requests before measuring, so cold-start and graph capture don't skew TTFT.
- Each concurrency level runs `--requests` calls; failures are counted, not hidden.
- Token counts come from the server's `usage` field when available, otherwise streamed chunks.
- GPU cost assumes the GPU is fully utilized at the SLA concurrency. Real utilization is lower, so treat it as a floor.

## Results

### AMD MI300X (AMD Developer Cloud, vLLM 0.27.1, ROCm 7.2), 2026-10-03

One MI300X, $1.99/hr. Client ran on the same droplet, so **no network time** is included.
Raw summaries are in `results/`.

**LLM only, Llama-3.1-8B-Instruct, voice prompts, 150 max tokens** (`llm_bench.py`)

| Concurrent calls | TTFT p50 | TTFT p95 | ITL p50 | Output tok/s |
|---|---|---|---|---|
| 1 | 14 ms | 15 ms | 4.9 ms | 195 |
| 8 | 21 ms | 25 ms | 6.8 ms | 1,051 |
| 32 | 27 ms | 54 ms | 8.5 ms | 2,664 |
| 64 | 70 ms | 92 ms | 11.1 ms | 3,621 |
| 128 | 59 ms | 222 ms | 12.7 ms | 6,518 |
| 256 | 192 ms | 418 ms | 16.2 ms | 7,026 |

Voice SLA (TTFT p95 ≤ 500 ms) still met at 256 concurrent calls: about **$0.08 per 1M output tokens**
at full utilisation. (A first c=16 run showed a one-off 933 ms p95 spike; the re-run gave 32 ms.)

**Full voice loop: Whisper-large-v3-turbo → Llama-3.1-8B → Kokoro-82M, all on the one GPU**
(`voice_turn_bench.py`, LibriSpeech clips; memory split Whisper 20% / Llama 60%; one TTS worker)

| Callers | STT p50 | LLM TTFT p50 | First sentence p50 | TTS p50 | **First audio p50 / p95** |
|---|---|---|---|---|---|
| 1 | 111 ms | 12 ms | 129 ms | 221 ms | **463 / 598 ms** |
| 4 | 123 ms | 14 ms | 138 ms | 626 ms | 923 / 1,008 ms |
| 8 | 118 ms | 13 ms | 153 ms | 1,588 ms | 1,884 / 2,160 ms |
| 16 | 124 ms | 14 ms | 174 ms | 3,414 ms | 3,727 / 3,958 ms |
| 32 | 401 ms | 19 ms | 207 ms | 6,653 ms | 7,066 / 7,670 ms |

Findings:
- A single caller hears the agent start speaking ~0.46 s after their audio arrives.
- STT and LLM barely move with load; the single, serialised TTS worker is the queue. More TTS replicas
  (or batching) is the capacity lever, not more GPU.
- ROCm gotcha: with MIOpen on, Kokoro took ~5.5 s per sentence because MIOpen re-tunes convolution kernels
  for every new input length (~1.5 s each). Disabling it (`torch.backends.cudnn.enabled=False`) gave a
  steady 150–220 ms; a fully warm MIOpen cache reaches 50–75 ms.
- Starting 8 TTS processes at once left 2 on CPU (silent fallback), which starved the box; start replicas
  one at a time and check the device.

**Synthetic throughput reference:** `vllm bench serve`, random 512 in / 128 out, 200 prompts all at once:
18,071 total tok/s, TTFT p50 2.5 s (burst queueing; not a voice-shaped load). See
`results/vllm_bench_serve_random_512in_128out.txt`.

### Still to run
- NVIDIA H100: same three models and scripts, plus `noisy_neighbor.py`
- Network-inclusive run (client in Azure, servers reached over the internet)
- Azure AI Foundry with Llama-3.1-8B (same-model managed comparison) and managed TTS for the full loop

## Roadmap

- [x] ASR (Whisper) and TTS stages, for full voice-turn latency: speech in → speech out (`bench/voice_turn_bench.py`)
- [x] Noisy-neighbour / multi-tenant test with chargeback (`bench/noisy_neighbor.py`)
- [ ] Larger model (70B) where MI300X's 192 GB HBM fits on one GPU and H100 needs two
- [ ] FP8 quantization comparison
- [ ] Charts for TTFT vs concurrency and $/turn
