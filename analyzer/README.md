# Voice AI Inference Cost & Latency Analyzer

How fast does a self-hosted voice agent respond, how many callers can one GPU serve, and what
does each minute of conversation actually cost? This repo benchmarks the full voice-agent pipeline
on **AMD Instinct MI300X (ROCm)**, **NVIDIA H100** and **managed APIs** (Azure AI Foundry, OpenAI)
with one harness and one set of metrics.

```
caller audio ──▶ Whisper (STT) ──▶ Llama 3 / Qwen via vLLM ──▶ TTS ──▶ first audio back to caller
                  /audio/transcriptions   /chat/completions (stream)   /audio/speech
```

Every stage is called through an **OpenAI-compatible HTTP API**, so swapping a self-hosted vLLM
server for a managed endpoint is a config change, not a code change.

## What it measures

| Metric | Meaning |
|---|---|
| `time_to_first_audio_ms` | Silence the caller hears after they stop talking. The headline number. |
| `stt_ms` | Speech-to-text latency |
| `llm_ttft_ms` | LLM time to first token (prefill + queueing) |
| `decode_tps` | Output tokens/s per session after the first token |
| `tts_ttfb_ms` | TTS time to first audio byte |
| `output_tokens_per_s_total` | Aggregate LLM throughput across all sessions |
| `cost_per_conversation_min_usd` | Cost divided by minutes of audio exchanged (caller + agent) |
| `meets_sla` | p95 time-to-first-audio under the SLA (default 1,000 ms) and error rate ≤ 1% |

All latency metrics are reported as p50 and p95 for each concurrency level (default
1, 2, 4, 8, 16, 32, 64, 100 simultaneous conversations).

**Cost model.** Self-hosted: `GPU $/hour × run time ÷ conversation-minutes served`. Managed:
the sum of per-minute STT, per-token LLM and per-character TTS prices from the config.
The interesting number is the cost at the *highest concurrency that still meets the SLA*. That
is what one GPU really costs per conversation-minute in production.

## How the load is generated

Each simulated caller holds a multi-turn conversation (5 turns by default, last 3 exchanges kept
as context so prefix caching has something to reuse).

- **`realtime` pacing (default):** the caller "speaks" for the length of the clip, waits for the
  agent, then "listens" for the length of the reply before speaking again. This is how voice
  traffic loads a GPU.
- **`max` pacing:** no pauses. Finds the raw throughput ceiling. Runs are tagged `-maxload` and
  should not be compared with realtime cost figures.

**Streaming.** By default TTS starts as soon as the LLM produces its first complete sentence,
while generation continues. `--no-streaming` makes TTS wait for the whole reply, to show what
streaming saves.

## Experiments

| # | Experiment | Command |
|---|---|---|
| 1 | Baseline: Llama 3.1 8B BF16, streaming, prefix caching on | `scripts/serve_rocm.sh` then `python -m via.cli sweep --config configs/mi300x-llama3-8b.yaml` |
| 2 | Streaming off | `... --no-streaming` |
| 3 | Throughput ceiling | `... --pacing max` |
| 4 | Prefix caching off | `PREFIX_CACHE=0 scripts/serve_rocm.sh`, sweep with `--tag nocache` |
| 5 | FP8 weights + FP8 KV cache | `QUANT=fp8 KV_FP8=1 scripts/serve_rocm.sh`, sweep with `--tag fp8` |
| 6 | 70B FP8 on a single MI300X (192 GB) | `MODEL=meta-llama/Llama-3.1-70B-Instruct QUANT=fp8 scripts/serve_rocm.sh` + `configs/mi300x-llama3-70b-fp8.yaml` |
| 7 | Same as 1–5 on H100 | `GPU=cuda scripts/run_matrix.sh` |
| 8 | Managed APIs | `configs/azure-foundry.yaml`, `configs/openai.yaml` |

`scripts/run_matrix.sh` runs experiments 1–6 back to back and builds the report.

## Quick start

### 1. Dry run on a laptop (no GPU, no keys)

```bash
pip install -r requirements.txt
python -m via.mock_server --port 9000 &          # fake STT/LLM/TTS with simulated contention
python -m via.cli audio --config configs/mock.yaml # build 20 caller clips from data/prompts.txt
python -m via.cli sweep --config configs/mock.yaml
python -m via.cli report results/*/
```

The mock numbers are synthetic. Use the dry run to check the harness, never to publish.

### 2. AMD Developer Cloud (MI300X)

**Fastest path:** create a 1× MI300X droplet, open its console, and paste:

```bash
export HF_TOKEN=hf_...   # with Llama 3.1 access
curl -fsSL https://raw.githubusercontent.com/globaledgepathway/voice-inference-analyzer/main/scripts/bootstrap.sh | bash
```

It installs everything and runs `scripts/run_budget.sh` in the background (`tail -f ~/via-run.log`).

**Manual path:**

```bash
git clone https://github.com/globaledgepathway/voice-inference-analyzer && cd voice-inference-analyzer
pip install -r requirements.txt
export HF_TOKEN=...                                 # Llama weights are gated
scripts/serve_rocm.sh                               # Whisper :8001, vLLM :8000, Kokoro TTS :8880
python -m via.cli audio --config configs/mi300x-llama3-8b.yaml --voice am_michael
python -m via.cli sweep --config configs/mi300x-llama3-8b.yaml
```

Set `gpu_hourly_usd` in the config to the rate you are actually billed (list price for
1× MI300X is $2.59/h).

**On a small credit:** `scripts/run_budget.sh` runs a smoke test and then the experiments in
priority order. Before each block it compares spend so far (machine uptime × rate) plus the block's
estimate against `SPEND_LIMIT` (default $8.50), and skips blocks that would go over. Expected total
is about 2 GPU-hours. When it finishes, copy `results/` off the machine and **destroy** the droplet:
a powered-off droplet is still billed.

### 3. H100

Same steps with `scripts/serve_cuda.sh` and `configs/h100-llama3-8b.yaml`. Use the same models,
`max-model-len`, prompts and clips so the comparison is fair.

### 4. Managed APIs

Fill in endpoint, deployment names and prices in `configs/azure-foundry.yaml` or
`configs/openai.yaml`, export the keys, then sweep. Keep concurrency within your rate limits.

## Fairness rules

- Same models, prompts, caller clips, `max_tokens`, temperature and SLA on every platform.
- TTS runs on the **CPU** (Kokoro) on both GPU platforms by default, so the GPU comparison
  isolates STT + LLM. Moving TTS onto the GPU is a separate experiment (`TTS_IMAGE=...`).
- Record the stack versions. The serve scripts write `results/stack-*.txt` with vLLM, PyTorch
  and driver versions.
- Warm-up turn before every sweep. Same region / network distance for managed APIs where possible.
- Report p95, not just averages, and publish the raw `turns_c*.jsonl` alongside the charts.

## Output

```
results/<config>-<timestamp>/
  config.json        # config used (API keys redacted)
  turns_c<N>.jsonl   # one row per turn: every stage timing, tokens, transcript, reply
  summary.csv        # one row per concurrency level
results/report/
  report.md          # max SLA-compliant concurrency and cost per config
  latency.png  cost.png  throughput.png
```

## Repo layout

```
via/
  pipeline.py     one instrumented turn (STT -> streaming LLM -> TTS)
  loadgen.py      concurrent multi-turn callers, realtime or max pacing
  clients.py      OpenAI-compatible STT / chat (SSE) / speech clients
  metrics.py      percentiles, SLA check, cost per conversation-minute
  report.py       cross-run table and charts
  mock_server.py  fake endpoints for dry runs
  cli.py          `python -m via.cli {audio,sweep,report}`
configs/          one YAML per platform/model
scripts/          serve_rocm.sh, serve_cuda.sh, run_matrix.sh
data/prompts.txt  caller utterances (customer-support scenarios)
docker/Dockerfile benchmark client image
```

## Known caveats (check before your first real run)

- **Whisper on vLLM:** the serve scripts use vLLM's `/v1/audio/transcriptions` endpoint and
  install `librosa`/`soundfile` at start. Confirm your vLLM image version supports Whisper on
  ROCm. If not, run Whisper behind any OpenAI-compatible transcription server and point
  `stt.base_url` at it.
- **Kokoro-FastAPI** image names and the `/v1/audio/speech` voice list may change; check the
  project's README.
- **Agent speech length** is estimated from word count at 150 words/min, not decoded from audio.
- Prices in configs are placeholders. Fill in real rates before quoting any cost.

## License

MIT
