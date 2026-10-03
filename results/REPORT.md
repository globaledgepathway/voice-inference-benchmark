# Voice inference benchmark: results

Voice SLA: TTFT p95 ≤ 500 ms. Throughput and self-hosted cost are taken at the highest concurrency that meets it.

| target | hardware | model | ttft_p50_ms@c=low | e2e_p50_s@c=low | sla_concurrency | tok_per_s@sla | usd_per_1M_out | usd_per_turn |
|---|---|---|---|---|---|---|---|---|
| mi300x_vllm | 1x AMD Instinct MI300X (AMD Developer Cloud), ROCm + vLLM | meta-llama/Llama-3.1-8B-Instruct | 14.4 | 0.169 | 256 | 7025.9 | 0.079 | 0.000003 |

## Per-concurrency detail

### mi300x_vllm

| concurrency | TTFT p50 ms | TTFT p95 ms | ITL p50 ms | E2E p95 s | out tok/s | errors |
|---|---|---|---|---|---|---|
| 1 | 14.4 | 15.1 | 4.86 | 0.252 | 194.7 | 0 |
| 4 | 19.9 | 21.8 | 6.44 | 0.335 | 561.3 | 0 |
| 8 | 21.1 | 24.8 | 6.75 | 0.319 | 1050.9 | 0 |
| 16 | 21.9 | 31.7 | 7.28 | 0.383 | 1979.3 | 0 |
| 32 | 26.7 | 53.7 | 8.47 | 0.429 | 2664.3 | 0 |
| 64 | 70.0 | 91.6 | 11.14 | 0.528 | 3620.9 | 0 |
| 128 | 59.1 | 222.0 | 12.73 | 0.713 | 6517.8 | 0 |
| 256 | 191.8 | 417.9 | 16.19 | 1.011 | 7025.9 | 0 |

