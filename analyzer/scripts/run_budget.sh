#!/usr/bin/env bash
# The MI300X experiments sized for a small credit (default $10 at $2.59/h ≈ 3.8 GPU-hours).
# Spend is estimated from machine uptime (≈ droplet uptime) × hourly rate. Before each block the
# script checks that the block's estimated cost still fits under SPEND_LIMIT, and skips it if not.
#
#   SPEND_LIMIT=8.50 RATE=2.59 scripts/run_budget.sh
#
# Afterwards: copy results/ off the machine, then DESTROY the droplet. A powered-off droplet
# is still billed.
set -uo pipefail
RATE=${RATE:-2.59}
SPEND_LIMIT=${SPEND_LIMIT:-8.50}   # leave ~$1.50 headroom for setup and copying results
CFG=configs/mi300x-llama3-8b.yaml
LEVELS="1 4 16 32 64 100"          # 6 levels instead of 8 to save time
LOG=results/budget.log
mkdir -p results

spent() {  # dollars since boot
  local up; up=$(cut -d. -f1 /proc/uptime)
  python3 -c "print(round($up/3600*$RATE, 2))"
}

block() {  # block <name> <est_minutes> <command...>
  local name=$1 mins=$2; shift 2
  local now projected
  now=$(spent)
  projected=$(python3 -c "print(round($now + $mins/60*$RATE, 2))")
  if python3 -c "import sys; sys.exit(0 if $projected <= $SPEND_LIMIT else 1)"; then
    echo "[$(date +%T)] \$$now spent. START $name (~${mins} min, projected \$$projected)" | tee -a "$LOG"
    "$@" || echo "[$(date +%T)] $name FAILED, continuing" | tee -a "$LOG"
  else
    echo "[$(date +%T)] \$$now spent. SKIP $name: projected \$$projected > limit \$$SPEND_LIMIT" | tee -a "$LOG"
  fi
}

sweep() { python -m via.cli sweep --config "$@"; }
# Restart + sweep stay in one block so a skipped restart can never mislabel a sweep.
fp8()     { QUANT=fp8 KV_FP8=1 scripts/serve_rocm.sh && sweep "$CFG" --levels $LEVELS --tag fp8; }
nocache() { PREFIX_CACHE=0 scripts/serve_rocm.sh && sweep "$CFG" --levels 16 32 64 100 --tag nocache; }
big70b()  { MODEL=RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8 LLM_MEM=0.80 STT_MEM=0.08 scripts/serve_rocm.sh \
              && sweep configs/mi300x-llama3-70b-fp8.yaml --levels $LEVELS; }

# 0. Smoke test: one caller, one turn. Catches broken servers before spending on sweeps.
scripts/serve_rocm.sh
[[ -z "$(ls data/audio/*.wav 2>/dev/null)" ]] && python -m via.cli audio --config "$CFG" --voice am_michael
python -m via.cli sweep --config "$CFG" --levels 1 --turns 1 --tag smoke || {
  echo "Smoke test failed. Fix before continuing (check: docker logs via-stt / via-llm / via-tts)."; exit 1; }

# Ordered by value: the most important results run first.
block baseline      12 sweep "$CFG" --levels $LEVELS
block no-streaming  10 sweep "$CFG" --levels $LEVELS --no-streaming
block max-load       6 sweep "$CFG" --levels 16 32 64 100 --pacing max --turns 3
block fp8          16 fp8
block nocache      11 nocache
# 70B: ~70 GB download + load. Most expensive block, so it runs last.
block 70b-fp8      42 big70b

rm -rf results/*-smoke-*
python -m via.cli report results/*/
echo "[$(date +%T)] Done. Total ≈ \$$(spent). Copy results/ off this machine, then destroy the droplet." | tee -a "$LOG"
