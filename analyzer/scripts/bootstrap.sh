#!/usr/bin/env bash
# One-command setup + unattended run on a fresh AMD Developer Cloud MI300X droplet.
#
#   export HF_TOKEN=hf_...            # Hugging Face token with Llama 3.1 access
#   curl -fsSL https://raw.githubusercontent.com/globaledgepathway/voice-inference-analyzer/main/scripts/bootstrap.sh | bash
#
# Runs in the background (survives closing the console). Follow progress with:
#   tail -f ~/via-run.log
# Optional: export GITHUB_TOKEN=... (repo scope) to push results to a `results` branch at the end.
set -euo pipefail
: "${HF_TOKEN:?Run: export HF_TOKEN=hf_...  (and accept the Llama 3.1 license on huggingface.co)}"
REPO=${REPO:-https://github.com/globaledgepathway/voice-inference-analyzer}
DIR=~/voice-inference-analyzer

echo "== Checking GPU and Docker"
command -v rocm-smi >/dev/null && rocm-smi --showproductname | grep -i -m1 mi300 || echo "warning: MI300X not detected by rocm-smi"
if ! command -v docker >/dev/null; then curl -fsSL https://get.docker.com | sh; fi

echo "== Getting the code"
if [[ -d $DIR/.git ]]; then git -C "$DIR" pull -q; else git clone -q "$REPO" "$DIR"; fi
cd "$DIR"
chmod +x scripts/*.sh   # web uploads drop the executable bit

echo "== Python environment"
if ! python3 -m venv .venv 2>/dev/null; then
  apt-get update -qq && apt-get install -y -qq python3-venv >/dev/null && python3 -m venv .venv
fi
.venv/bin/pip install -q -r requirements.txt

echo "== Pre-pulling images (not billed against the experiment blocks' time estimates)"
docker pull -q rocm/vllm:latest
docker pull -q ghcr.io/remsky/kokoro-fastapi-cpu:latest

cat > /tmp/via-run.sh <<RUN
set -uo pipefail
cd $DIR
export PATH=$DIR/.venv/bin:\$PATH HF_TOKEN=$HF_TOKEN
scripts/run_budget.sh
echo "===== REPORT ====="; cat results/report/report.md 2>/dev/null
tar czf ~/via-results.tar.gz results
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  git checkout -B results && git add -f results && git -c user.name=via-bot -c user.email=via@localhost \
    commit -qm "MI300X results \$(date +%F)" && \
  git push -qf "https://x-access-token:${GITHUB_TOKEN:-}@github.com/globaledgepathway/voice-inference-analyzer" results \
    && echo "Pushed results to the 'results' branch."
fi
docker rm -f via-stt via-llm via-tts >/dev/null 2>&1
echo "ALL DONE. Results: ~/via-results.tar.gz. Now DESTROY the droplet (powered-off droplets are still billed)."
RUN
chmod 600 /tmp/via-run.sh
nohup bash /tmp/via-run.sh > ~/via-run.log 2>&1 &
echo
echo "Started in the background (pid $!). Follow it with:  tail -f ~/via-run.log"
echo "Expected ~2 hours. It stops itself before spending more than \$8.50 on this droplet."
