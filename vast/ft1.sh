#!/usr/bin/env bash
# FT-1 on one rented Vast.ai GPU: a LoRA on Moshi (moshiko) from DailyTalk, PR-010's test fold removed.
#
# On the machine, from the folder holding ft1.sh, ft1_prep.py and ft1.yaml (copied there with scp):
#     tmux new -s ft1          then          bash ft1.sh
# Detach with Ctrl-b d; come back with: tmux attach -t ft1
#
# SAFE TO RUN AGAIN after an interruption: each step is skipped once it is done, and training resumes
# from the newest checkpoint (every 100 steps). When training ends, the adapter is copied to
# /workspace/ft1/out and the machine destroys itself GRACE_MIN minutes later: copy the adapter off
# before then. Independently, the machine destroys itself MAX_HOURS after this script starts.
# No key of yours is needed on the machine; only the adapter comes off it. (An optional read-only
# HF_TOKEN in the shell beats Hugging Face's rate limit: vast/README.md, step 4.)
#
# Fixed after FT-1's run (2026-09-25), from what it hit: the self-destroy never armed (Vast puts
# CONTAINER_ID and CONTAINER_API_KEY in the container's own environment, not the SSH shell's, and
# `vastai destroy` asks y/N); the cap was 10 h where FT-1 needed 12.3 h of machine; torchrun failed
# without CUDA_VISIBLE_DEVICES; the dataset download hit the rate limit twice.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK=/workspace/ft1
MF_COMMIT=2acc879fe7c48f885a18f6cc9548bccb2674d87b      # kyutai-labs/moshi-finetune, main on 2025-07-07
MOSHI_COMMIT=9837ca328d58deef5d7a4fe95a0fb49c902ec0ae   # kyutai-labs/moshi, its commit of the same morning
DATA_REV=33e1b501f725a6f4ed4ded95e16cd7f66b9d4bdc        # kyutai/DailyTalkContiguous, as s2slab.dailytalk.REVISION
MAX_HOURS=${MAX_HOURS:-15}                                  # FT-1: 11.4 h of training, 12.3 h of machine
GRACE_MIN=${GRACE_MIN:-45}

mkdir -p "$WORK"
cd "$WORK"
log() { echo "[ft1 $(date -u +%H:%M:%S)] $*" | tee -a "$WORK/ft1.log"; }
from_pid1() {    # a variable from the container's own environment: Vast sets it there, not in SSH shells
  tr '\0' '\n' < "${PID1_ENVIRON:-/proc/1/environ}" 2>/dev/null | sed -n "s/^$1=//p" | head -n 1
}
export CONTAINER_ID="${CONTAINER_ID:-$(from_pid1 CONTAINER_ID)}"
export CONTAINER_API_KEY="${CONTAINER_API_KEY:-$(from_pid1 CONTAINER_API_KEY)}"   # this instance's own key, not yours
can_destroy() {  # stores the instance's key where vastai looks for one, then checks it can see the instance
  [ -n "$CONTAINER_ID" ] && [ -n "$CONTAINER_API_KEY" ] &&
    vastai set api-key "$CONTAINER_API_KEY" >/dev/null 2>&1 && vastai show instance "$CONTAINER_ID" >/dev/null 2>&1
}
destroy_in() {   # destroy this machine after $1 seconds, surviving the end of this script; answers the y/N
  setsid nohup bash -c 'sleep "$0"; echo y | vastai destroy instance "$CONTAINER_ID"' "$1" \
    >>"$WORK/destroy.log" 2>&1 < /dev/null &
}

# 0. The hard cap: the machine removes itself, whatever else happens.
pip install -q vastai >/dev/null 2>&1 || true
if can_destroy; then
  destroy_in $((MAX_HOURS * 3600))
  log "hard cap armed: instance $CONTAINER_ID destroys itself in ${MAX_HOURS} h"
else
  log "WARNING: this machine cannot destroy itself (no CONTAINER_ID / CONTAINER_API_KEY that works). Destroy it yourself when done: vastai destroy instance <id>"
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "$WORK/ft1.log"

# 1. The code: moshi-finetune and moshi, pinned; the resume patch.
if [ ! -f .installed ]; then
  rm -rf moshi-finetune
  git clone -q https://github.com/kyutai-labs/moshi-finetune.git
  git -C moshi-finetune checkout -q "$MF_COMMIT"
  sed -i "s|moshi.git#subdirectory=moshi|moshi.git@${MOSHI_COMMIT}#subdirectory=moshi|" moshi-finetune/pyproject.toml
  grep -q "$MOSHI_COMMIT" moshi-finetune/pyproject.toml || { log "could not pin moshi"; exit 1; }
  pip install -q -e ./moshi-finetune
  touch .installed
  log "installed moshi-finetune $MF_COMMIT with moshi $MOSHI_COMMIT"
fi
python3 "$HERE/ft1_prep.py" patch moshi-finetune | tee -a "$WORK/ft1.log"

# 2. The data: all of DailyTalk (13.8 GB), then the training list without the test fold. Anonymous
# downloads hit Hugging Face's rate limit; a token raises it; either way a 429 waits and retries.
if [ ! -f data/dailytalk/.complete ]; then
  if [ -n "${HF_TOKEN:-}" ]; then log "HF_TOKEN is set: authenticated download"; else log "no HF_TOKEN: anonymous download (rate-limited)"; fi
  python3 "$HERE/ft1_prep.py" download data/dailytalk "$DATA_REV" | tee -a "$WORK/ft1.log"
  touch data/dailytalk/.complete
fi
log "training list: $(python3 "$HERE/ft1_prep.py" manifest data/dailytalk)"

# 3. Train, resuming from the newest checkpoint if there is one.
python3 "$HERE/ft1_prep.py" config "$HERE/ft1.yaml" "$WORK/ft1_run.yaml" "$WORK/run" "$WORK/data/dailytalk/train.jsonl" | tee -a "$WORK/ft1.log"
cp "$WORK/ft1_run.yaml" "$WORK/ft1_run.$(date -u +%Y%m%d-%H%M%S).yaml"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"     # moshi-finetune reads it; Vast's image does not set it
(cd moshi-finetune && torchrun --nproc-per-node 1 -m train "$WORK/ft1_run.yaml") 2>&1 | tee -a "$WORK/train.log"

# 4. Done: the adapter out, its checksum, then the grace period.
read -r STEP CKPT <<< "$(python3 "$HERE/ft1_prep.py" last "$WORK/run")"
mkdir -p out
cp "$CKPT/lora.safetensors" "$CKPT/config.json" out/
cp ft1.log train.log ft1_run*.yaml out/
cp data/dailytalk/train.jsonl out/
(cd out && sha256sum lora.safetensors config.json train.jsonl > SHA256SUMS)
log "done at step $STEP. The adapter and its evidence are in $WORK/out:"
ls -la out | tee -a "$WORK/ft1.log"
if can_destroy; then
  destroy_in $((GRACE_MIN * 60))
  log "copy it off now: this machine destroys itself in ${GRACE_MIN} min"
fi
