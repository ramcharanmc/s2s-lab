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
# No key of yours is ever placed on the machine; only the adapter comes off it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK=/workspace/ft1
MF_COMMIT=2acc879fe7c48f885a18f6cc9548bccb2674d87b      # kyutai-labs/moshi-finetune, main on 2025-07-07
MOSHI_COMMIT=9837ca328d58deef5d7a4fe95a0fb49c902ec0ae   # kyutai-labs/moshi, its commit of the same morning
DATA_REV=33e1b501f725a6f4ed4ded95e16cd7f66b9d4bdc        # kyutai/DailyTalkContiguous, as s2slab.dailytalk.REVISION
MAX_HOURS=${MAX_HOURS:-10}
GRACE_MIN=${GRACE_MIN:-45}

mkdir -p "$WORK"
cd "$WORK"
log() { echo "[ft1 $(date -u +%H:%M:%S)] $*" | tee -a "$WORK/ft1.log"; }
destroy_in() {   # destroy this machine after $1 seconds, surviving the end of this script
  setsid nohup bash -c "sleep $1; vastai destroy instance $CONTAINER_ID" >/dev/null 2>&1 < /dev/null &
}

# 0. The hard cap: the machine removes itself, whatever else happens.
pip install -q vastai >/dev/null 2>&1 || true
if [ -n "${CONTAINER_ID:-}" ] && vastai show instance "$CONTAINER_ID" >/dev/null 2>&1; then
  destroy_in $((MAX_HOURS * 3600))
  log "hard cap armed: instance $CONTAINER_ID destroys itself in ${MAX_HOURS} h"
else
  log "WARNING: this machine cannot destroy itself. Destroy it yourself when done: vastai destroy instance <id>"
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

# 2. The data: all of DailyTalk (13.8 GB), then the training list without the test fold.
if [ ! -f data/dailytalk/.complete ]; then
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('kyutai/DailyTalkContiguous', repo_type='dataset', revision='$DATA_REV', local_dir='data/dailytalk')"
  touch data/dailytalk/.complete
fi
log "training list: $(python3 "$HERE/ft1_prep.py" manifest data/dailytalk)"

# 3. Train, resuming from the newest checkpoint if there is one.
python3 "$HERE/ft1_prep.py" config "$HERE/ft1.yaml" "$WORK/ft1_run.yaml" "$WORK/run" "$WORK/data/dailytalk/train.jsonl" | tee -a "$WORK/ft1.log"
cp "$WORK/ft1_run.yaml" "$WORK/ft1_run.$(date -u +%Y%m%d-%H%M%S).yaml"
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
if [ -n "${CONTAINER_ID:-}" ] && vastai show instance "$CONTAINER_ID" >/dev/null 2>&1; then
  destroy_in $((GRACE_MIN * 60))
  log "copy it off now: this machine destroys itself in ${GRACE_MIN} min"
fi
