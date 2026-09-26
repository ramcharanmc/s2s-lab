# vast/ — FT-1 on one rented GPU

FT-1 (Notion page 17 §7) trains a LoRA on Moshi from DailyTalk, with PR-010's test fold held out.
Everything that runs on the rented machine is here: `ft1.sh` (the whole run, safe to rerun),
`ft1_prep.py` (resume patch, training list, config; standard library only), `ft1.yaml` (the
config: moshi-finetune's example with paths filled and checkpoints every 100 steps).

Rules from page 17 §5.1: **destroy, never just stop** (a stopped machine still bills storage);
checkpoint often; no keys of yours go on the machine (the one exception, and your call: a
read-only Hugging Face token for the download, step 4). The machine is on-demand (decided
2026-09-25), so no run is taken away mid-way.

## The steps, in order

**1. Download the 100 test dialogues** — only after PR-010 is committed (about 470 MB):

    PYTHONPATH=src python - <<'EOF'
    from huggingface_hub import snapshot_download
    from s2slab.dailytalk import REPO, REVISION, TEST_IDS
    snapshot_download(REPO, repo_type="dataset", revision=REVISION, local_dir="data/dailytalk",
                      allow_patterns=[f"data_stereo/{i}.{x}" for i in TEST_IDS for x in ("wav", "json")])
    EOF

**2. Find a machine.** One 80 GB GPU (moshi-finetune peaks at 39.6 GB), on-demand prices:

    vastai search offers 'gpu_ram>=79 num_gpus=1 disk_space>=120 reliability>0.98 inet_down>500' -o 'dph'

**3. Rent it**, on-demand (the image carries torch 2.6, which moshi-finetune pins):

    vastai create instance OFFER_ID --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel --disk 120 --ssh --direct

**4. Copy the kit over and start it.** Use the machine's direct address: the proxy that
`vastai ssh-url` prints (ssh4.vast.ai) timed out on FT-1. The direct one, once the status is
`running`:

    vastai show instance INSTANCE_ID --raw | python3 -c "import json, sys; d = json.load(sys.stdin); print(d['public_ipaddr'], d['ports']['22/tcp'][0]['HostPort'])"

prints `HOST PORT`. Then:

    scp -P PORT vast/ft1.sh vast/ft1_prep.py vast/ft1.yaml root@HOST:/workspace/
    ssh -p PORT root@HOST

The login shell is already inside tmux (Vast starts one; `tmux new` would nest it). Optionally,
to beat the download's rate limit, give the shell a **read-only** Hugging Face token without it
reaching the screen or the history -- a key of yours on a rented machine, so your call, and revoke
it afterwards:

    read -rs HF_TOKEN && export HF_TOKEN

Then:

    bash /workspace/ft1.sh

Detach with Ctrl-b d; after a new `ssh`, `tmux attach`. Or watch `tail -f /workspace/ft1/train.log`.
The log's first line says whether the machine can destroy itself (step 5). If the machine fails,
start it again (`vastai start instance INSTANCE_ID`) and run the script again: it resumes from the
newest checkpoint (a deviation for any registration that depends on the run).

**5. Copy the adapter off** as soon as the log says "done" (the machine destroys itself 45 minutes
later, and in any case 15 hours after the script started -- FT-1 needed 12.3). If the log said it
cannot destroy itself, destroy it yourself; either way, check with `vastai show instances`:

    mkdir -p data/ft1
    scp -P PORT 'root@HOST:/workspace/ft1/out/*' data/ft1/
    vastai destroy instance INSTANCE_ID
    vastai show instances

**6. Put it where Modal can read it**, then run PR-010:

    modal volume put ft1 data/ft1/lora.safetensors /lora.safetensors
    modal volume put ft1 data/ft1/config.json /config.json
    modal run modal_app/r5_bench.py::pr010 --confirm PR-010 2>&1 | tee results/ft1-pr010.txt

`data/` is ignored by git; the training evidence from `data/ft1/` (the config run, the logs, the
training list and SHA256SUMS -- not the adapter) is committed with PR-010's results.
