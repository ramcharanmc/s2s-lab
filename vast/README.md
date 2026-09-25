# vast/ — FT-1 on one rented GPU

FT-1 (Notion page 17 §7) trains a LoRA on Moshi from DailyTalk, with PR-010's test fold held out.
Everything that runs on the rented machine is here: `ft1.sh` (the whole run, safe to rerun),
`ft1_prep.py` (resume patch, training list, config; standard library only), `ft1.yaml` (the
config: moshi-finetune's example with paths filled and checkpoints every 100 steps).

Rules from page 17 §5.1: **destroy, never just stop** (a stopped machine still bills storage);
checkpoint often; no keys of yours go on the machine. The machine is on-demand (decided
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

**4. Copy the kit over and start it** (the SSH address comes from `vastai ssh-url INSTANCE_ID`):

    scp -P PORT vast/ft1.sh vast/ft1_prep.py vast/ft1.yaml root@HOST:/workspace/
    ssh -p PORT root@HOST
    tmux new -s ft1
    bash /workspace/ft1.sh

Detach with Ctrl-b d. Watch with `tmux attach -t ft1`, or `tail -f /workspace/ft1/train.log`.
If the machine fails, start it again (`vastai start instance INSTANCE_ID`) and run step 4's
last two lines again: the script resumes from the newest checkpoint (a deviation in PR-010).

**5. Copy the adapter off** as soon as the log says "done" (the machine destroys itself 45 minutes
later, and in any case 10 hours after the script first started):

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
