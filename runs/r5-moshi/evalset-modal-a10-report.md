✓ Initialized. View run at 
https://modal.com/apps/ramcharanmc/main/ap-rbc3UdBVSaZUjkeqpBXJee
✓ Created objects.
├── 🔨 Created mount 
│   <repo>/modal_app/r5_bench.py
├── 🔨 Created mount 
│   <repo>/src/s2slab
├── 🔨 Created function download_weights.
├── 🔨 Created function download_personaplex.
├── 🔨 Created mount 
│   <repo>/probes/endpointing
├── 🔨 Created mount <repo>/evalset
├── 🔨 Created function smoke.
├── 🔨 Created function evaluate.
├── 🔨 Created function smoke_personaplex.
└── 🔨 Created function evaluate_personaplex.

==========
== CUDA ==
==========

CUDA Version 12.6.3

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

/usr/local/lib/python3.12/site-packages/moshi/models/loaders.py:204: UserWarning: Repository kyutai/moshiko-pytorch-bf16 contains no config.json. Assuming this is a Moshi 7B. Support for such repository might be removed in the future.
  warnings.warn(
r5-moshi: R5 full duplex: Kyutai Moshi (moshiko, 7B, bf16) -- no turns, both streams at once
Modal NVIDIA A10, 580.95.05 · commit 6723cb4 · load 14 s · peak GPU memory 16,621 MB

stream time: what a caller would hear, independent of the GPU's speed

| cohort | clips | opened the call | response gap p50 ms | p90 ms | no answer within the tail | voiced while the caller spoke | backchannels | interruptions | median overlap | median RTF | median set-up s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 50 | 50 | 140 | — | 23 | 18 | 17 | 8 | 0.00 | 0.69 | 0.01 |
| accented | 50 | 50 | 140 | — | 25 | 16 | 12 | 11 | 0.00 | 0.69 | 0.01 |
| disfluent | 50 | 50 | 180 | — | 24 | 17 | 16 | 8 | 0.00 | 0.69 | 0.01 |
| slow | 50 | 50 | — | — | 32 | 14 | 10 | 8 | 0.00 | 0.69 | 0.01 |
| all | 200 | 200 | 150 | — | 104 | 65 | 55 | 35 | 0.00 | 0.69 | 0.01 |

already talking as the caller stopped (negative gap): 24 of 200
the same clip driven twice: clean-00 identical, clean-01 identical

what it said after the caller stopped (first five clips that got an answer):
  clean-03: "'m sorry. I know how frustrating that can be. Let's try to find a way out. What do you suggest?"
  clean-04: 'Okay.'
  clean-07: '"Oof, that\'s not cheap. What\' the problem?"'
  clean-09: "' a classic heist movie trope! The two thieves, one a mastermind and the other a thug, they work tog"
  clean-10: 'What a wonderful passage to read on a rainy day.'

saved -> runs/r5-moshi/evalset-modal-a10.json
torch 2.8.0+cu126, moshi 0.2.13
2833 s inside the function, 2839 s end to end (queue, cold start and run): about $0.867 at the A10 rate. Modal's dashboard has the exact bill.
Stopping app - local entrypoint completed.
✓ App completed. View run at 
https://modal.com/apps/ramcharanmc/main/ap-rbc3UdBVSaZUjkeqpBXJee
