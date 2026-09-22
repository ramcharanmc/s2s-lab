✓ Initialized. View run at 
https://modal.com/apps/ramcharanmc/main/ap-cGoMKTWQC0Rw5NUP6G3gOi
✓ Created objects.
├── 🔨 Created mount 
│   /mnt/c/Users/ramch/Desktop/Ram/Resources/Audio/Code/s2s-lab/modal_app/r3_probe.py
├── 🔨 Created function download_weights.
└── 🔨 Created function probe.

==========
== CUDA ==
==========

CUDA Version 12.6.3

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.


[1ALoading weights:   0%|          | 0/876 [00:00<?, ?it/s]
[1ALoading weights:   7%|▋         | 60/876 [00:00<00:01, 589.99it/s]
[1ALoading weights:  15%|█▍        | 128/876 [00:00<00:01, 631.94it/s]
[1ALoading weights:  24%|██▍       | 210/876 [00:00<00:00, 700.75it/s]
[1ALoading weights:  33%|███▎      | 285/876 [00:00<00:00, 675.72it/s]
[1ALoading weights:  40%|████      | 353/876 [00:00<00:01, 277.49it/s]
[1ALoading weights:  46%|████▌     | 401/876 [00:01<00:02, 215.71it/s]
[1ALoading weights:  50%|█████     | 438/876 [00:01<00:02, 198.06it/s]
[1ALoading weights:  53%|█████▎    | 468/876 [00:01<00:02, 177.60it/s]
[1ALoading weights:  56%|█████▋    | 493/876 [00:02<00:02, 161.50it/s]
[1ALoading weights:  59%|█████▊    | 514/876 [00:02<00:02, 151.31it/s]
[1ALoading weights:  61%|██████    | 532/876 [00:02<00:02, 145.58it/s]
[1ALoading weights:  63%|██████▎   | 549/876 [00:02<00:02, 147.97it/s]
[1ALoading weights:  64%|██████▍   | 565/876 [00:02<00:02, 148.63it/s]
[1ALoading weights: 100%|██████████| 876/876 [00:02<00:00, 343.59it/s]
5 own-voice pairs, each take windowed and as recorded · Qwen/Qwen2-Audio-7B-Instruct on NVIDIA A10, 580.95.05 · commit 4c5905d-dirty · load 17 s

windowed:
| wording | SAD: answered right · ranked right (of 2) | FEA: answered right · ranked right (of 1) | ANG: answered right · ranked right (of 2) |
|---|---|---|---|
| upset | 0 · 0 | 1 · 1 | 0 · 2 |
| tone | 0 · 1 | 0 · 1 | 0 · 2 |
| not_words | 1 · 2 | 0 · 1 | 0 · 2 |
| emotion | 0 · 1 | 0 · 1 | 0 · 1 |

upset: "Is the person speaking upset?" -- Yes to 3 of 10 clips, median p(Yes) 0.38, median mass on Yes-or-No 0.91
tone: "Judging by their tone of voice, is the caller upset?" -- Yes to 6 of 10 clips, median p(Yes) 0.84, median mass on Yes-or-No 0.98
not_words: "Listen to how this was said, not to what was said. Does the speaker sound upset?" -- Yes to 9 of 10 clips, median p(Yes) 1.00, median mass on Yes-or-No 0.99
emotion: "Does the speaker sound sad, fearful or angry, rather than calm?" -- Yes to 10 of 10 clips, median p(Yes) 0.98, median mass on Yes-or-No 0.00

per pair, wording 'upset':
| pair | p(Yes) neutral | p(Yes) loaded | answered | ranked | the model heard (neutral clip) |
|---|---:|---:|---|---|---|
| sad-o01 | 0.01 | 0.01 | **wrong** | wrong | "The transcription of the speech is: 'I'm fine.'" |
| ang-o02 | 0.01 | 0.02 | **wrong** | right | "The transcription of the speech is: 'that's great.'" |
| ang-o03 | 0.36 | 0.50 | **wrong** | right | "The transcription of the speech is: 'Sure, whatever you say." |
| fea-o04 | 0.39 | 0.77 | right | right | "The transcription of the speech is: 'I'll be there at six.'" |
| sad-o05 | 0.90 | 0.85 | **wrong** | wrong | "The transcription of the speech is: 'Okay, I'll do that.'" |

as recorded:
| wording | SAD: answered right · ranked right (of 2) | FEA: answered right · ranked right (of 1) | ANG: answered right · ranked right (of 2) |
|---|---|---|---|
| upset | 0 · 1 | 0 · 1 | 0 · 0 |
| tone | 0 · 1 | 0 · 1 | 0 · 0 |
| not_words | 0 · 2 | 0 · 1 | 0 · 0 |
| emotion | 0 · 0 | 0 · 1 | 0 · 2 |

upset: "Is the person speaking upset?" -- Yes to 3 of 10 clips, median p(Yes) 0.40, median mass on Yes-or-No 0.94
tone: "Judging by their tone of voice, is the caller upset?" -- Yes to 6 of 10 clips, median p(Yes) 0.55, median mass on Yes-or-No 0.98
not_words: "Listen to how this was said, not to what was said. Does the speaker sound upset?" -- Yes to 10 of 10 clips, median p(Yes) 1.00, median mass on Yes-or-No 0.99
emotion: "Does the speaker sound sad, fearful or angry, rather than calm?" -- Yes to 10 of 10 clips, median p(Yes) 0.97, median mass on Yes-or-No 0.00

per pair, wording 'upset':
| pair | p(Yes) neutral | p(Yes) loaded | answered | ranked | the model heard (neutral clip) |
|---|---:|---:|---|---|---|
| sad-o01 | 0.01 | 0.00 | **wrong** | wrong | "The transcription of the speech is 'I'm fine.'" |
| ang-o02 | 0.05 | 0.02 | **wrong** | wrong | "The transcription of the speech is 'that's great.'" |
| ang-o03 | 0.56 | 0.41 | **wrong** | wrong | "The transcription of the speech is: 'Sure, whatever you say." |
| fea-o04 | 0.39 | 0.48 | **wrong** | right | "The transcription of the speech is: 'I'll be there at six.'" |
| sad-o05 | 0.81 | 0.84 | **wrong** | right | "The transcription of the speech is 'Okay, I'll do that.'" |

saved to runs/prosody/ownvoice-r3.json

42 s inside the function, 47 s end to end (queue, cold start and run): about $0.014 at the A10 rate. Modal's dashboard has the exact bill.
Stopping app - local entrypoint completed.
✓ App completed. View run at 
https://modal.com/apps/ramcharanmc/main/ap-cGoMKTWQC0Rw5NUP6G3gOi
