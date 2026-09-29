✓ Initialized. View run at 
https://modal.com/apps/ramcharanmc/main/ap-RRT3K9eB24MMDf7MrID1aT
✓ Created objects.
├── 🔨 Created mount 
│   <repo>/modal_app/r3_probe.py
├── 🔨 Created mount 
│   <repo>/src/s2slab
├── 🔨 Created function download_weights.
├── 🔨 Created mount <repo>/evalset
├── 🔨 Created function probe.
└── 🔨 Created function evaluate.

==========
== CUDA ==
==========

CUDA Version 12.6.3

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.


[1ALoading weights:   0%|          | 0/290 [00:00<?, ?it/s]
[1ALoading weights:   0%|          | 1/290 [00:00<00:57,  5.05it/s]
[1ALoading weights:  84%|████████▍ | 243/290 [00:00<00:00, 942.20it/s]
[1ALoading weights: 100%|██████████| 290/290 [00:00<00:00, 867.37it/s]
/usr/local/lib/python3.12/site-packages/torch/nn/modules/rnn.py:1011: UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.2 and num_layers=1
  super().__init__("LSTM", *args, **kwargs)
/usr/local/lib/python3.12/site-packages/torch/nn/utils/weight_norm.py:145: FutureWarning: `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.
  WeightNorm.apply(module, name, dim)
/usr/local/lib/python3.12/site-packages/torch/jit/_script.py:1491: FutureWarning: `torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`.
  warnings.warn(

[1ALoading weights:   0%|          | 0/290 [00:00<?, ?it/s]
[1ALoading weights:   0%|          | 1/290 [00:00<00:36,  7.83it/s]
[1ALoading weights: 100%|██████████| 290/290 [00:00<00:00, 1663.03it/s]
/usr/local/lib/python3.12/site-packages/torch/nn/modules/rnn.py:1011: UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.2 and num_layers=1
  super().__init__("LSTM", *args, **kwargs)
/usr/local/lib/python3.12/site-packages/torch/nn/utils/weight_norm.py:145: FutureWarning: `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.
  WeightNorm.apply(module, name, dim)

[1ALoading weights:   0%|          | 0/876 [00:00<?, ?it/s]
[1ALoading weights:   4%|▍         | 39/876 [00:00<00:02, 384.61it/s]
[1ALoading weights:  12%|█▏        | 103/876 [00:00<00:01, 533.59it/s]
[1ALoading weights:  19%|█▊        | 163/876 [00:00<00:01, 563.64it/s]
[1ALoading weights:  26%|██▌       | 224/876 [00:00<00:01, 578.30it/s]
[1ALoading weights:  35%|███▍      | 306/876 [00:00<00:00, 658.28it/s]
[1ALoading weights:  42%|████▏     | 372/876 [00:00<00:01, 331.67it/s]
[1ALoading weights:  48%|████▊     | 422/876 [00:01<00:02, 212.08it/s]
[1ALoading weights:  52%|█████▏    | 459/876 [00:01<00:02, 172.76it/s]
[1ALoading weights:  56%|█████▌    | 488/876 [00:02<00:02, 150.93it/s]
[1ALoading weights:  58%|█████▊    | 511/876 [00:02<00:02, 144.49it/s]
[1ALoading weights:  61%|██████    | 531/876 [00:02<00:02, 134.11it/s]
[1ALoading weights:  63%|██████▎   | 548/876 [00:02<00:02, 135.77it/s]
[1ALoading weights:  64%|██████▍   | 564/876 [00:02<00:02, 132.75it/s]
[1ALoading weights:  66%|██████▌   | 579/876 [00:02<00:02, 131.51it/s]
[1ALoading weights: 100%|██████████| 876/876 [00:02<00:00, 315.14it/s]
/usr/local/lib/python3.12/site-packages/torch/nn/modules/rnn.py:1011: UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.2 and num_layers=1
  super().__init__("LSTM", *args, **kwargs)
/usr/local/lib/python3.12/site-packages/torch/nn/utils/weight_norm.py:145: FutureWarning: `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.
  WeightNorm.apply(module, name, dim)
r0-gpu: R0, same models on a CUDA GPU (fp16)
commit 960b491 · Modal NVIDIA A10, 580.95.05, one container: r0-gpu then r1-gpu then r3-gpu · run 2026-09-23 · eval set built 2026-09-15, seed 20260915
load 15.1 s, once, before the first turn · peak GPU memory 1,851 MB

### Processing time

| cohort | turns | p50 ms | p90 ms | p99 ms |
|---|---:|---:|---:|---:|
| clean | 50 | 443 | — | — |
| accented | 50 | 449 | — | — |
| disfluent | 50 | 457 | — | — |
| slow | 50 | 422 | — | — |
| all | 200 | 441 | 747 | — |

### First audio: when the first piece of the reply existed

| cohort | turns | p50 ms | p90 ms | p99 ms |
|---|---:|---:|---:|---:|
| clean | 50 | 443 | — | — |
| accented | 50 | 449 | — | — |
| disfluent | 50 | 457 | — | — |
| slow | 50 | 422 | — | — |
| all | 200 | 441 | 747 | — |

### First sound: first audio plus the silence before the first word

| cohort | turns | p50 ms | p90 ms | p99 ms |
|---|---:|---:|---:|---:|
| clean | 50 | 755 | — | — |
| accented | 50 | 764 | — | — |
| disfluent | 50 | 775 | — | — |
| slow | 50 | 734 | — | — |
| all | 200 | 760 | 1063 | — |

### Where the time went

| stage | p50 ms | p90 ms | p99 ms | n |
|---|---:|---:|---:|---:|
| first sound | 759.7 | 1062.6 | — | 200 |
| total | 441.4 | 747.5 | — | 200 |
| first audio | 441.4 | 747.5 | — | 200 |
| llm | 315.1 | 572.1 | — | 200 |
| tts | 79.8 | 113.6 | — | 200 |
| asr | 46.6 | 68.3 | — | 200 |
| (untimed) | 0.0 | 0.0 | — | 200 |

### Word error rate

| cohort | turns | ref words | raw WER | normalized WER |
|---|---:|---:|---:|---:|
| accented | 50 | 884 | 29.1% | 26.6% |
| clean | 50 | 846 | 6.3% | 5.6% |
| disfluent | 50 | 1039 | 12.1% | 9.7% |
| slow | 50 | 846 | 6.1% | 5.6% |
| **all** | 200 | 3615 | 13.5% | 12.0% |

### disfluent cohort against the tidied reference (fillers and false starts already removed)

| cohort | turns | ref words | raw WER | normalized WER |
|---|---:|---:|---:|---:|
| disfluent | 50 | 899 | 12.6% | 12.1% |

reply leading silence: p50   320.0  p90   320.0  p99   340.0  [n=200, unsupported: p99]
playback stall: p50     0.0  p90     0.0  p99     0.0  [n=200, unsupported: p99]  (0 of 200 turns ran dry)

saved -> runs/r0-gpu/evalset-modal-a10.json

r1-gpu: R1, same models on a CUDA GPU (fp16), reply streamed clause by clause
commit 960b491 · Modal NVIDIA A10, 580.95.05, one container: r0-gpu then r1-gpu then r3-gpu · run 2026-09-23 · eval set built 2026-09-15, seed 20260915
load 3.5 s, once, before the first turn · peak GPU memory 1,684 MB

### Processing time

| cohort | turns | p50 ms | p90 ms | p99 ms |
|---|---:|---:|---:|---:|
| clean | 50 | 455 | — | — |
| accented | 50 | 473 | — | — |
| disfluent | 50 | 451 | — | — |
| slow | 50 | 421 | — | — |
| all | 200 | 451 | 754 | — |

### First audio: when the first piece of the reply existed

| cohort | turns | p50 ms | p90 ms | p99 ms |
|---|---:|---:|---:|---:|
| clean | 50 | 345 | — | — |
| accented | 50 | 286 | — | — |
| disfluent | 50 | 328 | — | — |
| slow | 50 | 336 | — | — |
| all | 200 | 333 | 483 | — |

### First sound: first audio plus the silence before the first word

| cohort | turns | p50 ms | p90 ms | p99 ms |
|---|---:|---:|---:|---:|
| clean | 50 | 662 | — | — |
| accented | 50 | 610 | — | — |
| disfluent | 50 | 643 | — | — |
| slow | 50 | 655 | — | — |
| all | 200 | 651 | 787 | — |

### Where the time went

| stage | p50 ms | p90 ms | p99 ms | n |
|---|---:|---:|---:|---:|
| first sound | 650.7 | 786.9 | — | 200 |
| total | 450.9 | 754.2 | — | 200 |
| first audio | 332.7 | 482.7 | — | 200 |
| llm | 262.8 | 477.2 | — | 200 |
| tts | 153.4 | 278.1 | — | 200 |
| asr | 44.8 | 63.3 | — | 200 |
| (untimed) | 0.4 | 0.7 | — | 200 |

### Word error rate

| cohort | turns | ref words | raw WER | normalized WER |
|---|---:|---:|---:|---:|
| accented | 50 | 884 | 29.1% | 26.6% |
| clean | 50 | 846 | 6.3% | 5.6% |
| disfluent | 50 | 1039 | 12.1% | 9.7% |
| slow | 50 | 846 | 6.1% | 5.6% |
| **all** | 200 | 3615 | 13.5% | 12.0% |

### disfluent cohort against the tidied reference (fillers and false starts already removed)

| cohort | turns | ref words | raw WER | normalized WER |
|---|---:|---:|---:|---:|
| disfluent | 50 | 899 | 12.6% | 12.1% |

reply leading silence: p50   320.0  p90   360.0  p99   380.0  [n=200, unsupported: p99]
playback stall: p50     0.0  p90     0.0  p99     0.0  [n=200, unsupported: p99]  (0 of 200 turns ran dry)

saved -> runs/r1-gpu/evalset-modal-a10.json

r3-gpu: R3 speech-LLM: Qwen2-Audio-7B-Instruct hears the turn -> Kokoro-82M; no transcript (bf16, CUDA)
commit 960b491 · Modal NVIDIA A10, 580.95.05, one container: r0-gpu then r1-gpu then r3-gpu · run 2026-09-23 · eval set built 2026-09-15, seed 20260915
load 11.8 s, once, before the first turn · peak GPU memory 16,993 MB

### Processing time

| cohort | turns | p50 ms | p90 ms | p99 ms |
|---|---:|---:|---:|---:|
| clean | 50 | 1039 | — | — |
| accented | 50 | 1011 | — | — |
| disfluent | 50 | 1148 | — | — |
| slow | 50 | 972 | — | — |
| all | 200 | 1055 | 2016 | — |

### First audio: when the first piece of the reply existed

| cohort | turns | p50 ms | p90 ms | p99 ms |
|---|---:|---:|---:|---:|
| clean | 50 | 1039 | — | — |
| accented | 50 | 1011 | — | — |
| disfluent | 50 | 1148 | — | — |
| slow | 50 | 972 | — | — |
| all | 200 | 1055 | 2016 | — |

### First sound: first audio plus the silence before the first word

| cohort | turns | p50 ms | p90 ms | p99 ms |
|---|---:|---:|---:|---:|
| clean | 50 | 1350 | — | — |
| accented | 50 | 1329 | — | — |
| disfluent | 50 | 1458 | — | — |
| slow | 50 | 1279 | — | — |
| all | 200 | 1363 | 2318 | — |

### Where the time went

| stage | p50 ms | p90 ms | p99 ms | n |
|---|---:|---:|---:|---:|
| first sound | 1362.8 | 2317.5 | — | 200 |
| total | 1055.4 | 2015.5 | — | 200 |
| first audio | 1055.4 | 2015.5 | — | 200 |
| llm | 971.1 | 1861.5 | — | 200 |
| tts | 86.5 | 145.7 | — | 200 |
| (untimed) | 0.0 | 0.0 | — | 200 |

### Word error rate

no transcript in the loop: the audio goes straight into the LLM, so there is nothing to score

### asked to transcribe, outside the loop (untimed; not what the reply was made from)

| cohort | turns | ref words | raw WER | normalized WER |
|---|---:|---:|---:|---:|
| accented | 50 | 884 | 33.8% | 29.8% |
| clean | 50 | 846 | 10.5% | 8.6% |
| disfluent | 50 | 1039 | 18.3% | 15.2% |
| slow | 50 | 846 | 9.5% | 7.8% |
| **all** | 200 | 3615 | 18.2% | 15.5% |

reply leading silence: p50   300.0  p90   320.0  p99   340.0  [n=200, unsupported: p99]
playback stall: p50     0.0  p90     0.0  p99     0.0  [n=200, unsupported: p99]  (0 of 200 turns ran dry)

saved -> runs/r3-gpu/evalset-modal-a10.json

718 s inside the function, 724 s end to end (queue, cold start and run): about $0.221 at the A10 rate. Modal's dashboard has the exact bill.
Stopping app - local entrypoint completed.
✓ App completed. View run at 
https://modal.com/apps/ramcharanmc/main/ap-RRT3K9eB24MMDf7MrID1aT
