✓ Initialized. View run at 
https://modal.com/apps/ramcharanmc/main/ap-iNyHmVzEjwpUL5yw48aspV
✓ Created objects.
├── 🔨 Created mount 
│   <repo>/modal_app/r3_probe.py
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
[1ALoading weights:   7%|▋         | 65/876 [00:00<00:01, 624.27it/s]
[1ALoading weights:  15%|█▍        | 130/876 [00:00<00:01, 634.73it/s]
[1ALoading weights:  24%|██▍       | 209/876 [00:00<00:00, 695.94it/s]
[1ALoading weights:  32%|███▏      | 282/876 [00:00<00:00, 708.88it/s]
[1ALoading weights:  40%|████      | 353/876 [00:00<00:01, 427.87it/s]
[1ALoading weights:  47%|████▋     | 408/876 [00:01<00:01, 268.62it/s]
[1ALoading weights:  51%|█████▏    | 449/876 [00:01<00:01, 220.55it/s]
[1ALoading weights:  55%|█████▍    | 481/876 [00:01<00:01, 198.07it/s]
[1ALoading weights:  58%|█████▊    | 508/876 [00:01<00:02, 163.44it/s]
[1ALoading weights:  61%|██████    | 530/876 [00:02<00:02, 157.10it/s]
[1ALoading weights:  63%|██████▎   | 549/876 [00:02<00:02, 142.00it/s]
[1ALoading weights:  65%|██████▍   | 566/876 [00:02<00:02, 139.71it/s]
[1ALoading weights:  67%|██████▋   | 583/876 [00:02<00:02, 140.86it/s]
[1ALoading weights:  68%|██████▊   | 598/876 [00:02<00:02, 135.11it/s]
[1ALoading weights: 100%|██████████| 876/876 [00:02<00:00, 331.42it/s]

[1ALoading weights:   0%|          | 0/876 [00:00<?, ?it/s]
[1ALoading weights:  13%|█▎        | 112/876 [00:00<00:00, 1109.48it/s]
[1ALoading weights:  25%|██▌       | 223/876 [00:00<00:00, 751.67it/s] 
[1ALoading weights:  35%|███▍      | 305/876 [00:00<00:00, 683.34it/s]
[1ALoading weights:  43%|████▎     | 377/876 [00:00<00:00, 524.11it/s]
[1ALoading weights: 100%|██████████| 876/876 [00:00<00:00, 1277.22it/s]
39 test pairs · confirms PR-006 · Qwen/Qwen2-Audio-7B-Instruct on NVIDIA A10, 580.95.05 · commit 5c1f3aa · load 5 s 

| wording | SAD: answered right · ranked right (of 13) | FEA: answered right · ranked right (of 13) | ANG: answered right · ranked right (of 13) |
|---|---|---|---|
| upset | 11 · 13 | 11 · 13 | 9 · 13 |

upset: "Is the person speaking upset?" -- Yes to 47 of 78 clips, median p(Yes) 0.96, median mass on Yes-or-No 0.95

per pair, wording 'upset':
| pair | p(Yes) neutral | p(Yes) loaded | answered | ranked | the model heard (neutral clip) |
|---|---:|---:|---|---|---|
| sad-t01 | 0.21 | 1.00 | right | right | "The exact transcription of the speech is: 'That is exactly w" |
| sad-t02 | 0.23 | 0.99 | right | right | "The transcription of the speech is: 'I'm on my way to the me" |
| sad-t03 | 0.16 | 1.00 | right | right | "The transcription of the speech is: 'the airplane is almost " |
| sad-t04 | 0.06 | 1.00 | right | right | "The transcription of the speech is: 'Maybe tomorrow it will " |
| sad-t05 | 0.22 | 0.98 | right | right | "The transcription of the speech is: 'Maybe tomorrow it will " |
| sad-t06 | 0.33 | 0.99 | right | right | "The transcription of the speech is: 'it's eleven o'clock.'" |
| sad-t07 | 0.42 | 0.94 | right | right | "The transcription of the speech is: 'we'll stop in a couple " |
| sad-t08 | 0.79 | 1.00 | **wrong** | right | "The transcription of the speech is: 'it's eleven o'clock.'" |
| sad-t09 | 0.50 | 1.00 | right | right | "The transcription of the speech is: 'it's eleven o'clock.'" |
| sad-t10 | 0.10 | 0.99 | right | right | "The transcription of the speech is: 'Maybe tomorrow it will " |
| sad-t11 | 0.15 | 0.98 | right | right | "The transcription of the speech is: 'the airplane is almost " |
| sad-t12 | 0.98 | 0.99 | **wrong** | right | "The transcription of the speech is: 'I'm on my way to the me" |
| sad-t13 | 0.36 | 0.99 | right | right | "The transcription of the speech is: 'I think I have a doctor" |
| fea-t01 | 0.75 | 1.00 | **wrong** | right | "The transcription of the speech is: 'it's eleven o'clock.'" |
| fea-t02 | 0.15 | 1.00 | right | right | "The transcription of the speech is: 'the airplane is almost " |
| fea-t03 | 0.16 | 0.99 | right | right | "The speech transcribed from the audio is: 'the surface is sl" |
| fea-t04 | 0.20 | 0.99 | right | right | "The transcription of the speech is: 'it's eleven o'clock.'" |
| fea-t05 | 0.78 | 1.00 | **wrong** | right | "The transcription of the speech is: 'it's eleven o'clock.'" |
| fea-t06 | 0.32 | 0.99 | right | right | "The transcription of the speech is: 'I think I have a doctor" |
| fea-t07 | 0.50 | 1.00 | right | right | "The transcription of the speech is: 'I wonder what this is a" |
| fea-t08 | 0.26 | 1.00 | right | right | "The transcription of the speech is: 'I wonder what this is a" |
| fea-t09 | 0.29 | 1.00 | right | right | "The transcription of the speech is: 'I would like a new alar" |
| fea-t10 | 0.13 | 0.99 | right | right | "The transcription of the speech is: 'I think I've seen this " |
| fea-t11 | 0.13 | 0.98 | right | right | "The speech transcribed from the audio is 'the surface is sli" |
| fea-t12 | 0.04 | 0.98 | right | right | "The speech transcribed from the audio is: 'the surface is sl" |
| fea-t13 | 0.09 | 0.97 | right | right | "The transcription of the speech is: 'I think I've seen this " |
| ang-t01 | 0.53 | 0.99 | **wrong** | right | "The exact words spoken in the audio are: 'That is exactly wh" |
| ang-t02 | 0.44 | 1.00 | right | right | "The transcription of the speech is: 'we'll stop in a couple " |
| ang-t03 | 0.19 | 1.00 | right | right | "The transcription of the speech is: 'we'll stop in a couple " |
| ang-t04 | 0.25 | 1.00 | right | right | "The transcription of the speech is: 'it's eleven o'clock.'" |
| ang-t05 | 0.71 | 1.00 | **wrong** | right | "The transcription of the speech is: 'it's eleven o'clock.'" |
| ang-t06 | 0.93 | 1.00 | **wrong** | right | "The transcription of the speech is: 'it's eleven o'clock.'" |
| ang-t07 | 0.20 | 0.99 | right | right | "The transcription of the speech is: 'Don't forget a jacket.'" |
| ang-t08 | 0.45 | 0.99 | right | right | "The transcription of the speech is: 'Don't forget a jacket.'" |
| ang-t09 | 0.16 | 0.99 | right | right | "The transcription of the speech is: 'Don't forget a jacket.'" |
| ang-t10 | 0.39 | 0.98 | right | right | "The transcription of the speech is: 'I think I have a doctor" |
| ang-t11 | 0.26 | 1.00 | right | right | "The transcription of the speech is: 'I'm on my way to the me" |
| ang-t12 | 0.69 | 1.00 | **wrong** | right | "The transcription of the speech is: 'I would like a new alar" |
| ang-t13 | 0.04 | 0.99 | right | right | "The speech transcribed from the audio is 'The surface is sli" |

Claim A -- the input bottleneck: R3 ranks the upset clip above its calm twin
| criterion | ranked right | bar | |
|---|---:|---|---|
| A1 sad | 13 of 13 | at least 12 | PASS |
| A2 fear | 13 of 13 | at least 12 | PASS |
| A3 anger | 13 of 13 | at least 12 | PASS |

Claim B -- native beats the proxy: answered right, R3 against the side-channel
| criterion | R3 | side-channel | bar | |
|---|---:|---:|---|---|
| B1 sad | 11 of 13 | 7 of 13 | at least 7 | PASS |
| B1 fear | 11 of 13 | 7 of 13 | at least 7 | PASS |
| B1 anger | 9 of 13 | 11 of 13 | at least 11 | FAIL |
| B2 overall | 31 of 39 | 25 of 39 | more, and sign test p < 0.05 | FAIL |
  fixed 11: ang-t02, ang-t04, fea-t06, fea-t07, fea-t08, fea-t10, sad-t02, sad-t04, sad-t06, sad-t07, sad-t10
  broken 5: ang-t01, ang-t05, ang-t06, ang-t12, sad-t08
  exact two-sided sign test on the 16 pairs that differ: p = 0.210

PR-006 Claim A: SUPPORTED: R3 ranks at least 12 of 13 right for every emotion
PR-006 Claim B: NOT SUPPORTED: failed B1 anger, B2

| pair | p(Yes) neutral | p(Yes) loaded | R3 answered | R3 ranked | side-channel | R3 heard the sentence |
|---|---:|---:|---|---|---|---|
| sad-t01 | 0.21 | 1.00 | right | right | right | yes |
| sad-t02 | 0.23 | 0.99 | right | right | wrong | yes |
| sad-t03 | 0.16 | 1.00 | right | right | right | yes |
| sad-t04 | 0.06 | 1.00 | right | right | wrong | yes |
| sad-t05 | 0.22 | 0.98 | right | right | right | yes |
| sad-t06 | 0.33 | 0.99 | right | right | wrong | yes |
| sad-t07 | 0.42 | 0.94 | right | right | wrong | yes |
| sad-t08 | 0.79 | 1.00 | **wrong** | right | right | yes |
| sad-t09 | 0.50 | 1.00 | right | right | right | yes |
| sad-t10 | 0.10 | 0.99 | right | right | wrong | yes |
| sad-t11 | 0.15 | 0.98 | right | right | right | yes |
| sad-t12 | 0.98 | 0.99 | **wrong** | right | wrong | yes |
| sad-t13 | 0.36 | 0.99 | right | right | right | yes |
| fea-t01 | 0.75 | 1.00 | **wrong** | right | wrong | yes |
| fea-t02 | 0.15 | 1.00 | right | right | right | yes |
| fea-t03 | 0.16 | 0.99 | right | right | right | yes |
| fea-t04 | 0.20 | 0.99 | right | right | right | yes |
| fea-t05 | 0.78 | 1.00 | **wrong** | right | wrong | yes |
| fea-t06 | 0.32 | 0.99 | right | right | wrong | yes |
| fea-t07 | 0.50 | 1.00 | right | right | wrong | yes |
| fea-t08 | 0.26 | 1.00 | right | right | wrong | yes |
| fea-t09 | 0.29 | 1.00 | right | right | right | yes |
| fea-t10 | 0.13 | 0.99 | right | right | wrong | yes |
| fea-t11 | 0.13 | 0.98 | right | right | right | yes |
| fea-t12 | 0.04 | 0.98 | right | right | right | yes |
| fea-t13 | 0.09 | 0.97 | right | right | right | yes |
| ang-t01 | 0.53 | 0.99 | **wrong** | right | right | yes |
| ang-t02 | 0.44 | 1.00 | right | right | wrong | yes |
| ang-t03 | 0.19 | 1.00 | right | right | right | yes |
| ang-t04 | 0.25 | 1.00 | right | right | wrong | yes |
| ang-t05 | 0.71 | 1.00 | **wrong** | right | right | yes |
| ang-t06 | 0.93 | 1.00 | **wrong** | right | right | yes |
| ang-t07 | 0.20 | 0.99 | right | right | right | yes |
| ang-t08 | 0.45 | 0.99 | right | right | right | yes |
| ang-t09 | 0.16 | 0.99 | right | right | right | yes |
| ang-t10 | 0.39 | 0.98 | right | right | right | yes |
| ang-t11 | 0.26 | 1.00 | right | right | right | yes |
| ang-t12 | 0.69 | 1.00 | **wrong** | right | right | yes |
| ang-t13 | 0.04 | 0.99 | right | right | right | yes |

R3's transcript of the neutral clip contains the sentence, word for word after normalizing: 39 of 39

saved to runs/prosody/pr006.json

66 s inside the function, 200 s end to end (queue, cold start and run): about $0.061 at the A10 rate. Modal's dashboard has the exact bill.
Stopping app - local entrypoint completed.
✓ App completed. View run at 
https://modal.com/apps/ramcharanmc/main/ap-iNyHmVzEjwpUL5yw48aspV
