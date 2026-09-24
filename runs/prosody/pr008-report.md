Regenerated on 2026-09-24 from runs/prosody/pr008.json by the frozen r4probe.format_report and
format_criteria (the run's own printed output was overwritten; see PR-008's Deviations).

39 test prosody pairs · confirms PR-008 · r3-gpu and r4-gpu on NVIDIA L40S, 580.95.05 · commit f4da0bc

reply to the upset clip against the reply to its calm twin; 'same' is within +-0.5 st

| rung | emotion | pairs | wider | same | narrower | unmeasured | median d range (st) | median d pitch (st) | median d level (dB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R3 (Qwen2-Audio -> Kokoro) | SAD | 13 | 4 | 8 | 1 | 0 | +0.2 | -0.1 | +0.0 |
| R3 (Qwen2-Audio -> Kokoro) | FEA | 13 | 5 | 6 | 2 | 0 | +0.4 | -0.4 | -0.2 |
| R3 (Qwen2-Audio -> Kokoro) | ANG | 13 | 3 | 7 | 3 | 0 | -0.0 | -0.1 | -0.2 |
| R3 (Qwen2-Audio -> Kokoro) | all | 39 | 12 | 21 | 6 | 0 | +0.0 | -0.1 | -0.1 |
| R4 (Qwen2.5-Omni speaks) | SAD | 13 | 4 | 2 | 7 | 0 | -0.9 | -0.1 | -0.4 |
| R4 (Qwen2.5-Omni speaks) | FEA | 13 | 4 | 4 | 5 | 0 | -0.1 | -0.2 | -0.1 |
| R4 (Qwen2.5-Omni speaks) | ANG | 13 | 6 | 3 | 4 | 0 | +0.2 | +1.3 | +0.2 |
| R4 (Qwen2.5-Omni speaks) | all | 39 | 14 | 9 | 16 | 0 | -0.0 | +0.3 | -0.0 |

R3 (Qwen2-Audio -> Kokoro): reply words, median, to the calm clip 16, to the upset clip 17
R4 (Qwen2.5-Omni speaks): reply words, median, to the calm clip 17, to the upset clip 17
R3 (Qwen2-Audio -> Kokoro), the same clip answered twice: 1006_TIE_NEU_XX 0.159847, 1006_TIE_SAD_XX 0.213841
R4 (Qwen2.5-Omni speaks), the same clip answered twice: 1006_TIE_NEU_XX identical, 1006_TIE_SAD_XX identical

| pair | R3 (Qwen2-Audio -> Kokoro): d range (st) | R4 (Qwen2.5-Omni speaks): d range (st) |
|---|---:|---:|
| sad-t01 | +0.2 | +0.6 + |
| sad-t02 | +0.4 | +0.4 |
| sad-t03 | -0.4 | +0.7 + |
| sad-t04 | +0.8 + | +0.0 |
| sad-t05 | -0.5 | -1.8 - |
| sad-t06 | +12.3 + | -2.8 - |
| sad-t07 | +0.0 | -0.9 - |
| sad-t08 | -0.6 - | +0.6 + |
| sad-t09 | +0.0 | -1.6 - |
| sad-t10 | +0.6 + | -1.5 - |
| sad-t11 | +1.5 + | -4.3 - |
| sad-t12 | -0.4 | +1.8 + |
| sad-t13 | +0.3 | -2.9 - |
| fea-t01 | +1.4 + | +1.4 + |
| fea-t02 | -0.0 | -2.2 - |
| fea-t03 | +0.4 | +0.5 + |
| fea-t04 | +1.3 + | +2.7 + |
| fea-t05 | +1.0 + | -0.1 |
| fea-t06 | -0.3 | -1.0 - |
| fea-t07 | +2.0 + | +0.2 |
| fea-t08 | -0.1 | +4.0 + |
| fea-t09 | +0.4 | -0.3 |
| fea-t10 | -0.9 - | +0.3 |
| fea-t11 | +1.3 + | -0.9 - |
| fea-t12 | -2.5 - | -2.1 - |
| fea-t13 | +0.1 | -3.6 - |
| ang-t01 | -0.2 | +0.2 |
| ang-t02 | -1.5 - | -1.2 - |
| ang-t03 | +1.8 + | -1.0 - |
| ang-t04 | -0.1 | -1.9 - |
| ang-t05 | +1.5 + | +1.1 + |
| ang-t06 | +0.5 + | -0.0 |
| ang-t07 | +0.0 | -1.1 - |
| ang-t08 | -0.7 - | +4.5 + |
| ang-t09 | +0.1 | -0.1 |
| ang-t10 | -0.3 | +1.8 + |
| ang-t11 | -0.0 | +1.0 + |
| ang-t12 | +0.0 | +2.4 + |
| ang-t13 | -0.8 - | +2.1 + |
(+ wider, - narrower: beyond +-0.5 st)

| criterion | R4 | R3 | bar | |
|---|---|---|---|---|
| S1 R4's reply wider for the upset caller | 14 wider, 16 narrower, 9 same | | more wider, one-sided sign test p = 0.708 < 0.05 | FAIL |
| S2 R4 against R3, pair by pair | wider where R3 is not: 11 | wider where R4 is not: 9 | more, one-sided sign test p = 0.412 < 0.05 | FAIL |
| R3 for reference | | 12 wider, 6 narrower, 21 same | | |
  secondary, two-sided: R4's delivery moves consistently one way -- 14 wider against 16 narrower, p = 0.856

PR-008: NOT SUPPORTED: failed S1, S2
PR-008 secondary: no consistent direction
