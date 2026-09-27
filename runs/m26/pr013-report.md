SYNTHETIC SPEECH: text-to-speech, one model at several numeric precisions. Research code; no clinical validity is claimed or implied.

reference: onnx-fp32; 2 voices x 24 sentences; within = PR-009's tolerance (3 Hz; 10%)

| rung | size MB | audio | pitch offset | pitch range | rate | pauses | WER raw / norm | RTF |
|---|---:|---:|---|---|---|---|---:|---:|
| onnx-fp32 | 325.5 | 48 of 48 | 48 of 48 (median 0.00, max 0.00) | 48 of 48 (median 0.00, max 0.00) | 48 of 48 (median 0.00, max 0.00) | 48 of 48 (median 0.00, max 0.00) | 1.8% / 1.0% | 0.33 |
| onnx-fp16 | 163.2 | 42 of 48 | 41 of 42 (median 0.42, max 3.64) | 42 of 42 (median 0.01, max 0.08) | 42 of 42 (median 0.00, max 0.04) | 41 of 42 (median 0.00, max 0.25) | 2.2% / 1.3% | 0.33 |
| onnx-q8 | 92.4 | 48 of 48 | 12 of 48 (median 3.68, max 8.45) | 39 of 48 (median 0.07, max 0.16) | 48 of 48 (median 0.01, max 0.06) | 45 of 48 (median 0.02, max 0.30) | 2.1% / 1.2% | 1.36 |
| torch | — | 48 of 48 | 45 of 48 (median 0.79, max 5.27) | 47 of 48 (median 0.02, max 0.13) | 48 of 48 (median 0.00, max 0.05) | 46 of 48 (median 0.00, max 0.30) | 1.8% / 1.1% | 0.38 |

noise floor, torch against torch (4 repeats a sentence): deviation median / 95th percentile / max
  f0_offset_hz  0.278 / 1.438 / 3.066   (192 pairs)
  f0_dev_scale  0.008 / 0.037 / 0.071   (192 pairs)
  rate          0.000 / 0.003 / 0.008   (192 pairs)
  pause_scale   0.000 / 0.024 / 0.026   (192 pairs)

renders that are not audio: 6
  onnx-fp16 af_sarah v02: 139200 of 139200 samples not a number
  onnx-fp16 af_sarah v06: 131400 of 131400 samples not a number
  onnx-fp16 af_sarah v12: 133200 of 133200 samples not a number
  onnx-fp16 af_sarah v18: 141000 of 141000 samples not a number
  onnx-fp16 am_adam v06: 122400 of 122400 samples not a number
  onnx-fp16 am_adam v13: 135000 of 135000 samples not a number

PR-013, the registered criteria (docs/preregistration.md):
  Claim A: at the 8-bit rung the pitch has left its guardrail while the words have not
  A1  8-bit pitch offset within 3 Hz on fewer than 90% of renders: 12 of 48 (25.0%)   PASS
  A2  its mean offset's 95% interval excludes zero and lies beyond the noise floor (1.44 Hz): -3.96 Hz, -4.36 to -3.58   PASS
  A3  normalized WER, 8-bit minus fp32, interval's upper end at most +2.0 points: +0.24 points, +0.00 to +0.61 (10 against 8 errors in 820 words)   PASS
  Claim B: fp16 on a CPU is a usable rung
  B1  fp16 returns audio for every render: 42 of 48   FAIL
  B2  its renders that are audio: pitch offset within 3 Hz on at least 90%, range within 10% on at least 80%: 41 and 42 of 42   PASS
PR-013, Claim A: SUPPORTED
PR-013, Claim B: NOT SUPPORTED -- failed B1

saved -> runs/m26/pr013.json; renders: results/m26/pr013/; 1185 s on the CPU
