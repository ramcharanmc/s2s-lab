Modal Tesla T4, 610.57.04 · commit 65c833c · 200 clips
### small.en (small.en)

| cohort | turns | ref words | raw WER | normalized WER |
|---|---:|---:|---:|---:|
| accented | 50 | 884 | 26.7% | 23.5% |
| clean | 50 | 846 | 3.5% | 3.3% |
| disfluent | 50 | 1039 | 9.9% | 7.6% |
| slow | 50 | 846 | 3.7% | 3.5% |
| **all** | 200 | 3615 | 11.1% | 9.6% |
recognizer time p50 144 ms, load 1.9 s

### ft2 (/ft2/export/export-runs_train-f37d0df-20260926-235222_csv_version_0_adapters_step-000900-float16/ct2)

| cohort | turns | ref words | raw WER | normalized WER |
|---|---:|---:|---:|---:|
| accented | 50 | 884 | 18.3% | 17.6% |
| clean | 50 | 846 | 3.0% | 2.8% |
| disfluent | 50 | 1039 | 2.3% | 2.4% |
| slow | 50 | 846 | 3.2% | 3.1% |
| **all** | 200 | 3615 | 6.6% | 6.4% |
recognizer time p50 145 ms, load 1.9 s

PR-011, the registered criteria (docs/preregistration.md), 200 eval-set clips:
  S1  raw WER on the 50 disfluent clips: base 9.9%, LoRA 2.3%; -7.6 points, 95% interval -9.6 to -5.7   PASS
  S2  fillers kept on those clips: LoRA 29 of 29 (100.0%; base 1), bar 90%   PASS
  S3  normalized WER on the 150 other clips: base 10.4%, LoRA 8.0%; -2.3 points, interval -3.8 to -0.9, allowed up to +1.0   PASS
PR-011: SUPPORTED
saved -> runs/ft2/pr011.json
67 s inside the function, 76 s end to end (queue, cold start and run): about $0.012 at the T4 rate. Modal's dashboard has the exact bill.
