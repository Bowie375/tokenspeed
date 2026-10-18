# Recorded Step 11 validation

The notebook executes successfully as an evaluation program. The model's eager
cached-decode correctness gate currently **fails**. Runtime/kernel code was not
changed as part of this evaluation work.

## Restart-and-run-all evidence

[Run report](../../outputs/homework1/correctness/runs/20261018T185139Z_947acc98/report.md),
[manifest](../../outputs/homework1/correctness/runs/20261018T185139Z_947acc98/manifest.json),
and [executed notebook](../../outputs/homework1/correctness/executed_11_correctness_final.ipynb).
These generated files remain ignored; the notebook reproduces them locally.

This run used raw lengths **1, 3, 129** and all four chat fixtures, four
teacher-forced outputs and at most four free outputs. It used the pinned local
Qwen3.5-0.8B checkpoint, BF16 eager Ascend execution with FP32 recurrent state,
and independent eager HF CPU FP32/BF16 references. Both implementations stopped
on the explicit EOS set `[248044, 248046]`. This is a declared subset of the
notebook's default length matrix, not a passing result for unexecuted lengths.

| Check | Passed | Failed |
| --- | ---: | ---: |
| Evaluation infrastructure tests | 12 | 0 |
| Existing NPU operator tests | 278 | 0 |
| CPU BF16/FP32 model/layer/state rounding baselines | 112 | 0 |
| Real GDN decoder output, convolution history and recurrent state | 84 | 0 |
| Whole-model raw/centered logits using identical captured positions | 56 | 0 |
| Exact canonical text position checks | 10 | 18 |
| Independent versus batched/fresh requests, intermediate outputs and states | 544 | 0 |

There were **806 passing and 18 failing tensor comparisons**. Real two-request
batches and physical state-block reuse were both observed. No reference/operator
checks were skipped or unavailable in this run. Free generation matched exactly
in **6/7** cases; the Chinese fixture first differed at output index 2, where the
reference top-two margin was approximately **0.010315**. After that point the
prefixes differ; later numerical values are not teacher-forced evidence.

The position failures are material. A three-token prompt is followed by native
positions `5, 6, 7`, whereas canonical cached text positions are `3, 4, 5`.
The reference receives captured native positions when comparing arithmetic,
while independent exact checks retain this semantic failure. Matching logits
under matched positions therefore does not clear the cached-position gate.

## Long-sequence boundary evidence

[2,048-token metrics](../../outputs/homework1/correctness/runs/boundary2048_5c377a3b/metrics.csv),
[failures](../../outputs/homework1/correctness/runs/boundary2048_5c377a3b/failures.json),
and [original native log](../../outputs/homework1/correctness/runs/20261018T183641Z_c4504b8f/native/worker.log).

The original full-matrix attempt reached the 2,048-token fixture and failed in
native attention decode: actual KV length **4097** exceeded its **4096**-entry
buffer. Only prefill and one cached-decode output were available, so the focused
report explicitly counts missing later steps and the runtime failure.

The available real GDN layer/state comparisons passed. Two independent CPU
BF16/FP32 recurrent-state rounding-baseline comparisons exceeded the predeclared
elementwise criterion: maximum absolute error was about **0.0398**, despite
relative L2 around **0.0120**. The thresholds were retained; long-state acceptance
criteria need further justification. Cached raw logits also failed (RMSE about
**0.659**, relative L2 about **0.189**), along with the exact position check.
The focused report records 13 passing and 7 failing/unavailable checks in total.

The notebook defaults to a larger explicit context buffer so it can collect
boundary diagnostics. This does not repair or suppress the independent position
check. Full default-matrix model correctness, graphs, prefix caching,
retraction/transfer, vision and performance are not established by these results.
