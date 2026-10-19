# Recorded Step 11 validation

The prefill sampling-offset implementation now preserves the live accepted
cache frontier. The full default notebook matrix has **1,059 passing and five
failing tensor comparisons**. All position and whole-model logit comparisons
pass; the strict long-sequence state gate remains failing.

## Full-matrix evidence after the frontier fix

[Run report](../../outputs/homework1/correctness/runs/20261019T160802Z_620a51ca/report.md),
[summary](../../outputs/homework1/correctness/runs/20261019T160802Z_620a51ca/summary.json),
and [executed notebook](../../outputs/homework1/correctness/executed_11_correctness_20261019T160755Z.ipynb).
Generated evidence remains ignored; the notebook reproduces it locally.

This run used raw lengths **1, 3, 4, 5, 127, 128, 129, 512, 2048** and all four
chat fixtures, four teacher-forced outputs and at most 16 free outputs. It used
the pinned local Qwen3.5-0.8B checkpoint, BF16 eager Ascend execution with FP32
recurrent state, and independent eager HF CPU FP32/BF16 references. Both
implementations used the explicit EOS set `[248044, 248046]`.

| Check | Passed | Failed |
| --- | ---: | ---: |
| Frontier regression and evaluation infrastructure tests | 16 | 0 |
| Existing NPU operator tests | 278 | 0 |
| CPU BF16/FP32 whole-model rounding baselines | 52 | 0 |
| CPU BF16/FP32 layer/state rounding baselines | 152 | 4 |
| Real GDN decoder output, convolution history and recurrent state | 155 | 1 |
| Exact canonical text positions | 52 | 0 |
| Whole-model raw/centered logits | 104 | 0 |
| Independent versus batched/fresh outputs and states | 544 | 0 |

No checks were skipped or unavailable. Real two-request batches and physical
state-block reuse were observed. Free generation matched in **12/13** cases,
compared with **9/13** before the fix. The Chinese fixture first differs at
output index 2, where the reference top-two logit margin is approximately
**0.010315**. Subsequent outputs use different prefixes and are not
teacher-forced arithmetic comparisons.

## Fixed cache-frontier corruption

On the installed Ascend stack, functional `Tensor.index_copy()` unexpectedly
modifies its input as well as returning a new tensor. Building prefill sampling
offsets therefore changed `RuntimeStates.valid_cache_lengths` before accepted
frontier advancement. A prompt of length `L` started decode at `2L - 1` instead
of `L`; a 2,048-token prompt started at 4,095.

`_with_prefill_sample_offsets` now clones the frontier before its indexed
write. CPU/NPU regression tests exercise noncontiguous request slots, existing
prefix lengths, accepted advancement and next-decode position preparation.
The full notebook removes all **36 position failures and 12 cached-logit
failures** seen in the otherwise identical pre-fix full-matrix run
`20261019T155515Z_22f80dfb`.

## Remaining numerical acceptance failures

Four failures are CPU BF16-versus-FP32 reference state comparisons for the
2,048-token fixture. Their maximum absolute errors are approximately
**0.0397–0.0415**, with relative L2 around **0.012**. These execute independently
of the Ascend implementation.

The fifth is native first-layer recurrent state at the fourth output of that
fixture: maximum absolute error **0.026789**, RMSE **0.000248**, relative L2
**0.00786**. One of 262,144 state elements exceeds the fixed
`0.02 + 0.05 * abs(reference)` criterion. The notebook retains its original
thresholds and reports this failure explicitly; passing logits does not clear
the state gate.

An additional exact-input diagnostic captured the first layer's real Q/K/V,
gates, initial states and kernel results for this fixture. Independent CPU FP64
recurrence gave maximum state errors of **4.23e-5** after prefill and at most
**2.56e-7** for each cached step. These are far below the FP32 model-reference
difference; the evidence points to upstream BF16 rounding rather than an
incorrect single-token recurrence. This diagnostic does not replace the failed
whole-layer criterion. See
[exact-input metrics](../../outputs/homework1/correctness/recurrence_diagnostic/same_inputs_fp64.json).
The diagnostic directory also retains its worker log, captured tensors and
standalone scripts.

A subsequent [component trace](../../outputs/homework1/correctness/state_trace/report.md)
identifies BF16 rounding after convolution/SiLU, especially V, as the dominant
source at the failing element. Actual NPU kernel replay reproduces all saved
BF16 states exactly. Promoting the same quantized projections before
convolution and retaining FP32 convolution outputs clears all state-element
checks in this fixture's four steps, reducing final maximum error to **0.01111**.
This is an isolated precision experiment; it has not changed production
execution or established full-model correctness with that precision policy.

The broader `test/runtime/test_zero_output_prefill.py` run has five failures in
existing orchestration stand-ins (40 tests pass). The same five failures
reproduce using unchanged HEAD sources in a temporary directory; they are not
introduced by the frontier fix. See
[baseline log](../../outputs/homework1/correctness/runs/20261019T160802Z_620a51ca/baseline_zero_output.log).

Graphs, prefix caching, transfer/retraction, vision and performance are outside
this eager correctness run.


## Step 12 performance workflow validation

The [performance notebook](notebooks/12_performance.ipynb) completed a real
single-visible-NPU smoke run on the same pinned checkpoint:
[report](../../outputs/homework1/performance/runs/20261019T165601Z_3d82ccc9/report.md),
[summary](../../outputs/homework1/performance/runs/20261019T165601Z_3d82ccc9/summary.json),
and [executed notebook](../../outputs/homework1/performance/executed_12_performance_smoke_20261019T165600Z.ipynb).

This run used input lengths **128 and 2048**, deliberately reduced fixed output
lengths **2 and 3**, one warmup per shape/pass, and one measured repetition per
case/pass. All **20 recorded native requests** were valid: eight warmups, eight
measured latency/forward requests, two memory requests and two profile requests.
Unmeasured one-token control requests exported buffered evidence outside timing.
Each real request executed one complete uncached prefill and exactly `N-1`
one-token decode forwards. No correctness or sampling overrides were installed.

The real short/long profiles both contain Ascend hardware events. Name-based
device-duration grouping identifies GDN as the largest observed category in
these two diagnostic traces; this does not establish a bandwidth or other
hardware limit. Profiler durations are separate from primary timing and account
for overlap explicitly. Framework allocated/reserved/peak counters and raw
idle/allocated/workload `npu-smi` snapshots are retained. The snapshot collection
can outlast these tiny output sequences, so those device-wide snapshots are
not claimed as precisely timed workload HBM peaks.

All **32 experiment infrastructure tests** passed, including 20 new performance
helper tests. Repository hooks passed, and independent review checked timing
isolation, partial errors, stream token counts and profile interpretation.
The captured Python source hashes match the implementation used for this run.

This is workflow smoke coverage, **not** the default 128/2048-input by
32/128-output, ten-repetition performance baseline. The saved completeness
flags remain false for that baseline. The notebook retains those full defaults
for the homework measurement. The five linked Step 11 state-precision failures
also remain unresolved by this performance work.
