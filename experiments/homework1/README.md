# Homework 1 evaluation and performance experiments

Correctness evaluation uses one notebook backed by ordinary Python helpers. Source, fixtures,
comparisons and native captures have separate lifetimes:

- `notebooks/correctness.ipynb`: ordered, restartable evaluation and inspection.
- `evaluation/fixtures.py`: exact saved token IDs and environment provenance.
- `evaluation/metrics.py`: explicit limits, FP64 metrics, persistent tensor evidence.
- `evaluation/model.py`, `gdn.py`: native serving and independent HF comparisons.
- `evaluation/capture.py`, `worker.py`: disposable, process-local instrumentation.
- `test/experiments/`: behavioral tests for evaluation infrastructure.
- `outputs/homework1/correctness/runs/<run_id>/`: ignored generated evidence.

The native runner uses the real Engine, scheduler, kernels and managed caches.
Only teacher-forced sampling overrides the chosen next-token ID, after capturing
unmodified logits. The reference runs Transformers eager CPU FP32; serving uses
BF16 on one Ascend chip. This precision difference is explicit in the manifest.
No runtime or kernel implementation is replaced by the evaluation harness.

Comparisons persist tensors and failures before reporting status. Empty or skipped
suites are not passing evidence. Acceptance thresholds must be selected before
running adaptation comparisons; generation exact matches are reported separately.

## Run the notebook

Activate the project environment and CANN before starting the notebook kernel.
Choose one free chip using `ASCEND_RT_VISIBLE_DEVICES`; do not reuse a serving
process's chip. Open [correctness.ipynb](notebooks/correctness.ipynb), select
that environment's Python kernel, and restart/run all. The repository root is
found from the notebook's working directory. The checkpoint defaults to the
existing `outputs/models/Qwen3.5-0.8B`; set `HOMEWORK_EVAL_MODEL` to another local
copy of the same architecture when needed.

Optional environment inputs are explicit in the saved manifest:

| Variable | Default | Meaning |
| --- | --- | --- |
| `HOMEWORK_EVAL_LENGTHS` | `1,3,4,5,127,128,129,512,2048` | Exact raw-token lengths |
| `HOMEWORK_EVAL_INCLUDE_CHAT` | `1` | Include all four formatted chat fixtures |
| `HOMEWORK_EVAL_TEACHER_STEPS` | `4` | Prefill plus cached teacher-forced outputs |
| `HOMEWORK_EVAL_FREE_STEPS` | `16` | Maximum free greedy output length |

A small validation run can set lengths to `1,3,129` and free steps to `4`.
That is a documented subset, not evidence for unexecuted lengths. To select a
new matrix within a live kernel, edit the configuration cell and restart/run all.
To rerun only comparisons against existing captures, keep the existing `run` and
`native` variables and use a new `RunArtifacts` directory; comparison names are
unique and existing evidence is never overwritten.

## Read the results

`fixtures.json` stores exact IDs after chat formatting and explicit continuation
IDs. `manifest.json` identifies source/checkpoint revisions, small configuration
hashes, environment versions and all evaluation choices. `operators.xml` and
`operators.log` retain the real tests' failures and skips. `native/` contains the
worker command/log, Engine responses and each forward's input IDs, positions,
raw logits, actual sampled IDs, first-layer tensors, request slot and managed
state-block indices. Worker hooks also cover non-speculative decode's unified
`verify(N=1)` route; hooking prefill `sample()` alone misses decode.

`metrics.csv`, `metrics.json`, `tensors/` and `report.md` retain every numerical
comparison, including failures. Max absolute error, RMSE and relative L2 use CPU
FP64, shape equality is mandatory and nonfinite values fail. Compare raw logits
as well as centered diagnostics. CPU BF16-vs-FP32 model/layer/state baselines run
with fixed, predeclared limits; the helpers never adjust thresholds from results.

`generation.json` compares actual native sampled tokens with independent HF
CPU greedy generation, records EOS handling, first differing token and its
reference margin. Later values after a divergence use different prefixes.
`lifecycle.json` records real batched execution and physical block reuse; 32
bounded fresh-request attempts exercise allocator reuse without modifying it.
If reuse or actual batching is not observed, the final status stays incomplete.

This suite uses BF16 eager single-chip text inference with FP32 SSM state, no
prefix caching, no retraction pool, no speculation and no graph capture.
Retraction/transfer, graph replay, vision and performance require separate
validation. Capture synchronizations make these runs unsuitable as benchmarks.

Runtime failures are recorded in `native/status.json` and `failures.json`; available
captures still reach the final report. The longest fixture runs last to retain
shorter cases and lifecycle evidence if the backend fails at a length boundary.
The notebook selects `failure_policy="record"`; direct callers can explicitly
select `"raise"` instead. Source hashes identify uncommitted evaluation code.

Whole-model numerical references receive the captured native position IDs.
Separate zero-tolerance rows compare those positions with canonical text positions.
This distinguishes arithmetic/cache errors from an incorrect runtime position
sequence without masking either problem. A failed position check keeps the
complete result failed, even when the associated logit error is small.

Free generation explicitly uses the union of the tokenizer and model EOS IDs
for both implementations, saved in `generation_policy.json`. For this checkpoint
those IDs are 248046 (`im_end`) and 248044. Comparing each library's implicit
EOS default would incorrectly report a divergence after an already completed
arithmetic answer.

See [VALIDATION.md](VALIDATION.md) for recorded coverage, the fixed cached-position
bug, remaining state-precision failures and links to local evidence.


## Performance baseline

Open [performance.ipynb](notebooks/performance.ipynb), select the activated
project environment, and restart/run all. Select one free chip before starting
Jupyter. The notebook loads exact saved correctness token-ID fixtures, records the
checkpoint/source/configuration, and starts a disposable native Engine. It uses
BF16 eager text inference, FP32 recurrent state, greedy sampling, one request at
a time, no prefix caching, no overlap and no speculation. The main computation
runs on the selected NPU. Current numerical failures remain linked in the report;
benchmarking does not establish full correctness.

Reusable modules live under `performance/`:

- `config.py`: explicit workload/EOS/execution policy and exact Engine options.
- `measurements.py`: cumulative-stream TTFT/TPOT and median/min/max statistics.
- `hooks.py`: process-local complete-forward timing, NPU allocator and profiler capture.
- `worker.py`: fresh streaming requests, warmups and separate measurement passes.
- `profiling.py`: CPU/device event separation, name-based groups and overlap-aware busy time.
- `runner.py`, `results.py`: pinned artifacts, isolated process execution and reports.

Defaults cover the four 128/2048-input by 32/128-output combinations, three
warmups of every shape in each timing pass, then ten repetitions per case/pass.
The supported Engine `ignore_eos=True` enforces fixed output lengths; natural
requests can select `natural`, in which case actual lengths and finish reasons
remain explicit. Each call creates a new request and exercises managed state
initialization/allocation; no old recurrent state is manually replayed.

| Variable | Default | Meaning |
| --- | --- | --- |
| `HOMEWORK_PERF_MODEL` | local Qwen3.5-0.8B | Local checkpoint; no download |
| `HOMEWORK_PERF_FIXTURES` | newest compatible correctness run | Exact saved fixtures plus generation policy |
| `HOMEWORK_PERF_INPUT_LENGTHS` | `128,2048` | Two distinct saved raw lengths |
| `HOMEWORK_PERF_OUTPUT_LENGTHS` | `32,128` | Two distinct output targets |
| `HOMEWORK_PERF_WARMUPS` | `3` | Warmups per shape in each timing pass |
| `HOMEWORK_PERF_REPETITIONS` | `10` | Measured repetitions per case/pass |
| `HOMEWORK_PERF_EOS_POLICY` | `fixed_length` | Explicit `fixed_length` or `natural` |
| `HOMEWORK_PERF_PROFILE` | `1` | Collect short/long warmed CPU+NPU profiles |

For infrastructure smoke validation, use output lengths `2,3`, one warmup and
one repetition. The saved summary labels this as reduced coverage, not a
successful ten-repetition performance baseline. Keep the default matrix for the
actual homework report.

Primary timing uses the in-process streaming client, including IPC/scheduling
and detokenization, rather than HTTP transport. TTFT ends at the first nonempty
text event. TPOT is `(last token-count arrival - first token-count arrival)/(N-1)`;
a coalesced first event makes it unavailable. Stream chunks are never assumed
to correspond one-to-one with tokens. Primary timing adds no device barriers,
but lightweight scalar/allocator hooks add host overhead. After each request,
an unmeasured one-token control request flushes buffered evidence and profiles
on the same forward thread before the next client timer starts.

A separate synchronized pass measures complete `ModelExecutor._forward_step`
wall time, including model/logits/sampling and state work within that method,
with Python/launch overhead. Earlier input/attention-metadata preparation,
scheduler queueing and response transport are excluded. Prefill and decode
forward times are not labelled pure device-kernel durations, and their samples
are never mixed with primary latency samples. Every raw decode duration is
retained; prefill/decode forward counts must match actual generated counts.

Allocator peaks reset at the prompt boundary, after model/cache allocation.
They include weights, preallocated pools and workspace. Startup peaks and raw
allocated/reserved bytes are separate from warmed peaks. `npu-smi` device-wide
snapshots are collected at idle, after allocation and in separate memory
requests, with explicit collection intervals. Missing tools/APIs are unavailable,
not zero usage. Results use MiB (`2**20` bytes).

Profiling follows warmup and timing, in separate short/long requests. Raw
CPU/NPU timelines and recognized device-event groups support bottleneck
investigation; overlapping duration sums are not wall time, and slow decode
alone does not demonstrate a memory-bandwidth limit. Unclassified events and
missing device traces remain visible.

Each unique `outputs/homework1/performance/runs/<run_id>/` contains the manifest,
requested launch configuration, effective Engine server arguments, fixture copy,
linked correctness summary, every raw
request/finish/error, native forward evidence, samples/statistics CSV and JSON,
HBM snapshots, profiler traces, worker log/status and `report.md`. Failures and
timeouts retain available evidence. Completed prepared runs can also be launched
without IPython:

```bash
python -m experiments.homework1.performance.runner \
  --directory outputs/homework1/performance/runs/REPLACE_WITH_PREPARED_RUN \
  --timeout-seconds 7200 --failure-policy record
```

Use a new `prepare_run(...)` directory for each launch; evidence is never
silently overwritten. The Python preparation API is shown in the notebook.
