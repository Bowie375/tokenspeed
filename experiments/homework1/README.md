# Homework 1 numerical evaluation

Step 11 uses one notebook backed by ordinary Python helpers. Source, fixtures,
comparisons and native captures have separate lifetimes:

- `notebooks/11_correctness.ipynb`: ordered, restartable evaluation and inspection.
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
process's chip. Open [11_correctness.ipynb](notebooks/11_correctness.ipynb), select
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

See [VALIDATION.md](VALIDATION.md) for recorded coverage, the remaining cached
position failure, the 2,048-token boundary diagnostics and links to local evidence.
