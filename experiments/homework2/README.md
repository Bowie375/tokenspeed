# Ascend tensor-parallel implementation and tests

This directory provides executable Homework 2 validation and benchmarks for
Qwen3.5-0.8B, 4B, 9B and 27B. It reuses TokenSpeed's real TP execution path;
it does not emulate parallel inference or replace kernels with CPU formulas.
The tests establish rank-local cache/state semantics needed by Homework 3.
See [VALIDATION.md](VALIDATION.md) for the executed acceptance evidence.

## Environment and checkpoints

Activate the project Python environment, source CANN, and run from the repository
root. Select two available chips. Existing services are never stopped.
The first selected chip runs TP1; the pair is sorted for TP2 because this Ascend
runtime rejects descending visible-device lists. The manifest records both mappings.

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PYTHONPATH="${PWD}:${PWD}/python:${PWD}/tokenspeed-kernel/python:${PWD}/tokenspeed-kernel-npu/python:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=6,7  # choose an available pair
export OMP_NUM_THREADS=8
python -m experiments.homework2.download \
  --models Qwen3.5-4B Qwen3.5-9B Qwen3.5-27B --destination outputs/models
```

Downloads use pinned checkpoint revisions and retain local Hugging Face metadata.
The three larger checkpoints require approximately 78.4 GiB in total, plus
experiment artifacts. No Python dependencies are installed by these tools.

## Correctness

```bash
python -m pytest -q \
  test/runtime/distributed/test_hccl_backend.py \
  test/runtime/distributed/test_hccl_collectives.py \
  test/runtime/test_logits_processor.py \
  test/experiments/test_homework_tensor_parallel.py

python -m experiments.homework2.run \
  --model outputs/models/Qwen3.5-0.8B --devices "$ASCEND_RT_VISIBLE_DEVICES" \
  --config experiments/homework2/configs/baseline.json \
  --task correctness --reference hf

# Repeat for Qwen3.5-4B, Qwen3.5-9B and Qwen3.5-27B:
python -m experiments.homework2.run \
  --model outputs/models/Qwen3.5-4B --devices "$ASCEND_RT_VISIBLE_DEVICES" \
  --config experiments/homework2/configs/baseline.json \
  --task correctness --reference none
```

Every invocation creates a unique `outputs/homework2/runs/` directory and runs
TP1 followed by TP2 in separate disposable Engine processes. The explicit JSON
configuration covers lengths 1/7/65/257, 16 outputs, batched requests and repeated
fresh requests. The 27B TP1 run has an explicit reduced-context/concurrency override; this is
capacity validation, not a claim of an equal-resource performance comparison.

`validation.json` reports every acceptance check and exits nonzero on failure.
Both TP2 ranks must be present. Checks include:

- loaded projection/embedding shard shapes and NPU parameter placement;
- one real prefill followed by 15 cached single-token decode steps;
- exact canonical positions and exact free-running peer tokens/inputs;
- TP1/TP2 full-vocabulary logits with fixed numerical criteria;
- actual KV writes from the first full-attention layer, local KV-head counts,
  and KV concatenation compared with TP1;
- first-GDN-layer FP32 recurrent state and raw convolution history, restoring
  global Q/K/V channel order before comparison;
- real two-request batching, fresh-request isolation and observed physical
  state-block reuse on every rank.

Cross-TP KV values follow earlier TP reductions and use the existing decoder
activation envelope (`atol=0.08, rtol=0.05, relative_l2=0.06, rmse=0.03`).
Same-TP batch/fresh-request isolation uses the tighter lifecycle envelope
(`0.03, 0.01, 0.02, 0.01`). Full logits use `0.6, 0.03, 0.06, 0.18`; first-layer
SSM uses `0.02, 0.05, 0.08, 0.02`. All four conditions must hold and nonfinite
values fail. Every saved comparison records its numerical limits. These are
explicit acceptance contracts, not promises of topology-invariant BF16 bits.

Whole-model outputs cover the complete decoder; detailed state captures cover
the first GDN and first full-attention layer. They are not a claim of explicit
per-layer state inspection throughout all layers.

Teacher-forcing jobs supply identical continuation IDs after sampling on every
rank, while preserving real model/cache work. Separate free jobs retain native
greedy sampling. A permitted BF16 near-tie is identified only at the first
aligned divergence against TP1/HF, with the reference Top-5 and one-ULP gap
recorded. It is distinct from exact agreement and never permits different
free-running tokens between the two ranks of the same deployment.

The HF CPU reference dtype is explicit (`reference_dtype` in JSON); the provided
configuration uses FP32. Model weights and activations on NPU remain BF16 with
FP32 recurrent state. To regenerate only the independent reference and reanalyze
already completed native captures:

```bash
python -m experiments.homework2.reference \
  --directory outputs/homework2/runs/EXISTING_RUN --dtype float32
```

Reference dtype/device are saved separately in `reference_settings.json`.
To reanalyze saved native evidence and preserve its prior status history:

```bash
python -m experiments.homework2.reanalyze \
  --directory outputs/homework2/runs/EXISTING_RUN \
  --reference none --communication check
```

`--communication unavailable` explicitly limits reanalysis to numerical/state
checks for legacy captures recorded before communication observation existed.
New correctness runs always require the communication validation as well.
The independent HF reference command regenerates numerical reference evidence;
it does not replace communication acceptance.
Unavailable captures, missing steps, failed numerical gates, absent real batches
or unobserved physical reuse do not count as passing.

## Actual HCCL validation and communication timing

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  --module experiments.homework2.collectives \
  --directory outputs/homework2/collectives --warmups 10 --repetitions 100
```

To validate saved model communication counts, shapes, payload bytes and peer
agreement independently:

```bash
python -m experiments.homework2.communication \
  --directory outputs/homework2/runs/EXISTING_RUN
```

This writes `communication_validation.json` and fails on missing observations.
The collective microbenchmark above tests actual peer values for AllReduce, AllGather along multiple axes,
ReduceScatter and unequal/empty token shards. It benchmarks the five explicit
BF16 shapes in `collectives.py`, resetting inputs before every iteration to
avoid overflow from repeated reductions. Both rank JSON files must report
`arithmetic_passed=true`. Timing uses pre/post NPU synchronization. Payload
bandwidth is **per-rank payload bytes / measured time**, in decimal GB/s,
without claiming a physical-link bus bandwidth or multiplying bytes by eight.

## Comparable performance

```bash
python -m experiments.homework2.run \
  --model outputs/models/Qwen3.5-0.8B --devices "$ASCEND_RT_VISIBLE_DEVICES" \
  --config experiments/homework2/configs/baseline.json \
  --task performance --reference none
```

The baseline compares 128/32 and 512/16 input/output lengths, three measured
repetitions, four sequential requests and four concurrent requests per trial.
Each shape is warmed first. TP1 and TP2 retain the same model, fixed-length
EOS policy, greedy sampling, eager mode and resource settings. Performance
workers do not install correctness tensor-capture/teacher-forcing hooks.

TTFT uses first token-count growth; first nonempty text arrival is retained
separately. TPOT requires a first event containing exactly one token and uses
`(last_token_time-first_token_time)/(N-1)`. Coalesced streams or missing required
latency values are invalid, even when the requested output count was reached.
Concurrent means four clients submit together; cache admission may queue requests,
so this does not promise four simultaneously resident device requests. The logical
performance preflight checks that one request fits.
Throughput divides total generated tokens by the complete trial wall time;
raw request events and completion counts are preserved in `performance.jsonl`.

Performance acceptance requires every trial, request, latency and each rank's
actual model/memory evidence. `performance_summary.json` contains the median
request latencies and median trial throughput; the original events remain intact.
Saved results can be independently rechecked with:

```bash
python -m experiments.homework2.performance \
  --directory outputs/homework2/runs/EXISTING_PERFORMANCE_RUN
```

A separate memory request records allocator allocated/reserved/peak values in
both device processes. Reading a small control file at forward boundaries adds
some instrumentation overhead; no per-step synchronization or tensor snapshots
are added to the primary performance pass. Memory includes preallocated pools,
weights and workspaces and does not equal device-wide `npu-smi` consumption.
Use an idle, exclusive pair for publication-quality timing; shared-chip results
must be identified as such. Capacity savings and latency improvement are
separate outcomes. Do not compare timings with unequal resource overrides;
use `configs/large_fair_performance.json` for an equal-budget 27B comparison.

The logical cache cap must cover all state groups, including live and checkpoint
blocks. The supplied 32K cap admits the two-request 27B correctness batch; an
8K cap produces only five parents and cannot admit even one request. `capacity.py`
checks the actual recipe and scheduler page rules before loading weights and
saves `cache_admission.json`. Available HBM remains a runtime check.

## Code and state ownership

- `run.py`: explicit configurations, private bootstrap, worker lifecycle and exit status.
- `worker.py`: real sequential/batched inference and streaming workload execution.
- `hooks.py`: process-local per-rank observation and optional teacher forcing via
  the existing Homework 1 capture; no new runtime state owner.
- `validation.py`, `analyze.py`: fixed numerical criteria, state reconstruction,
  cross-rank and request-lifecycle acceptance.
- `measurements.py`: token-based TTFT and required latency availability.
- `collectives.py`: real HCCL arithmetic and synchronized microbenchmarks.
- `communication.py`: actual model collective payloads and runtime callers in correctness captures.
- `capacity.py`: host-only logical admission using the real cache recipe and scheduler.
- `reference.py`: independent HF CPU computation.
- `reanalyze.py`: saved-evidence acceptance with explicit communication scope and status history.
- `performance.py`: complete performance acceptance, per-rank evidence and metric summaries.
- `download.py`: pinned model retrieval.

The runtime fixes use standard HCCL tensor collectives and avoid CUDA-only
capture queries for NPU logits. KV and GDN state stay in the existing common
cache groups; no model-private Host cache, transfer policy, eviction or restore
is introduced here. Homework 3 should extend the cache subsystem while retaining
these same rank-local state partitions and canonical accepted positions.
