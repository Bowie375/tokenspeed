# Homework 3: Ascend Host offloading

This directory contains executable acceptance and performance experiments for
`outputs/hw3.md`. It uses the runtime's real two-level cache, C++ scheduler,
pinned Host storage, and Ascend DMA transport. No replacement cache or simulated
offload is used. See [VALIDATION.md](VALIDATION.md) for measured results.

## Run

Use the project's installed Python environment and source your CANN environment.
The scheduler must be built from this checkout: the experiment needs the new
`host_pool_free_blocks`, `host_pool_cached_blocks`, and `host_cache_evictions`
bindings. Install the local scheduler with the project's normal development
installation procedure, rather than mixing a released extension with this source.

```bash
export PYTHONPATH="$PWD:$PWD/python:$PWD/tokenspeed-kernel/python:$PWD/tokenspeed-kernel-npu/python:${PYTHONPATH:-}"
python -m experiments.homework3.suite \
  --small-model outputs/models/Qwen3.5-0.8B \
  --large-model outputs/models/Qwen3.5-4B \
  --devices 6,7 --task all --output outputs/homework3/new-run
```

Select two available chips. The runner sorts physical IDs for Ascend TP mapping,
uses private worker process groups, and cleans up only those groups. Output
directories must be new, so previous evidence cannot be mistaken for this run.
`--task functional` and `--task performance` run the corresponding parts alone.

For a single case:

```bash
python -m experiments.homework3.run \
  --model outputs/models/Qwen3.5-0.8B --devices 6,7 --tp 2 \
  --task functional --host-ratio 4 --device-tokens 8192 \
  --force-retraction 0 --output outputs/homework3/single-case
python -m experiments.homework3.analyze outputs/homework3/single-case \
  --task functional --require-eviction no
```

`--host-ratio 0` disables Host L2. `0.5` and `4` select the two tested budgets,
as multiples of actual local Device LCM parent capacity. The budget applies per
rank; explicit `kvstore_size` in the runtime can instead select a byte budget.
Active-request retraction uses a separate bounded, non-evictable snapshot pool;
the suite enables it explicitly and compares with uninterrupted execution.

## Modules and evidence

- `run.py`: explicit launch settings and fixed workloads.
- `worker.py`: native Engine requests and quiescent Device-index controls.
- `hooks.py`: disposable observers in each rank, actual transfer events, byte
  verification, authoritative scheduler gauges and model-state captures.
- `analyze.py`: fails on missing evidence, incomplete generation, incorrect
  continuation, corrupted states, missing transfers, leaked pins, or different
  TP recovery positions.
- `summary.py`: rejects comparisons with unequal launch settings or resolved
  Device block capacity, then summarizes measured access patterns.
- `suite.py`: executes and validates the complete matrix.

Each run retains launch settings, exact token workloads, responses, per-rank
transfer rows, event timings, capacity gauges, and numerical captures. Large
binary tensors and model weights belong under ignored `outputs/`, not in Git.
`analysis.json` is only written as passing after its checks complete; a failed
CLI analysis records the error and exits nonzero.

## Cache lifecycle and safety

The cache identity is a chained prefix hash plus a computed token frontier,
cache group and page offset. The C++ `CacheCoordinator` owns both Device and
Host allocation, prefix matching, LRU eviction and request references. Runtime
code consumes its operations through `cache_ops_from_plan`; model-private
allocation is unnecessary.

| State | Granularity and frontier | Host handling |
| --- | --- | --- |
| Attention K/V | History blocks of the common cache arena, valid through computed position T | Copy each declared field's payload, excluding allocation padding |
| GDN recurrent state | State slots at natural computed checkpoint T; FP32 in these runs | Save/restore every state group's fields at that same T |
| Convolution history | History associated with the GDN state slot, BF16 | Copied with its state group; never independently matched at a different T |
| Other request-slot state | `SlotStateExporter` image for an actively retracted request | Bounded private image plus request identity; restored into the new request slot |

Normal immutable prefix writebacks pin their Device sources until completion.
`HostCacheExecutor` submits D2H on its write stream; completed events become
scheduler ACKs. Unreferenced Device entries can then be evicted/reallocated.
Active retraction uses stream-ordered stores ahead of the plan's zeroing and
reuse, with a fence covering both cache fields and slot-state export. A snapshot
combines published Host L2 pages with private tail pages; its pinned pages cannot
be evicted while suspended.

Demand prefix loads restore into allocated Device destinations. DMA consumer
events prevent a layer reading partially restored data. Active-request restores
return one completion covering both Host tiers and the slot image; the request
cannot resume before its ACK. `CacheOpHooks` intersects ACK tickets across TP
ranks before advancing their mirrored schedulers.

Host capacity pressure evicts only unpinned entries. An evicted checkpoint is a
normal prefix miss and goes through ordinary prefill recomputation. A private
snapshot cannot be silently evicted: insufficient snapshot capacity follows the
runtime's explicit abort policy. L2 budget and private snapshot budget are
reported separately.

## Correctness criteria

The functional workload covers aligned T=128 and unaligned T=129 checkpoints,
three distinct sessions alternating over three cycles, insufficient Host budget,
and repeated release/reallocation. A control command clears only the Device
prefix index while preserving Host L2, after requests and transfers are idle.
This deliberately forces demand restoration; performance workloads instead
create pressure naturally with a working set exceeding Device capacity.

Every actual copied cache field, including all layers' K/V, convolution and GDN
states, is checked byte-for-byte against compact Host storage in functional
runs. The check synchronizes only the disposable correctness worker. Sources
can then be overwritten and restored into reusable Device blocks. Independent
DMA/automatic-backend tests also exercise padded layouts, fresh destinations,
unchanged guard bytes, empty transfers and repeated metadata reuse.

Host continuation is compared against Device reuse at the same checkpoint:
full-vocabulary logits and captured first-layer convolution/GDN states must be
bitwise equal at every step. Output tokens must also equal the cold full-prompt
reference. A cold batched prefill and suffix-only prefill can have different BF16
rounding, so the cold result is not incorrectly required to match the warm
result's intermediates bit-for-bit. Misses use the cold reference. No teacher
forcing or replacement sampling is used.

Response `cached_tokens` gives the recovered frontier; the first captured
position range identifies the actual suffix processed. For T=129 followed by
a 130-token prompt, the suffix is the one new token at position 129. A cache
miss processes the complete prompt. Active retraction is checked against
uninterrupted execution at every subsequent computed position, with no extra
prefix replay in the tested cases. Recompute work is included in request time.

## Statistics and fair comparisons

The scheduler supplies empty/active Device parents, total Device parents, empty
Host parents, published Host children, pinned children and lifetime Host capacity
evictions. Parent and child counts are different units. Host occupied bytes are
occupied parents times compact parent bytes, including within-parent slack;
allocated Host bytes describe the entire pinned pool. Explicit flushes are not
capacity evictions. Per-rank post-response samples may straddle publication ACKs;
they are not an atomic cross-rank snapshot of published-child counts.

Transfer byte counts sum actual cache field payloads for submitted rows. The
alignment-padded serialized slot-image width is reported separately as
`slot_blob_serialized_bytes`; it is not mislabeled as additional cache DMA.
Device-event durations measure transfer-stream work, including consumer copies
and slot handling where present. Recovered positions and completion evidence
are checked on both ranks.

Performance runs disable numerical capture and byte verification. The same
Device cap, model, execution settings and access sequence are used with Host
off, small Host and large Host. Each access mode has four rounds over three
sessions, fixed 258-token prompts and eight generated tokens. High reuse repeats
sessions; low reuse changes their prefix each time. Median end-to-end request
time and sequential output-token throughput include all measured cold starts,
restores and recomputations, excluding model loading and the initial warmup.
Control commands are outside measured request time. `prefill_input_tokens` counts
all uncached prompt work; `recomputed_prompt_tokens` counts that work only for
identical prompts that already appeared in the same access pattern.

`hits`/`misses` count prefix reuse from response metadata, not necessarily Host
hits: H2D operations independently prove actual Host use. The off case can run
by evicting/recomputing, even though its Device pool cannot retain the whole
working set. HBM is preallocated; reclaimed block counts and reuse of actual
transfer destination IDs demonstrate reclamation without requiring `npu-smi`
total memory to drop. These are small sequential acceptance workloads on shared
hardware, not production concurrency or throughput claims.

## Regression checks

```bash
python -m pytest -q \
  test/runtime/test_host_pool_gauges.py \
  test/runtime/test_homework3_acceptance.py \
  test/runtime/test_host_executor.py \
  test/runtime/test_host_sync.py \
  test/runtime/test_l2_cache_hooks.py \
  test/runtime/test_slot_state.py
ASCEND_RT_VISIBLE_DEVICES=6 python -m pytest -q \
  test/runtime/test_ascend_host_roundtrip.py
```

The architecture and invariants remain those in
[`cache-concepts.md`](../../docs/design/cache-concepts.md) and
[`scheduler.md`](../../docs/design/scheduler.md). New scheduler counters expose
that ownership; they do not introduce another execution or allocation path.
