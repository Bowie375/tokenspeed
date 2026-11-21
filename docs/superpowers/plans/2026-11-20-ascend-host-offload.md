# Ascend Host Offloading Implementation Plan

> **For agentic workers:** Use the executing-plans workflow inline; retain the existing scheduler and cache execution path.

**Goal:** Complete actual Ascend HBM/Host cache transfers, safe block reuse and restore, with reproducible Homework 3 acceptance on 0.8B and 4B including TP2.

**Architecture:** Reuse scheduler-owned L2 prefix caching and request-private retraction images. Device and Host blocks remain pinned until existing completion events permit release; TP cache completions converge through CacheOpHooks. Use Ascend DMA through the existing kernel boundary, not a model-private cache or serialized activation substitute.

**Tech Stack:** Python, PyTorch/torch-npu, existing C++ scheduler and cache executor.

**Spec:** `outputs/hw3.md`; invariants in `docs/design/cache-concepts.md`, `scheduler.md`, `event-loop.md`, `unified_path.md`.

## Global Constraints

- Preserve KV, FP32 recurrent and convolution state at the same accepted context boundary on both ranks.
- Keep existing services; use available chips and clean only owned processes.
- Missing Host state recomputes through ordinary admission; never claim a hit without actual H2D and native model continuation.
- Host allocation and snapshot budgets are explicit. Separate reserved HBM from reusable scheduler blocks.
- No report or publication is needed; source, tests, documentation and raw acceptance evidence are the deliverable.

## Review Focus

- Asynchronous copy completion versus reuse of source/destination blocks and metadata.
- Non-aligned natural state checkpoints and the exact replay frontier.
- Budget eviction while a transfer or suspended request pins a block.
- TP peers observing different completion times or cache frontiers.
- Repeated restore/miss/abort leaving leaked pins or request-private snapshot slots.

## Task 1: Establish the real Ascend transfer baseline

- [x] Run existing scheduler/cache behavior tests and a real NPU Host-enabled Engine probe.
- [x] Add real byte-roundtrip regression tests through HostTransferLane/HostCacheExecutor for discovered failures.
- [x] Fix only demonstrated device/transport/fence defects; verify both explicit direct and supported kernel-selection paths.

Files: `python/tokenspeed/runtime/cache/{l2,transfer}/`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/kvcache/host_transfer.py`, NPU-specific tests.

## Task 2: Expose lifecycle and transfer evidence

- [x] Record actual scheduler plans, completed cache operations, field byte counts and event timing in disposable experiment observers.
- [x] Obtain authoritative device/Host block gauges; add scheduler/runtime observability only where existing interfaces cannot report them.
- [x] Test pin/release, budget eviction, misses and repeated reuse through real scheduler classes, not an alternate cache implementation.

## Task 3: Native end-to-end acceptance

- [x] Add reusable `experiments/homework3` runner, workers, explicit configs and evidence validators.
- [x] Compare uninterrupted/cold recomputation with Host restore using real full-vocabulary logits/tokens and KV/SSM/conv state at aligned boundaries.
- [x] Exercise alternating sessions, forced device eviction, small-Host eviction/miss, repeated cycles and actual active-request retraction/restore.
- [x] Verify rank-identical frontiers and required copy completion on TP2 for 0.8B and 4B.

## Task 4: Capacity/performance and final verification

- [x] Run Host off/on, high/low reuse, and two Host budgets under equal device caps; retain recomputation and any capacity failure explicitly.
- [x] Save occupied/free blocks, Host allocated/used bytes, D2H/H2D bytes, hit/miss/eviction counts, transfer time, recompute and request latency/throughput.
- [x] Complete relevant CPU/NPU regressions, independent review and full pre-commit; document runnable commands and actual evidence.
