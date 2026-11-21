# Native validation evidence

These are implementation acceptance results, not a homework report. Raw responses, tensors, transfer rows and launch settings remain under `outputs/homework3/`.

## Functional coverage

| Model | TP | Workload | Result |
| --- | --- | --- | --- |
| 0.8B | 1, 2 | Aligned/unaligned Host restore, three alternating sessions, repeated cycles | Strict matched-path logits and states pass |
| 4B | 1, 2 | Same native workload | Strict matched-path logits and states pass |
| 0.8B | 1 | Small Host budget, eviction, miss and recompute | Strict cold/matched-path comparisons pass |
| 0.8B | 1, 2 | Active retraction and restore | Twenty continuation steps match uninterrupted execution bit-for-bit |

Functional verification checks the actual transferred bytes of all declared cache fields; model captures additionally compare full-vocabulary logits and first-layer GDN/convolution states. Device-index clears return all allocator blocks; actual block IDs recur on subsequent transfers. Final pins and active allocations are zero.

## Fixed-capacity performance

All requests generate eight tokens. Each access pattern includes 12 requests, including measured cold starts. Host budgets are multiples of local Device parent capacity.

| Model | Device parents | Host ratio | Pattern | Prefix hits / 12 | Median request ms | Output tokens/s | Prefill input tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B | 8 | 0 | high | 0 | 3428.6 | 2.353 | 3096 |
| Qwen3.5-0.8B | 8 | 0 | low | 0 | 3307.3 | 2.419 | 3096 |
| Qwen3.5-0.8B | 8 | 0.5 | high | 0 | 3446.6 | 2.339 | 3096 |
| Qwen3.5-0.8B | 8 | 0.5 | low | 0 | 3309.0 | 2.417 | 3096 |
| Qwen3.5-0.8B | 8 | 4 | high | 6 | 2700.5 | 3.386 | 1944 |
| Qwen3.5-0.8B | 8 | 4 | low | 0 | 3313.1 | 2.415 | 3096 |
| Qwen3.5-4B | 16 | 0 | high | 0 | 8169.2 | 0.981 | 3096 |
| Qwen3.5-4B | 16 | 0 | low | 0 | 8007.0 | 0.999 | 3096 |
| Qwen3.5-4B | 16 | 0.5 | high | 0 | 8157.6 | 0.983 | 3096 |
| Qwen3.5-4B | 16 | 0.5 | low | 0 | 8001.9 | 1.000 | 3096 |
| Qwen3.5-4B | 16 | 4 | high | 6 | 4769.7 | 1.676 | 1560 |
| Qwen3.5-4B | 16 | 4 | low | 0 | 7992.6 | 1.001 | 3096 |

Large Host budgets reduce recomputation for the repeated working set. The small budget thrashes and produces write traffic without useful restores. Low-reuse requests have no cache hits and no material latency improvement. The off cases finish through ordinary eviction/recomputation; no capacity failure was suppressed. These shared-hardware sequential results are acceptance measurements, not concurrency benchmarks.

## Artifacts and reproducibility

- Strict functional runs: `0.8b_tp1_functional_strict`, `0.8b_tp2_functional_strict`, `4b_tp1_functional_strict`, `4b_tp2_functional_strict`.
- Budget pressure: `0.8b_tp1_functional_small_strict`.
- Active recovery: `0.8b_tp1_retraction_final` with `0.8b_tp1_uninterrupted_final`; `0.8b_tp2_retraction_final` with `0.8b_tp2_uninterrupted_final`.
- Performance: `0.8b_tp1_performance_v2_{0,0.5,4}` and `4b_tp1_performance_{0,0.5,4}`; aggregate `performance_matrix.json`.
- Earlier probes and failed instrumentation runs remain separate and are excluded from this matrix.

See [README.md](README.md) for exact commands, counters, units, cache lifecycle and correctness criteria. Native binary evidence is intentionally ignored by Git.

Regression checks: **100 passed, 7 skipped** across Host orchestration, TP completion hooks, slot state, parser rejection cases, real scheduler gauges, and Ascend DMA/automatic transport round trips. Independent review findings in evidence validation were reproduced by five failing tests and fixed; all five now pass.
