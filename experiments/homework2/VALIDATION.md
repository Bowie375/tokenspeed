# Executed acceptance evidence

This is a verification receipt for the reusable implementation, not a homework
report. Checkpoints and raw artifacts stay under ignored `outputs/`; commands,
configuration, acceptance logic and regression tests are source files.

## Implementation and regression checks

- Standard PyTorch HCCL tensor collectives replace unavailable aliases in direct
  and ragged paths. The regression uses actual two-process Gloo peer values,
  including BF16 vocabulary logits, and is also exercised on actual HCCL.
- CUDA stream-capture queries no longer run on NPU vocabulary logits.
- Launchers preserve existing services, clean only their own process groups,
  order TP2 visible chips correctly, and reject unequal performance resources
  even through model-directory symlinks.
- Real cache-recipe/scheduler preflight rejects insufficient logical capacity.
  For 27B, caps 8K/16K/32K yield 5/10/21 parents; a two-request batch needs 13.
  Original failures and the sorted-device retry remain in the run artifacts.
- Final CPU suite: **73 passed, 1 CUDA-only test skipped**
  ([log](../../outputs/hw2_final_cpu_tests.log)). NPU operator/construction suite:
  **280 passed** ([log](../../outputs/hw2_npu_operator_tests.log)).
- Full repository pre-commit, including newly staged files, passed on the
  submission branch. Main-branch protection is respected by committing on a
  feature branch. See [hook log](../../outputs/hw2_precommit.log).

## Four-model correctness

| Checkpoint | Accepted checks | Strict generation | Native evidence |
| --- | ---: | --- | --- |
| 0.8B | 2,404 | yes, including independent FP32 HF CPU reference | [run](../../outputs/homework2/runs/Qwen3.5-0.8B_correctness_20261119T182737Z_6fd45064/validation.json) |
| 4B | 2,476 | BF16 tie classification | [run](../../outputs/homework2/runs/Qwen3.5-4B_correctness_20261119T183332Z_efb19093/validation.json) |
| 9B | 2,478 | yes | [run](../../outputs/homework2/runs/Qwen3.5-9B_correctness_20261119T184407Z_0f305bfb/validation.json) |
| 27B | 2,782 | yes | [run](../../outputs/homework2/runs/Qwen3.5-27B_correctness_20261119T190430Z_b1730f0e/validation.json) |

All rows passed. Each covers TP1 plus both TP2 ranks, four input lengths
(1/7/65/257), 16 output steps, real cached decode, teacher-forced full logits,
free-running rank agreement, actual first-full-attention KV writes, first-GDN
SSM/conv reconstruction, real two-request batching, fresh-request isolation and
observed physical state-block reuse. All layers contribute to full-model logits;
detailed state inspection covers the first GDN and first full-attention layer.

For 4B `raw_1`, step index 2 has reference candidates 1001 and 348 with a zero
logit gap (BF16 ULP 0.0625). This is recorded as a tie rather than exact token
agreement. The two ranks within TP2 still agree exactly. The original over-tight
cross-TP KV comparison is retained as `validation_initial_strict_kv.json`; final
cross-TP KV uses the established decoder-activation envelope, separately from
same-TP request-isolation limits. Every saved comparison records its limits.

The 27B TP1 correctness baseline reduces context/concurrency to fit HBM;
performance uses a separate equal-resource configuration. Numerical reference
reanalysis preserves prior status in `reanalysis_history.jsonl`.

## Actual communication

The additional [0.8B observed run](../../outputs/homework2/runs/Qwen3.5-0.8B_correctness_20261119T191624Z_18d20333/validation.json)
passed 2,400 numerical/state checks and
[192 communication checks](../../outputs/homework2/runs/Qwen3.5-0.8B_correctness_20261119T191624Z_18d20333/communication_validation.json).
Every observed forward has **49 AllReduces + 1 vocabulary AllGather** on each
rank. Single-row hidden-state payloads are 2,048 bytes; the local BF16 vocabulary
payload is 248,320 bytes. Longer prefill payloads scale with actual input rows.
Peer operations, shapes, dtypes and byte counts agree exactly.

Actual HCCL arithmetic passed on both ranks, including unequal/empty token shards.
The five BF16 message shapes each have 10 warmups and 100 recorded repetitions:
[rank 0](../../outputs/homework2/collectives_verified/rank0.json),
[rank 1](../../outputs/homework2/collectives_verified/rank1.json).
Bandwidth is per-rank payload bytes divided by synchronized wall time, in decimal
GB/s; it is not measured link traffic. See the
[communication guide](../../docs/guides/ascend-tensor-parallel.md) for the model
call sites, head/weight/state partitions, and limits of microbenchmark attribution.

## Performance matrix

The matrix uses 128/32 and 512/16 input/output lengths; sequential and four-client
concurrent submission; three measured repetitions of four requests each. Each
completed model therefore has 96 measured requests across TP1/TP2. Acceptance
requires all outputs, usable token-based TTFT/TPOT, and post-load/post-forward
memory evidence from all three device processes. Concurrent clients can queue;
this does not promise four requests remain resident simultaneously.

Results are measured on shared hardware with existing services preserved; they
are functional performance evidence, not exclusive-hardware performance claims.
The 0.8B run used the recorded 16K cap before the provided default became 32K.
Within each model's performance comparison, TP1 and TP2 resources match. Saved
configurations, launch files and raw stream events are authoritative.

| Checkpoint | Recorded cap | Acceptance | Metrics |
| --- | ---: | --- | --- |
| 0.8B | 16384 | [passed; 96 measured requests](../../outputs/homework2/runs/Qwen3.5-0.8B_performance_20261119T185233Z_067b38ac/validation.json) | [summary](../../outputs/homework2/runs/Qwen3.5-0.8B_performance_20261119T185233Z_067b38ac/performance_summary.json) |
| 4B | 32768 | [passed; 96 measured requests](../../outputs/homework2/runs/Qwen3.5-4B_performance_20261119T190450Z_97830d07/validation.json) | [summary](../../outputs/homework2/runs/Qwen3.5-4B_performance_20261119T190450Z_97830d07/performance_summary.json) |
| 9B | 32768 | [passed; 96 measured requests](../../outputs/homework2/runs/Qwen3.5-9B_performance_20261119T192134Z_5d6567b3/validation.json) | [summary](../../outputs/homework2/runs/Qwen3.5-9B_performance_20261119T192134Z_5d6567b3/performance_summary.json) |
| 27B | 32768 | [passed; 96 measured requests](../../outputs/homework2/runs/Qwen3.5-27B_performance_20261119T191440Z_1ad5b415/validation.json) | [summary](../../outputs/homework2/runs/Qwen3.5-27B_performance_20261119T191440Z_1ad5b415/performance_summary.json) |

All four complete matrices passed: **384 measured requests**, no invalid or
missing requests, and actual memory/model evidence from every rank. The aggregate
[acceptance receipt](../../outputs/homework2/acceptance.json) references all runs.
For 27B, post-load allocated memory is 50.164 GiB for TP1 and 25.176 GiB per TP2
rank; post-forward and peak counters are retained separately in `memory.json`.


[Environment](../../outputs/homework2/environment.json) and
[hardware snapshot](../../outputs/homework2/hardware.txt) accompany per-run
checkpoint revisions, package versions, source hashes, launch parameters,
rank-local weights, stream events, and allocator counters.

## Homework 3 boundary

KV, SSM and convolution state remain owned by existing common cache groups and
partitioned by rank-local heads/channels. Canonical accepted positions are
checked through actual prefill and cached decode. This provides the execution
and state-isolation baseline for Host offloading; Host budgets, transfer,
eviction and restore are not implemented in this stage.

All owned model workers exited after validation. The original service PIDs remain
present in the [post-cleanup snapshot](../../outputs/homework2/hardware_after_cleanup.txt).
