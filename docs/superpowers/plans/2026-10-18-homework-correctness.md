# Homework step 11 implementation plan

Use the approved notebook-plus-Python layout. Work inline in the shared checkout;
no worktrees or subagents. Keep runtime and kernel source unchanged. Evaluation
hooks belong only to the experiment worker and preserve original computations;
teacher forcing replaces sampled token IDs explicitly, never the recorded logits.

1. Add behavioral CPU tests for metrics (shape mismatch, nonfinite values, zero
   references, multidimensional errors), exact-length fixtures, and independent
   artifact directories. Implement `fixtures.py` and `metrics.py`. Commit after
   focused tests and `pre-commit run --all-files` pass.
2. Add an isolated Engine worker and `model.py` orchestration. Use pinned raw
   token IDs, BF16 eager NPU execution, CPU FP32 Hugging Face references, actual
   cached decode, and explicit shared continuation IDs. Capture vocabulary
   logits, first GDN layer inputs/outputs and real cache-owned state. Record
   original command, environment, dtype and every failure before asserting.
3. Implement `gdn.py` comparisons against the pinned HF decoder layer and
   independently maintained cache; transpose reference K,V state into runtime
   V,K order and compare only matching convolution history. Compare independent
   and batched A/B runs, and verify actual block reuse where observed rather
   than claiming reuse from repeated captions. Missing reuse becomes unverified.
4. Build a notebook with setup, fixed fixtures, existing operator tests, whole
   layer, request lifecycle, prefill logits, teacher forcing, free generation and
   summary cells. Helpers save tensor evidence and CSV/JSON independently of
   notebook display. Document restart/run-all and explicit one-chip selection.
5. Run CPU tests, operator regressions and hardware evaluation on a free chip;
   execute the notebook with a small validation workload. Record exact coverage
   and unsupported cases. Finish documentation and signed-off commits after
   exact repository hooks pass.

Acceptance: reusable evaluation code and independently runnable notebook cells;
identical saved model inputs for references and NPU; all three numerical metrics;
state snapshots and true cached teacher forcing; no fabricated/skipped successes;
no benchmark claims from diagnostic timing. Full long-length runs remain selectable
in notebook configuration, and the validation manifest lists the cases actually run.
