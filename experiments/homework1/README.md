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
