# Normalization kernels

## Gated grouped RMSNorm on Ascend

The public `gated_rmsnorm` API dispatches the linear-attention output norm to
`tokenspeed_kernel_npu.ops.layernorm.gated_rmsnorm` on Ascend. Runtime
`linear.layernorm_gated.rmsnorm_fn` and `RMSNorm` retain their call contracts;
the GPU Triton implementation lives under `_triton/gated.py`, including its
PDL and static FP8 behavior.

Each last-dimension group has an independent RMS statistic. The scale vector
can cover the whole last dimension or one group, shared across all groups.
This operation multiplies by the weight directly. Qwen3.5 uses a shared
128-element weight and `norm(x) * silu(z)` for each value head. The same API
supports sigmoid gating, gating before normalization, or omitted gates.
Execution choices are explicit in the shared API.

The Ascend fused Triton kernel computes statistics, weight multiplication and
gates in FP32, then rounds once to the activation dtype (BF16, FP16 or FP32).
Both input and gate can be strided, including the gate view into the packed
projection. Leading dimensions are preserved; flattening may copy views whose
leading dimensions cannot be merged. Inputs and weights remain unchanged and
output storage is independent. Empty batches return without a kernel launch.
Row counts and strides do not specialize the kernel; group width and gate
choices are fixed model properties.

Ascend follows ordinary device stream ordering for both values of the GPU
`weights_independent` PDL promise. Static FP8 output requests raise
`NotImplementedError`; their scales are never silently ignored. Inference
only, with no backward implementation.

```bash
python -m pytest -q tokenspeed-kernel-npu/test/test_gdn_output_norm.py
```

Tests exercise the real runtime `RMSNorm` module against independent CPU FP64
references, including `[tokens,2048]` Qwen geometry, packed-projection gate
views, both gate orders, shared and per-feature weights, empty batches and
noncontiguous inputs. Compilation guards sweep token counts and strides.
