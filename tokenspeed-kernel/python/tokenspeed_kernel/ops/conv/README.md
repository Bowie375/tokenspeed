# Convolution kernels

## Ascend causal convolution

The shared `causal_conv1d_fn` and `causal_conv1d_update` APIs select the
Ascend implementations in `tokenspeed_kernel_npu.ops.causal_conv1d`.
Runtime compatibility wrappers retain their existing signatures and defaults.
Packed prefill consumes `[channels,tokens]`; single-token decode overwrites
`[requests,channels]` or `[requests,channels,1]` with the convolution output.
Weights are depthwise `[channels,width]`, oldest tap first, with width 2–5.
FP16, BF16 and FP32 inputs support optional bias and SiLU, computed in FP32.

History belongs to the existing cache pool. Both `[slots,channels,history]`
and `[slots,history,channels]` views are supported, including strided storage.
The first `width-1` positions hold raw inputs in chronological order. Prefill
uses supplied scheduling maps and boundaries; omitted initial-history flags
mean logical zero history. Short chunks retain the needed old prefix, empty
sequences leave state untouched, and padded slots yield zero output.
A separate state-update launch ensures every prefill output reads old history.

Decode can read a checkpoint slot and write a different live slot. Negative
or explicitly padded reads leave input and state untouched; negative writes
skip state storage. Live destinations must be unique and cannot overwrite
another request's input checkpoint. Updating a request's own slot is valid.
Index and boundary values are caller-owned metadata contracts. Optional
`validate_data=True` checks contents with a host synchronization; ordinary
execution validates tensor metadata without reading device values on the host.

Circular history, multi-token/speculative updates and tree updates are not
supported by this Ascend baseline. Unsupported arguments are rejected. GPU
implementations and their existing dispatch remain available through the
runtime compatibility entry points. Batch-dependent counts and strides are
runtime kernel parameters, with compilation guards covering varying batches.

```bash
python -m pytest -q tokenspeed-kernel-npu/test/test_causal_conv1d.py
```
