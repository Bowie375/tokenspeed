# GDN kernels

## Ascend recurrence

Ascend registers `ascend_gdn_chunk_prefill` and `ascend_gdn_decode_step`
under the shared `gdn_chunk_prefill` / `gdn_decode_step` APIs with solution
`triton`. The implementation lives in `tokenspeed_kernel_npu.ops.gdn`.
Both operations use one recurrence, with serial token traversal per sequence
and parallel work across requests, heads and narrow value tiles. This is a
correctness baseline; it does not use the GPU chunked scan algorithm.

Q/K/V may be BF16, FP16 or FP32; recurrent state must be FP32 in K-last
`[rows, value_heads, value_width, key_width]` layout. Tensor strides are
honored. Q/K normalization, gates and state updates compute in FP32, and
outputs round to the Q/K/V dtype. Optional Q/K normalization divides by
`sqrt(sum(x*x) + 1e-6)`. The query scale is applied before the state/output
contraction. Prefill consumes log decays and already transformed beta gates;
decode computes `-exp(A_log) * softplus(a + dt_bias)` and `sigmoid(b)`.

Prefill reads packed device sequence boundaries without a host copy. Valid
boundaries start at zero, are nondecreasing and end at the packed token count.
Empty sequences return their initial state. The initial state is never
modified; final state can be omitted. Intermediate checkpoints (`output_h`)
and speculative verification are unsupported, and this implementation does
not advertise prefill graph capture.

Decode reads through `initial_state_indices` and writes through
`output_state_indices`, or the input indices when no output indices are
provided. Negative read indices supply logical zero state, and negative
destinations skip writes. Nonnegative indices must be in bounds. Writable
state rows must be nonoverlapping and destinations unique; a destination
cannot overwrite another request's input checkpoint. In-place evolution of
a request's own row is supported. These are the existing cache ownership
requirements, not a second state lifecycle. Index values and packed boundary
contents are caller-owned metadata contracts; the wrapper validates their
shapes, dtypes and devices without adding serving-time host synchronization.

Sequence lengths, request counts and state/tensor strides are runtime values,
not compile-time specialization keys. NPU tests compare to an independent
CPU FP64 recurrence, including unequal key/value dimensions, strided tensors,
ragged/empty sequences, multi-step decode, state destination isolation and
the actual Qwen3.5-0.8B head geometry. Compilation guards sweep token and
request counts after warmup.

```bash
python -m pytest -q tokenspeed-kernel-npu/test/test_gdn.py
```

## FlashInfer PDL adapters

The FlashInfer adapter retains the upstream launch geometry and runtime ABI,
while wrapping device bodies with PDL synchronization. Adapted functions and
in-memory compilation caches live in private namespaces.

Decode and prefill explicitly select FlashInfer's CuTe backend. FlashInfer
0.7.0's automatic backend selection can use Cake GDN for supported shapes,
which bypasses the PDL-wrapped device bodies. Regression tests cover both
power-of-two and other head groupings and reject calls into that alternate path.

FlashInfer 0.7.0 also persists CuTe-DSL kernels to disk. PDL artifacts use a
separate `tokenspeed_pdl_` module namespace so an ordinary kernel cannot satisfy
a PDL cache lookup, or vice versa. Cache invalidation includes all upstream
source files plus the local PDL and GDN adapter sources. Installed FlashInfer
modules and the process-wide cache configuration remain untouched.

GPU regression tests cover persistent cache hits after clearing the in-memory
cache, invalidation after source changes, and both launch orders. Runtime GDN
tests additionally check CUDA Graph dependency edges and exact replay results
for decode, MTP, BF16 state, and prefill.

## Chain verification

ReplaySSM chains use chunked verification on Blackwell for 13–16 tokens and
at least 24 (request, value-head) states. Other shapes and platforms retain
step-by-step verification. `test/runtime/test_gdn_verify_policy.py` checks this
selection and its boundaries without launching device kernels.

## Variable prefill lengths

The Triton L2-normalization fallback uses a runtime token count. The launch
grid and masked block pointers cover the exact rows, while head width and tile
size stay compile-time parameters. Consecutive agentic turns therefore reuse
one compiled kernel across different token counts instead of compiling a new
variant for each length. Tests compare irregular lengths around the usual
prefill graph buckets against an independent PyTorch result.

Compilation guards warm these kernels, then sweep token counts (and PLE
request counts) without allowing new Triton specializations. PLE covers
transitions between single-request, uniform multi-request and ragged indexing
within one compiled kernel. Those selections use runtime values so startup
warmup also covers batch layouts first seen during serving.
