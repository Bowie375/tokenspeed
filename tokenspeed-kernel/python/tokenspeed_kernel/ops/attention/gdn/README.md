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

## Ascend preparation

The shared APIs `pack_gdn_projection`, `gdn_decay_gate` and
`fused_qkv_split_gdn_prefill` register native Ascend PyTorch implementations in
`tokenspeed_kernel_npu.ops.gdn_helpers`. They are a correctness baseline;
packing and splitting allocate independent dense buffers and are not fused
performance kernels. Runtime callers use these shared APIs rather than GPU
implementation modules.

Packing copies the possibly strided QKV, decay-logit `a` and update-logit `b`
views into contiguous tensors, preserving each dtype. Gate preparation returns
FP32 `-exp(A_log) * softplus(a + dt_bias)` with explicitly supplied softplus
beta and threshold; stable softplus preserves small decays for negative logits.
The existing recurrence owns sigmoid transformation of decode update gates;
prefill receives already transformed gates from its caller.

Post-convolution split returns independent `[1,tokens,heads,width]` Q/K/V
buffers. Optional Q/K L2 normalization computes in FP32 with epsilon `1e-6`
before rounding to the input dtype. Callers select normalization explicitly
and must avoid normalizing twice. An optional device `valid_tokens` scalar
zeros padded rows, including NaN padding, without a host read. Replay payloads
are explicitly unsupported on Ascend. BF16, FP16 and FP32 inputs are supported.

```bash
python -m pytest -q tokenspeed-kernel-npu/test/test_gdn_preparation.py
```

Tests compare preparation against CPU references and connect packing,
convolution, splitting and recurrence, checking prefill against repeated
decode with matching convolution and recurrent state.

## Ascend state preparation and metadata

`prepare_prefill_state_inputs` stages convolution history and gathers recurrent
inputs through the shared GDN API. Its Ascend implementation is fused in
`tokenspeed_kernel_npu.ops.state`. Cache tensors can have arbitrary strides.
Positive input ids resume history; nonpositive inputs materialize zero
recurrent state without reading the null block. Fresh convolution windows
remain unchanged and are interpreted as logical zeros by the convolution's
history flags. Nonpositive destinations are padding: they access no cache,
return zero recurrent state and have a false history flag. This avoids Python
negative indexing into the last cache block.

Resumed convolution history is copied only when input and destination differ.
Shared checkpoints are read-only; writable destinations must be unique and
cannot overlap another request's input. The existing two-slot resolver keeps
fresh reads at id 0, chooses a positive checkpoint before a live block, and
always writes the live slot. Padding remains visible in the output ids as -1.
Its optional host-synchronizing debug validation checks cross-request input
aliasing as well as missing and duplicate live blocks. Ordinary execution
validates tensor shapes, devices and integer types without reading values.

`tokenspeed_kernel.ops.kvcache.store_state_rows` dispatches recurrent writeback
to a fused masked Ascend scatter. It preserves strides, casts to the cache
dtype and skips negative destinations, including NaN dummy scan results.
The generic scatter permits destination 0; the state-table contract prevents
live requests from writing the null block. Index bounds and destination
ownership remain caller contracts.

`build_causal_conv1d_prefill_metadata` is exposed through the shared GDN API.
The host length mirror sizes exact maps and retains host int64 boundaries;
a portable Triton kernel builds the Ascend request/chunk arrays from device
boundaries in one launch, with no per-request host assignments. CPU and GPU
entry points retain their previous implementations. Empty sequences emit no
programs. The maps belong to one forward and are reused across layers.

The existing runtime `scrub_padding_tail` keeps live token rows and overwrites
padded rows with zero, including NaNs and strided views. An unspecified live
extent is a no-op; a negative extent is rejected. It uses the host metadata
mirror and adds no device-to-host synchronization.

```bash
python -m pytest -q tokenspeed-kernel-npu/test/test_gdn_state_helpers.py
```

Tests cover fresh/resumed/padded gathers, shared checkpoint readers, strided
cache views, masked writeback, casts, exact ragged maps and compilation reuse.
A chunked-prefill integration test resolves blocks, gathers state, constructs
convolution metadata, runs convolution and recurrence, then writes back to a
new live block while preserving the old checkpoint. This uses the existing
cache lifecycle and introduces no backend-private request state.

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
