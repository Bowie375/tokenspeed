# MHA kernels

## Ascend layout contract

CANN 9.0 rejects unquantized `TND` attention with head dimension 256. The
Ascend adapter therefore uses `BSH` for both uncached prefill and paged
extension. This covers Qwen3.5's eight query heads, two KV heads and 256-wide
heads while retaining the public packed `[tokens,heads,head_dim]` ABI.

Portable Triton kernels pack ragged sequences into padded BSH buffers and
unpack live output rows. Packing honors input strides, zeroes padding and
reads device sequence boundaries. It introduces temporary buffers sized by
batch count and maximum query length. Batch counts, maximum lengths and
strides are runtime quantities; model head geometry alone specializes the
layout kernels. There is no device-to-host length read or per-request upload.

BSH consumes per-request lengths, rather than the cumulative endpoints used
by TND. Uncached prefill derives those lengths from the existing host mirror;
extension uses device boundary differences. Host prefill boundaries must
cover all query rows and fit the supplied maximum length. Empty queries
return without calling CANN. Inputs and paged caches remain unchanged.

Single-token query capacity selects CANN's decode-like BSH tiler, which rejects
the compressed causal mask. Prefill and causal extension omit that mask and
use dense visibility for this case: an uncached one-token sequence has one
key, and a sole cached query occupies the rightmost live position. Empty
requests contribute no packed output. Multi-token causality is unchanged.

Causal uncached prefill uses the upper-triangular mask. Paged causal extension
uses CANN's right-aligned causal mask so each new query attends its cached
prefix and preceding new tokens. GQA extension with multiple query tokens
requires page sizes divisible by 128 in [128,1024]; unsupported sizes raise
before the native call. Qwen3.5's default page geometry is 128. Single-token
paged decode retains its existing BSH implementation and ACL graph metadata
update contract.

```bash
python -m pytest -q tokenspeed-kernel-npu/test/test_mha.py
```

Tests cover 64- and 256-wide heads, GQA, single-token/ragged/empty sequences, strided
queries, custom softmax scaling, multi-page cached prefixes, decode and graph
replay with changing cache lengths. CPU FP64 comparisons use dtype-specific
absolute/relative tolerances and relative L2 bounds for native mixed-precision
attention. Layout conversion itself is bit-exact and has compilation guards
across request counts, sequence capacities and input strides.

## Ascend encoder attention

`mha_encoder_attention` performs bidirectional self-attention without a KV cache.
The Ascend solution uses native `npu_fusion_attention` in TND layout with dropout
disabled and unrestricted left/right visibility. Inputs are FP16/BF16
`[tokens, heads, width]`; strided views are accepted without modifying storage.
Cumulative token boundaries isolate images, including zero-length segments.
Empty input returns an empty output without a native launch.

The runtime backend is `--mm-attention-backend ascend`, also the Ascend default.
Qwen vision metadata supplies a host boundary mirror once per encode, avoiding
per-block device-to-host reads. Callers without that mirror use an eager host
read. Scale and maximum length are preserved; cuDNN workspace/length metadata
is rejected. Encoder graph capture, video and encoder parallelism are outside
this initial validation scope; set `TOKENSPEED_MM_ENABLE_ENCODER_CUDA_GRAPH=0`.

Run `python -m pytest -q tokenspeed-kernel-npu/test/test_vision_attention.py` on
Ascend. Coverage includes noncausality, image isolation, ragged/empty segments,
strides, explicit scale through the full runtime module, rank restoration and
both supported dtypes against
an independent CPU FP64 reference.
