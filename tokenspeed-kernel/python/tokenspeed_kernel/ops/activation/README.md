# Activation kernels

## Full-attention output sigmoid gate

The shared `sigmoid_mul(x, gate, head_bias=None)` API dispatches to the fused
Ascend implementation in `tokenspeed_kernel_npu.ops.activation` on NPU and
the existing Triton implementation on GPU. Qwen3.5 imports this shared API.
The operation overwrites contiguous `[tokens,width]` attention output and
returns the same tensor. The caller may ignore the return value.

Gate logits are either `[tokens,width]` or `[tokens,heads,head_dim]` with dense
last-dimension columns. Row and head strides are honored directly, including
the gate view produced by splitting a packed Q/gate/K/V projection. Gate and
bias storage remain read-only. FP16, BF16 and FP32 activations are supported;
gate dtype and device must match the output. An optional contiguous floating
bias supplies one value per head. With a flat gate, its length divides width
into equally sized heads. Supplied biases are applied before sigmoid.

Ascend loads values, logits and bias in FP32, computes
`x * sigmoid(gate + bias)` and rounds once when storing to x. Empty tensors
launch no kernel. Token counts and gate strides are runtime parameters;
width, head dimension and bias presence are fixed model properties. GPU
PDL behavior and its existing bias arithmetic remain unchanged. This is an
inference operation with no backward implementation.

```bash
python -m pytest -q tokenspeed-kernel-npu/test/test_attention_output_gate.py
```

Tests compare against an independent CPU FP64 reference, including Qwen3.5's
`[tokens,2048]` output and strided `[tokens,8,256]` gate. They check in-place
mutation, unchanged projection storage, optional biases, extreme logits,
empty batches, flat gate rows and invalid arguments. Compilation guards sweep
token counts and projection strides after warmup.
