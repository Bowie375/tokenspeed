# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Ascend fused activation operators."""

from functools import lru_cache

import torch
import torch_npu
from tokenspeed_kernel_npu._triton import tl, triton


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Apply SwiGLU to last-dimension [gate, up], optionally writing ``out``."""
    if x.numel() == 0:
        if out is not None:
            return out
        return x.new_empty(x.shape[:-1] + (x.shape[-1] // 2,))
    result = torch_npu.npu_swiglu(x, dim=-1)
    if out is not None:
        out.copy_(result)
        return out
    return result


def dequant_silu_and_mul_quant(x, weight_scale, activation_scale):
    """Dequantize INT32 [gate,up], apply SwiGLU and dynamically quantize rows."""
    if x.numel() == 0:
        return (
            x.new_empty(x.shape[:-1] + (x.shape[-1] // 2,), dtype=torch.int8),
            x.new_empty(x.shape[:-1], dtype=torch.float32),
        )
    values, scales = torch_npu.npu_dequant_swiglu_quant(
        x.reshape(-1, x.shape[-1]),
        weight_scale=weight_scale,
        activation_scale=activation_scale.reshape(-1),
        activate_left=True,
        quant_mode=1,
    )
    return (
        values.reshape(x.shape[:-1] + (x.shape[-1] // 2,)),
        scales.reshape(x.shape[:-1]),
    )


@lru_cache(maxsize=None)
def _unit_smooth_scale(device: torch.device, width: int) -> torch.Tensor:
    # CANN 9.0's SwiGluQuant reads this optional input unconditionally.
    # Keep the identity tensor alive across graph capture/replay; do not create
    # a Fill kernel on every activation call.
    return torch.ones((1, width), device=device, dtype=torch.float32)


def silu_and_mul_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused floating SwiGLU + INT8 quantization with dequantization scales."""
    if x.numel() == 0:
        return (
            x.new_empty(x.shape[:-1] + (x.shape[-1] // 2,), dtype=torch.int8),
            x.new_empty(x.shape[:-1], dtype=torch.float32),
        )
    values, inverse_scale = torch_npu.npu_swiglu_quant(
        x,
        smooth_scales=_unit_smooth_scale(x.device, x.shape[-1] // 2),
        activate_left=True,
        quant_mode=1,
    )
    # Unlike DynamicQuant/DequantSwigluQuant this op returns 127/amax.
    return values, inverse_scale.reciprocal()


@triton.jit(do_not_specialize=["n", "gate_row_stride", "gate_head_stride"])
def _sigmoid_mul_kernel(
    X,
    Gate,
    Bias,
    n,
    gate_row_stride,
    gate_head_stride,
    WIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < n
    row = offset // WIDTH
    column = offset % WIDTH
    head = column // HEAD_DIM
    within_head = column % HEAD_DIM
    values = tl.load(X + offset, mask=mask, other=0).to(tl.float32)
    gate = tl.load(
        Gate + row * gate_row_stride + head * gate_head_stride + within_head,
        mask=mask,
        other=0,
    ).to(tl.float32)
    if HAS_BIAS:
        gate += tl.load(Bias + head, mask=mask, other=0).to(tl.float32)
    tl.store(X + offset, values * tl.sigmoid(gate), mask=mask)


def sigmoid_mul(
    x: torch.Tensor,
    gate: torch.Tensor,
    head_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the full-attention output sigmoid gate in place on Ascend.

    Args:
        x: Contiguous FP16/BF16/FP32 [tokens,width] attention output to overwrite.
        gate: Same-dtype/device [tokens,width] or [tokens,heads,head_dim] logits.
            The last dimension is dense; rows and heads may be strided views.
        head_bias: Optional contiguous floating [heads] bias on the same device.
            For a flat gate, its length partitions width into equal heads.

    Returns:
        x itself, overwritten with FP32 x*sigmoid(gate+optional head bias),
        rounded once to x's dtype. Gate and bias storage are read-only. Empty
        tensors return without launching. Counts and gate strides are runtime
        parameters; model width and head dimension specialize the kernel.
    """
    dtypes = (torch.bfloat16, torch.float16, torch.float32)
    if x.device.type != "npu" or x.dtype not in dtypes:
        raise ValueError("sigmoid_mul requires FP16, BF16 or FP32 Ascend input")
    if x.ndim != 2 or not x.is_contiguous():
        raise ValueError("sigmoid_mul x must be contiguous 2D [tokens,width]")
    if gate.device != x.device or gate.dtype != x.dtype:
        raise ValueError("sigmoid_mul gate must match the input device and dtype")
    tokens, width = x.shape
    if gate.ndim == 2:
        if gate.shape != x.shape:
            raise ValueError("sigmoid_mul flat gate shape must match x")
        head_dim, head_stride = width, width
    elif gate.ndim == 3:
        if gate.shape[0] != tokens or gate.shape[1] * gate.shape[2] != width:
            raise ValueError("sigmoid_mul headed gate shape must match x")
        head_dim, head_stride = gate.shape[2], gate.stride(1)
    else:
        raise ValueError("sigmoid_mul gate must be 2D or 3D")
    if gate.stride(-1) != 1:
        raise ValueError("sigmoid_mul gate must have dense columns")
    if head_bias is not None:
        if (
            head_bias.ndim != 1
            or not head_bias.is_contiguous()
            or head_bias.device != x.device
            or head_bias.dtype not in dtypes
        ):
            raise ValueError(
                "sigmoid_mul head_bias must be a contiguous floating vector on the input device"
            )
        heads = head_bias.numel()
        if gate.ndim == 3:
            if heads != gate.shape[1]:
                raise ValueError(
                    "sigmoid_mul head_bias must have one value per gate head"
                )
        elif heads == 0 or width % heads:
            raise ValueError("sigmoid_mul head_bias head count must divide width")
        else:
            head_dim = head_stride = width // heads
    if x.numel():
        _sigmoid_mul_kernel[(triton.cdiv(x.numel(), 1024),)](
            x,
            gate,
            gate if head_bias is None else head_bias,
            x.numel(),
            gate.stride(0),
            head_stride,
            WIDTH=width,
            HEAD_DIM=head_dim,
            HAS_BIAS=head_bias is not None,
            BLOCK=1024,
        )
    return x
