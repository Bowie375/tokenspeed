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

"""Ascend normalization operators."""

from __future__ import annotations

import torch
import torch_npu
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit
def _gemma_rmsnorm_kernel(
    X,
    Residual,
    X2,
    Weight,
    Out,
    stride_x,
    stride_residual,
    stride_x2,
    stride_out,
    eps,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_X2: tl.constexpr,
    GEMMA: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < WIDTH
    addend = tl.load(X + row * stride_x + cols, mask=mask, other=0)
    if HAS_X2:
        second = tl.load(X2 + row * stride_x2 + cols, mask=mask, other=0)
        # The optional addends are summed in the activation dtype first.
        addend = (addend.to(tl.float32) + second.to(tl.float32)).to(X.dtype.element_ty)
    total = addend.to(tl.float32)
    if HAS_RESIDUAL:
        total += tl.load(
            Residual + row * stride_residual + cols, mask=mask, other=0
        ).to(tl.float32)
        tl.store(Residual + row * stride_residual + cols, total, mask=mask)
    # Keep the unrounded sum for normalization, independently of the residual store.
    variance = tl.sum(total * total, axis=0) / WIDTH
    weight = tl.load(Weight + cols, mask=mask, other=0).to(tl.float32)
    if GEMMA:
        weight = 1.0 + weight
    result = total * tl.rsqrt(variance + eps) * weight
    tl.store(Out + row * stride_out + cols, result, mask=mask)


def _validate_norm_inputs(
    x: torch.Tensor, weight: torch.Tensor, *operands: torch.Tensor | None
) -> None:
    if x.ndim != 2 or x.shape[1] == 0 or x.stride(1) != 1:
        raise ValueError("RMSNorm input must be [rows, width] with dense columns")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("RMSNorm requires FP16, BF16 or FP32 activations")
    if weight.shape != (x.shape[1],) or not weight.is_contiguous():
        raise ValueError(
            "RMSNorm weight must be a contiguous vector matching the width"
        )
    if weight.device != x.device or weight.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("RMSNorm weight must be floating point on the input device")
    for operand in operands:
        if operand is not None and (
            operand.shape != x.shape
            or operand.dtype != x.dtype
            or operand.device != x.device
            or operand.stride(1) != 1
        ):
            raise ValueError(
                "RMSNorm operands must match the input shape, dtype and device "
                "with dense columns"
            )


def gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Normalize [rows, width] with FP32 ``1 + weight`` scaling.

    With a residual, update it in place and overwrite x with normalized rows,
    returning (x, residual). Statistics use the FP32 sum before its residual
    dtype cast. Without a residual, return a new tensor and leave x unchanged.
    """
    _validate_norm_inputs(x, weight, residual)
    if residual is not None:
        add_rmsnorm(
            x,
            residual,
            weight,
            eps,
            x2=None,
            out=x,
            out_fp8=None,
            fp8_scale=None,
            out_fp4=None,
            fp4_scale=None,
            gemma=True,
        )
        return x, residual
    out = torch.empty_like(x)
    if x.shape[0]:
        _gemma_rmsnorm_kernel[(x.shape[0],)](
            x,
            x,
            x,
            weight,
            out,
            x.stride(0),
            0,
            0,
            out.stride(0),
            eps,
            WIDTH=x.shape[1],
            BLOCK=triton.next_power_of_2(x.shape[1]),
            HAS_RESIDUAL=False,
            HAS_X2=False,
            GEMMA=True,
        )
    return out


def add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    x2: torch.Tensor | None,
    out: torch.Tensor,
    out_fp8: torch.Tensor | None,
    fp8_scale: torch.Tensor | None,
    out_fp4: tuple[torch.Tensor, torch.Tensor] | None,
    fp4_scale: torch.Tensor | None,
    gemma: bool,
) -> None:
    """Fuse residual addition and RMSNorm in one Ascend launch.

    Args:
        x: [rows, width] activation with dense columns.
        residual: Same-shaped residual, updated in place in its original dtype.
        weight: Contiguous [width] normalization weight.
        eps: RMS denominator epsilon.
        x2: Optional addend, summed with x in x's dtype before adding residual.
        out: Same-shaped output, which may alias x.
        out_fp8: Unsupported optional FP8 output; must be None.
        fp8_scale: Unsupported FP8 scale; must be None.
        out_fp4: Unsupported optional NVFP4 outputs; must be None.
        fp4_scale: Unsupported NVFP4 scale; must be None.
        gemma: Multiply by FP32 (1 + weight) instead of weight when True.

    Returns:
        None. The normalized output and updated residual are written in place.
        Normalization reads the unrounded FP32 residual sum.
    """
    if any(value is not None for value in (out_fp8, fp8_scale, out_fp4, fp4_scale)):
        raise NotImplementedError(
            "Ascend add_rmsnorm does not support FP8 or NVFP4 outputs"
        )
    _validate_norm_inputs(x, weight, residual, x2, out)
    if x.shape[0] == 0:
        return
    _gemma_rmsnorm_kernel[(x.shape[0],)](
        x,
        residual,
        x if x2 is None else x2,
        weight,
        out,
        x.stride(0),
        residual.stride(0),
        0 if x2 is None else x2.stride(0),
        out.stride(0),
        eps,
        WIDTH=x.shape[1],
        BLOCK=triton.next_power_of_2(x.shape[1]),
        HAS_RESIDUAL=True,
        HAS_X2=x2 is not None,
        GEMMA=gemma,
    )


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply RMSNorm, optionally adding and updating a residual tensor."""
    if x.shape[0] == 0:
        result = x if out is None else out
        return (result, residual) if residual is not None else result
    if residual is not None:
        result, _, residual_out = torch_npu.npu_add_rms_norm(
            x, residual, weight, epsilon=eps
        )
        residual.copy_(residual_out)
        return result, residual
    result, _ = torch_npu.npu_rms_norm(x, weight, epsilon=eps)
    if out is not None:
        out.copy_(result)
        result = out
    return result


def qk_rmsnorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply per-head RMSNorm to query and key tensors."""
    del enable_pdl
    head_dim = q_weight.numel()
    if k_weight.numel() != head_dim:
        raise ValueError("q and k weights must have the same head dimension")
    if q.shape[-1] % head_dim or k.shape[-1] % head_dim:
        raise ValueError("q and k widths must be divisible by the head dimension")

    q_shape, k_shape = q.shape, k.shape
    q_heads = q.reshape(-1, q.shape[-1] // head_dim, head_dim)
    k_heads = k.reshape(-1, k.shape[-1] // head_dim, head_dim)
    q_scale = torch.rsqrt(q_heads.float().square().mean(-1, keepdim=True) + eps)
    k_scale = torch.rsqrt(k_heads.float().square().mean(-1, keepdim=True) + eps)
    q_out = (q_heads * q_scale.to(q.dtype) * q_weight).reshape(q_shape)
    k_out = (k_heads * k_scale.to(k.dtype) * k_weight).reshape(k_shape)
    return q_out, k_out


__all__ = ["add_rmsnorm", "gemma_rmsnorm", "qk_rmsnorm", "rmsnorm"]
