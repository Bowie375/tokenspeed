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

"""Ascend GDN gate preparation and projection layout helpers."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _validate_device(reference, *tensors):
    if reference.device.type != "npu" or any(
        t.device != reference.device for t in tensors
    ):
        raise ValueError("GDN preparation tensors must be on the same Ascend device")
    if any(t.dtype not in _DTYPES for t in (reference, *tensors)):
        raise ValueError("GDN preparation supports FP16, BF16 and FP32 tensors")


def ascend_gdn_decay_gate(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    beta: float,
    threshold: float,
) -> torch.Tensor:
    """Return FP32 log decay, using stable native Ascend softplus.

    Args:
        A_log: Floating-point [heads] logarithmic decay rates.
        a: [tokens,heads] decay logits; row/column strides are honored.
        dt_bias: Floating-point [heads] bias.
        beta: Positive softplus sharpness, passed unchanged to softplus.
        threshold: Finite softplus linearization threshold.

    Returns:
        Contiguous FP32 [tokens,heads] -exp(A_log)*softplus(a+dt_bias).
        Inputs are unchanged; no host read of tensor values is performed.
    """
    _validate_device(a, A_log, dt_bias)
    if a.ndim != 2 or A_log.shape != (a.shape[1],) or dt_bias.shape != A_log.shape:
        raise ValueError(
            "GDN decay requires [tokens,heads] logits and [heads] parameters"
        )
    if not math.isfinite(beta) or beta <= 0 or not math.isfinite(threshold):
        raise ValueError(
            "softplus beta must be finite and positive; threshold must be finite"
        )
    return (
        -A_log.float().exp()
        * F.softplus(a.float() + dt_bias.float(), beta=beta, threshold=threshold)
    ).contiguous()


def ascend_pack_gdn_projection(
    mixed_qkv: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Copy projection views into independent dense QKV/a/b buffers.

    Args:
        mixed_qkv: [tokens,qkv_width] projection view.
        a: [tokens,value_heads] decay projection view.
        b: Same-shaped beta projection view.

    Returns:
        Contiguous, nonaliasing copies of the three inputs with their original
        dtypes. Empty token dimensions and noncontiguous input views are valid.
    """
    _validate_device(mixed_qkv, a, b)
    if (
        mixed_qkv.ndim != 2
        or a.ndim != 2
        or a.shape != b.shape
        or a.shape[0] != mixed_qkv.shape[0]
    ):
        raise ValueError(
            "GDN packing requires [tokens,qkv_width] and matching [tokens,heads] gates"
        )
    return tuple(
        t.clone(memory_format=torch.contiguous_format) for t in (mixed_qkv, a, b)
    )


def ascend_split_gdn_qkv(
    mixed_qkv: torch.Tensor,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_q: int,
    head_k: int,
    head_v: int,
    *,
    fuse_l2norm: bool,
    replay,
    valid_tokens: torch.Tensor | None,
):
    """Split post-convolution QKV into dense [1,tokens,heads,width] tensors.

    Args:
        mixed_qkv: [tokens,Qwidth+Kwidth+Vwidth] packed post-convolution rows.
        num_q_heads: Query heads.
        num_k_heads: Key heads.
        num_v_heads: Value heads.
        head_q: Per-query-head width.
        head_k: Per-key-head width.
        head_v: Per-value-head width.
        fuse_l2norm: Normalize Q/K in FP32 with epsilon 1e-6, then cast back.
        replay: Unsupported speculative payload; must be None.
        valid_tokens: Optional device integer scalar with the live row count.
            Rows beyond it become zero, including NaN-filled padded rows.

    Returns:
        Three independent contiguous Q/K/V tensors in the input dtype. Input
        strides are honored and the input is not modified. No query scaling
        is applied; recurrence owns that operation.
    """
    if replay is not None:
        raise NotImplementedError("Ascend GDN split does not support replay payloads")
    _validate_device(mixed_qkv)
    geometry = (num_q_heads, num_k_heads, num_v_heads, head_q, head_k, head_v)
    if min(geometry) <= 0 or mixed_qkv.ndim != 2:
        raise ValueError(
            "GDN split requires positive head geometry and two-dimensional input"
        )
    widths = (num_q_heads * head_q, num_k_heads * head_k, num_v_heads * head_v)
    if mixed_qkv.shape[1] != sum(widths):
        raise ValueError("GDN split projection width does not match head geometry")
    rows = mixed_qkv.shape[0]
    source = mixed_qkv
    if valid_tokens is not None:
        if (
            valid_tokens.numel() != 1
            or valid_tokens.dtype not in (torch.int32, torch.int64)
            or valid_tokens.device != mixed_qkv.device
        ):
            raise ValueError("valid_tokens must be a device integer scalar")
        mask = torch.arange(rows, device=mixed_qkv.device) < valid_tokens.reshape(())
        source = torch.where(mask[:, None], mixed_qkv, 0)
    result = []
    for index, (part, heads, width) in enumerate(
        zip(source.split(widths, dim=-1), geometry[:3], geometry[3:])
    ):
        dense = part.clone(memory_format=torch.contiguous_format).view(
            1, rows, heads, width
        )
        if fuse_l2norm and index < 2:
            values = dense.float()
            dense = (
                values * torch.rsqrt(values.square().sum(-1, keepdim=True) + 1e-6)
            ).to(mixed_qkv.dtype)
        result.append(dense)
    return tuple(result)
