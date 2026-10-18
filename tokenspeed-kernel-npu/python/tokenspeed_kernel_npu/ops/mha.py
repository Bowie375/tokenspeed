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

"""Ascend multi-head attention kernels."""

from __future__ import annotations

import math

import torch
import torch_npu
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit(do_not_specialize=["st", "sh", "sd", "capacity"])
def _pack_bsh_kernel(
    Source,
    Bounds,
    Out,
    st,
    sh,
    sd,
    capacity,
    WIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0).to(tl.int64)
    offset = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    token, feature = offset // WIDTH, offset % WIDTH
    start = tl.load(Bounds + request).to(tl.int64)
    end = tl.load(Bounds + request + 1).to(tl.int64)
    mask = token < capacity
    value = tl.load(
        Source
        + (start + token) * st
        + feature // HEAD_DIM * sh
        + feature % HEAD_DIM * sd,
        mask=mask & (start + token < end),
        other=0,
    )
    tl.store(Out + (request * capacity + token) * WIDTH + feature, value, mask=mask)


@triton.jit(do_not_specialize=["sb", "st", "sf", "capacity"])
def _unpack_bsh_kernel(
    Source, Bounds, Out, sb, st, sf, capacity, WIDTH: tl.constexpr, BLOCK: tl.constexpr
):
    request = tl.program_id(0).to(tl.int64)
    offset = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    token, feature = offset // WIDTH, offset % WIDTH
    start = tl.load(Bounds + request).to(tl.int64)
    end = tl.load(Bounds + request + 1).to(tl.int64)
    live = (token < capacity) & (start + token < end)
    value = tl.load(
        Source + request * sb + token * st + feature * sf, mask=live, other=0
    )
    tl.store(Out + (start + token) * WIDTH + feature, value, mask=live)


def _pack_bsh(
    source: torch.Tensor, boundaries: torch.Tensor, capacity: int
) -> torch.Tensor:
    """Pack live token rows and zero ragged padding without reading lengths on host."""
    if (
        capacity <= 0
        or source.ndim != 3
        or boundaries.ndim != 1
        or not boundaries.is_contiguous()
    ):
        raise ValueError(
            "BSH packing requires 3D rows, contiguous boundaries and positive capacity"
        )
    if (
        boundaries.device != source.device
        or boundaries.dtype not in (torch.int32, torch.int64)
        or boundaries.numel() < 2
        or source.shape[1] <= 0
        or source.shape[2] <= 0
    ):
        raise ValueError(
            "BSH packing requires positive head geometry and integer boundaries on the input device"
        )
    rows = boundaries.numel() - 1
    width = source.shape[1] * source.shape[2]
    result = torch.empty(
        (rows, capacity, width), device=source.device, dtype=source.dtype
    )
    if rows:
        _pack_bsh_kernel[(rows, triton.cdiv(capacity * width, 1024))](
            source,
            boundaries,
            result,
            *source.stride(),
            capacity,
            WIDTH=width,
            HEAD_DIM=source.shape[2],
            BLOCK=1024,
        )
    return result


def _unpack_bsh(
    source: torch.Tensor, boundaries: torch.Tensor, query: torch.Tensor
) -> torch.Tensor:
    """Return contiguous token/head rows, preserving the public packed ABI."""
    output = torch.zeros(query.shape, device=query.device, dtype=query.dtype)
    width = query.shape[1] * query.shape[2]
    rows, capacity = source.shape[:2]
    if rows:
        _unpack_bsh_kernel[(rows, triton.cdiv(capacity * width, 1024))](
            source,
            boundaries,
            output,
            *source.stride(),
            capacity,
            WIDTH=width,
            BLOCK=1024,
        )
    return output


_CAUSAL_MASKS: dict[torch.device, torch.Tensor] = {}


def _causal_mask(device: torch.device) -> torch.Tensor:
    mask = _CAUSAL_MASKS.get(device)
    if mask is None:
        mask = torch.triu(
            torch.ones((2048, 2048), dtype=torch.bool, device=device), diagonal=1
        )
        _CAUSAL_MASKS[device] = mask
    return mask


def _scale(q: torch.Tensor, softmax_scale: float | None) -> float:
    return softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])


def _check_options(
    *,
    window_left: int,
    logit_cap: float,
    sinks: torch.Tensor | None,
    return_lse: bool,
) -> None:
    if window_left >= 0:
        raise NotImplementedError("Ascend MHA does not support sliding windows")
    if logit_cap:
        raise NotImplementedError("Ascend MHA does not support logit caps")
    if sinks is not None:
        raise NotImplementedError("Ascend MHA does not support attention sinks")
    if return_lse:
        raise NotImplementedError("Ascend MHA does not return LSE")


def mha_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: list[int],
    max_seqlen: int,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run packed causal MHA through CANN's BSH layout, including 256-wide heads.

    Ragged rows are packed/unpacked on device. The CPU boundary mirror provides
    per-request lengths to CANN; original inputs and the packed output ABI remain
    unchanged. No device metadata is read back to the host.
    """
    _check_options(
        window_left=window_left,
        logit_cap=logit_cap,
        sinks=sinks,
        return_lse=return_lse,
    )
    if q.shape[0] == 0:
        return torch.empty_like(q)
    lengths = [end - start for start, end in zip(cu_seqlens_cpu, cu_seqlens_cpu[1:])]
    if (
        len(cu_seqlens_cpu) != cu_seqlens.numel()
        or not cu_seqlens_cpu
        or cu_seqlens_cpu[0] != 0
        or cu_seqlens_cpu[-1] != q.shape[0]
        or any(length < 0 or length > max_seqlen for length in lengths)
    ):
        raise ValueError(
            "prefill host boundaries must cover the query and fit max_seqlen"
        )
    output, _ = torch_npu.npu_fused_infer_attention_score(
        _pack_bsh(q, cu_seqlens, max_seqlen),
        _pack_bsh(k, cu_seqlens, max_seqlen),
        _pack_bsh(v, cu_seqlens, max_seqlen),
        atten_mask=_causal_mask(q.device),
        actual_seq_lengths=lengths,
        actual_seq_lengths_kv=lengths,
        num_heads=q.shape[1],
        num_key_value_heads=k.shape[1],
        scale=_scale(q, softmax_scale),
        input_layout="BSH",
        sparse_mode=2,
    )
    return _unpack_bsh(output, cu_seqlens, q)


def mha_extend_with_kvcache(
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Run variable-length MHA over a paged K/V cache."""
    del cu_seqlens_kv, max_seqlen_k, enable_pdl
    _check_options(
        window_left=window_left,
        logit_cap=logit_cap,
        sinks=sinks,
        return_lse=return_lse,
    )
    if q_scale is not None or k_scale is not None or v_scale is not None:
        raise NotImplementedError("Ascend MHA does not support scaled FP8 cache")

    if (
        q.shape[1] != k_cache.shape[2]
        and max_seqlen_q > 1
        and (k_cache.shape[1] % 128 or not 128 <= k_cache.shape[1] <= 1024)
    ):
        raise NotImplementedError(
            "Ascend paged GQA extension with multiple queries requires a page size divisible by 128 in [128,1024]"
        )
    if q.shape[0] == 0:
        return torch.empty_like(q)
    output, _ = torch_npu.npu_fused_infer_attention_score(
        _pack_bsh(q, cu_seqlens_q, max_seqlen_q),
        k_cache.flatten(2),
        v_cache.flatten(2),
        atten_mask=_causal_mask(q.device) if is_causal else None,
        actual_seq_lengths=cu_seqlens_q[1:] - cu_seqlens_q[:-1],
        actual_seq_lengths_kv=cache_seqlens,
        block_table=page_table,
        num_heads=q.shape[1],
        num_key_value_heads=k_cache.shape[2],
        scale=_scale(q, softmax_scale),
        input_layout="BSH",
        sparse_mode=3 if is_causal else 0,
        block_size=k_cache.shape[1],
    )
    return _unpack_bsh(output, cu_seqlens_q, q)


def mha_decode_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    max_seqlen_q: int = 1,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Run fixed-shape, graph-capturable decode over a paged K/V cache."""
    _check_options(
        window_left=window_left,
        logit_cap=logit_cap,
        sinks=sinks,
        return_lse=return_lse,
    )
    if max_seqlen_q != 1:
        raise NotImplementedError("Ascend MHA decode supports one query per request")
    if q_scale is not None or k_scale is not None or v_scale is not None:
        raise NotImplementedError("Ascend MHA does not support scaled FP8 cache")

    del max_seqlen_k, enable_pdl
    batch_size = cache_seqlens.shape[0]
    actual_seq_lengths_kv = (
        [1] * batch_size if torch.npu.is_current_stream_capturing() else cache_seqlens
    )
    output, _ = torch_npu.npu_fused_infer_attention_score(
        q.reshape(batch_size, 1, -1),
        k_cache.flatten(2),
        v_cache.flatten(2),
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        block_table=page_table,
        num_heads=q.shape[1],
        num_key_value_heads=k_cache.shape[2],
        scale=_scale(q, softmax_scale),
        input_layout="BSH",
        block_size=k_cache.shape[1],
    )
    return output.reshape_as(q)


__all__ = [
    "mha_decode_with_kvcache",
    "mha_extend_with_kvcache",
    "mha_prefill",
]
