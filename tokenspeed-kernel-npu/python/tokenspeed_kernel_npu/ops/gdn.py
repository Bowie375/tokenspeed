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

"""Ascend FP32 gated delta recurrence over K-last recurrent state."""

from __future__ import annotations

import torch
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit(
    do_not_specialize=[
        "q_token",
        "q_head",
        "q_col",
        "k_token",
        "k_head",
        "k_col",
        "v_token",
        "v_head",
        "v_col",
        "g_token",
        "g_head",
        "b_token",
        "b_head",
        "s_row",
        "s_head",
        "s_value",
        "s_key",
        "d_row",
        "d_head",
        "d_value",
        "d_key",
        "read_stride",
        "write_stride",
        "bound_stride",
        "log_stride",
        "bias_stride",
    ]
)
def _gdn_recurrence_kernel(
    Q,
    K,
    V,
    Gate,
    Beta,
    Source,
    Destination,
    Out,
    Bounds,
    Reads,
    Writes,
    ALog,
    Bias,
    q_token,
    q_head,
    q_col,
    k_token,
    k_head,
    k_col,
    v_token,
    v_head,
    v_col,
    g_token,
    g_head,
    b_token,
    b_head,
    s_row,
    s_head,
    s_value,
    s_key,
    d_row,
    d_head,
    d_value,
    d_key,
    read_stride,
    write_stride,
    bound_stride,
    log_stride,
    bias_stride,
    scale,
    HEADS: tl.constexpr,
    VALUE_HEADS: tl.constexpr,
    KEY_WIDTH: tl.constexpr,
    VALUE_WIDTH: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    DECODE: tl.constexpr,
    NORMALIZE: tl.constexpr,
    WRITE_STATE: tl.constexpr,
):
    request = tl.program_id(0)
    head = tl.program_id(1)
    values = tl.program_id(2) * BLOCK_V + tl.arange(0, BLOCK_V)
    keys = tl.arange(0, BLOCK_K)
    key_head = head // (VALUE_HEADS // HEADS)
    if DECODE:
        start = request
        end = request + 1
        source = tl.load(Reads + request * read_stride).to(tl.int64)
        destination = tl.load(Writes + request * write_stride).to(tl.int64)
    else:
        start = tl.load(Bounds + request * bound_stride)
        end = tl.load(Bounds + (request + 1) * bound_stride)
        source = request.to(tl.int64)
        destination = source
    mask = (values[:, None] < VALUE_WIDTH) & (keys[None, :] < KEY_WIDTH)
    memory = tl.load(
        Source
        + source * s_row
        + head * s_head
        + values[:, None] * s_value
        + keys[None, :] * s_key,
        mask=mask & (source >= 0),
        other=0,
    ).to(tl.float32)
    if DECODE:
        log_decay = tl.load(ALog + head * log_stride).to(tl.float32)
        bias = tl.load(Bias + head * bias_stride).to(tl.float32)
    for token in range(start, end):
        position = token.to(tl.int64)
        query = tl.load(
            Q + position * q_token + key_head * q_head + keys * q_col,
            mask=keys < KEY_WIDTH,
            other=0,
        ).to(tl.float32)
        key = tl.load(
            K + position * k_token + key_head * k_head + keys * k_col,
            mask=keys < KEY_WIDTH,
            other=0,
        ).to(tl.float32)
        value = tl.load(
            V + position * v_token + head * v_head + values * v_col,
            mask=values < VALUE_WIDTH,
            other=0,
        ).to(tl.float32)
        gate = tl.load(Gate + position * g_token + head * g_head).to(tl.float32)
        beta = tl.load(Beta + position * b_token + head * b_head).to(tl.float32)
        if DECODE:
            x = gate + bias
            exponential = tl.exp(-tl.abs(x))
            # Ascend's libdevice.log1p is not implemented. Below 1/16, six
            # Taylor terms avoid 1+x cancellation with relative truncation
            # error below 9e-9; elsewhere log(1+x) is well conditioned.
            small_log = exponential * (
                1.0
                + exponential
                * (
                    -0.5
                    + exponential
                    * (
                        1.0 / 3.0
                        + exponential
                        * (-0.25 + exponential * (0.2 - exponential / 6.0))
                    )
                )
            )
            log_term = tl.where(
                exponential < 0.0625, small_log, tl.log(1.0 + exponential)
            )
            softplus = tl.maximum(x, 0.0) + log_term
            gate = -tl.exp(log_decay) * softplus
            beta = tl.sigmoid(beta)
        if NORMALIZE:
            query *= tl.rsqrt(tl.sum(query * query, 0) + 1e-6)
            key *= tl.rsqrt(tl.sum(key * key, 0) + 1e-6)
        query *= scale
        memory *= tl.exp(gate)
        prediction = tl.sum(memory * key[None, :], 1)
        delta = beta * (value - prediction)
        memory += delta[:, None] * key[None, :]
        result = tl.sum(memory * query[None, :], 1)
        tl.store(
            Out + (position * VALUE_HEADS + head) * VALUE_WIDTH + values,
            result,
            mask=values < VALUE_WIDTH,
        )
    if WRITE_STATE:
        tl.store(
            Destination
            + destination * d_row
            + head * d_head
            + values[:, None] * d_value
            + keys[None, :] * d_key,
            memory,
            mask=mask & (destination >= 0),
        )


_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _same_device(reference: torch.Tensor, *tensors: torch.Tensor) -> None:
    if reference.device.type != "npu" or any(
        t.device != reference.device for t in tensors
    ):
        raise ValueError("GDN tensors must reside on the same Ascend device")


def _validate_inputs(q, k, v, state) -> None:
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or v.shape[:2] != q.shape[:2]:
        raise ValueError("GDN requires Q/K [B,T,H,K] and V [B,T,Hv,V]")
    heads, key_width = q.shape[2:]
    value_heads, value_width = v.shape[2:]
    if min(heads, key_width, value_heads, value_width) <= 0 or value_heads % heads:
        raise ValueError("GDN value heads must be a positive multiple of query heads")
    if q.dtype not in _FLOAT_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("GDN Q/K/V must share FP16, BF16 or FP32 dtype")
    if state.ndim != 4 or state.shape[1:] != (value_heads, value_width, key_width):
        raise ValueError(
            "GDN state must be [rows, value_heads, value_width, key_width]"
        )
    if state.dtype != torch.float32:
        raise ValueError("Ascend GDN requires FP32 recurrent state")
    _same_device(q, k, v, state)


def _validate_gates(q, gate, beta, shape) -> None:
    if gate.shape != shape or beta.shape != shape:
        raise ValueError("GDN gates must match token and value-head dimensions")
    if gate.dtype not in _FLOAT_DTYPES or beta.dtype not in _FLOAT_DTYPES:
        raise ValueError("GDN gates require FP16, BF16 or FP32 dtype")
    _same_device(q, gate, beta)


def _validate_indices(q, indices, rows) -> None:
    if indices.shape != (rows,) or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("GDN indices must be one integer per request")
    _same_device(q, indices)


def _launch(
    q,
    k,
    v,
    gate,
    beta,
    source,
    destination,
    bounds,
    reads,
    writes,
    A_log,
    bias,
    *,
    scale: float | None,
    normalize: bool,
    decode: bool,
    write_state: bool,
):
    axis = 0 if decode else 1
    requests = q.shape[0] if decode else source.shape[0]
    out = torch.empty(v.shape, dtype=q.dtype, device=q.device)
    if requests:
        # Narrow value tiles keep the loop-carried FP32 state and compiler
        # temporaries inside the 910C vector-core unified buffer at K=128.
        _gdn_recurrence_kernel[(requests, v.shape[2], triton.cdiv(v.shape[3], 4))](
            q,
            k,
            v,
            gate,
            beta,
            source,
            destination,
            out,
            bounds,
            reads,
            writes,
            A_log,
            bias,
            q.stride(axis),
            q.stride(2),
            q.stride(3),
            k.stride(axis),
            k.stride(2),
            k.stride(3),
            v.stride(axis),
            v.stride(2),
            v.stride(3),
            gate.stride(axis),
            gate.stride(2),
            beta.stride(axis),
            beta.stride(2),
            *source.stride(),
            *destination.stride(),
            reads.stride(0),
            writes.stride(0),
            bounds.stride(0),
            A_log.stride(0),
            bias.stride(0),
            q.shape[3] ** -0.5 if scale is None else scale,
            HEADS=q.shape[2],
            VALUE_HEADS=v.shape[2],
            KEY_WIDTH=q.shape[3],
            VALUE_WIDTH=v.shape[3],
            BLOCK_K=triton.next_power_of_2(q.shape[3]),
            BLOCK_V=4,
            DECODE=decode,
            NORMALIZE=normalize,
            WRITE_STATE=write_state,
        )
    return out


def gdn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    qk_l2norm: bool,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Scan packed sequences on Ascend, returning output and optional final state.

    Args:
        q: [1, tokens, query_heads, key_width] floating-point queries.
        k: Keys with the same shape and dtype as q.
        v: [1, tokens, value_heads, value_width] values, same dtype as q.
        g: [1, tokens, value_heads] FP32 or activation-dtype log decays.
        beta: Same-shaped update gates, already sigmoid transformed.
        scale: Query multiplier; None means key_width**-0.5.
        initial_state: FP32 [requests, value_heads, value_width, key_width].
            Read-only; arbitrary non-overlapping strides are supported.
        cu_seqlens: Device int32/int64 [requests+1] packed token boundaries.
            Must start at zero, be nondecreasing, and end at tokens.
        qk_l2norm: Normalize Q/K in FP32 with sum-of-squares epsilon 1e-6.
        output_final_state: Return the final FP32 state when True.

    Returns:
        (output shaped like v in q's dtype, optional final state). Empty
        sequences retain their initial state. No host read of boundaries is
        performed. The scan is serial within each sequence/value tile.
    """
    _validate_inputs(q, k, v, initial_state)
    if q.shape[0] != 1:
        raise ValueError("GDN packed prefill requires a leading dimension of 1")
    _validate_gates(q, g, beta, v.shape[:3])
    _validate_indices(q, cu_seqlens, initial_state.shape[0] + 1)
    if initial_state.shape[0] == 0 and q.shape[1] != 0:
        raise ValueError("nonempty prefill requires at least one request")
    final = (
        torch.empty(initial_state.shape, device=q.device, dtype=torch.float32)
        if output_final_state
        else None
    )
    out = _launch(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        initial_state if final is None else final,
        cu_seqlens,
        cu_seqlens,
        cu_seqlens,
        g,
        g,
        scale=scale,
        normalize=qk_l2norm,
        decode=False,
        write_state=output_final_state,
    )
    return out, final


def gdn_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float | None,
    output_state_indices: torch.Tensor | None,
    use_qk_l2norm: bool,
) -> torch.Tensor:
    """Run one indexed GDN decode step and update the FP32 state pool in place.

    Args:
        q: [requests, 1, query_heads, key_width] floating-point queries.
        k: Keys with q's shape/dtype.
        v: [requests, 1, value_heads, value_width], same dtype as q.
        A_log: [value_heads] logarithm of the positive decay rate.
        a: [requests, 1, value_heads] input-dependent decay logits.
        dt_bias: [value_heads] bias added to a before softplus.
        b: Same-shaped beta logits; sigmoid is evaluated inside the kernel.
        initial_state: FP32 [pool_rows, value_heads, value_width, key_width].
        initial_state_indices: Device integer [requests] read indices. Negative
            indices read a logical zero without accessing the pool.
        scale: Query multiplier; None means key_width**-0.5.
        output_state_indices: Device integer [requests] write indices, or None
            to use read indices. Negative destinations suppress writes.
        use_qk_l2norm: Normalize Q/K in FP32 with epsilon 1e-6.

    Returns:
        Output shaped like v in q's dtype. Nonnegative indices must be in
        bounds. Destinations must be unique and must not overwrite another
        request's input row; this is the runtime cache ownership contract.
        A request may update its own input row in place. Strides are honored.
    """
    _validate_inputs(q, k, v, initial_state)
    if q.shape[1] != 1:
        raise ValueError("GDN single-token decode requires T=1")
    _validate_gates(q, a, b, v.shape[:3])
    for parameter in (A_log, dt_bias):
        if parameter.shape != (v.shape[2],) or parameter.dtype not in _FLOAT_DTYPES:
            raise ValueError(
                "GDN decay parameters must be floating-point [value_heads]"
            )
    _same_device(q, A_log, dt_bias)
    _validate_indices(q, initial_state_indices, q.shape[0])
    writes = (
        initial_state_indices if output_state_indices is None else output_state_indices
    )
    _validate_indices(q, writes, q.shape[0])
    return _launch(
        q,
        k,
        v,
        a,
        b,
        initial_state,
        initial_state,
        initial_state_indices,
        initial_state_indices,
        writes,
        A_log,
        dt_bias,
        scale=scale,
        normalize=use_qk_l2norm,
        decode=True,
        write_state=True,
    )
