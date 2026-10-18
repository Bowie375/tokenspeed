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

"""Ascend causal depthwise convolution with cache-managed history."""

from __future__ import annotations

import torch
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit(
    do_not_specialize=[
        "xc",
        "xt",
        "wc",
        "ww",
        "sr",
        "sc",
        "st",
        "oc",
        "ot",
        "bound_stride",
        "index_stride",
        "history_stride",
        "map_stride",
        "offset_stride",
        "bias_stride",
        "pad",
    ],
    do_not_specialize_on_alignment=["Maps", "Offsets"],
)
def _conv_prefill_kernel(
    X,
    Weight,
    Bias,
    State,
    Bounds,
    Indices,
    History,
    Maps,
    Offsets,
    Out,
    xc,
    xt,
    wc,
    ww,
    sr,
    sc,
    st,
    oc,
    ot,
    bound_stride,
    index_stride,
    history_stride,
    map_stride,
    offset_stride,
    bias_stride,
    pad,
    CHANNELS: tl.constexpr,
    WIDTH: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HAS_INDEX: tl.constexpr,
    HAS_HISTORY: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
    HAS_PAD: tl.constexpr,
):
    request = tl.load(Maps + tl.program_id(0) * map_stride).to(tl.int64)
    valid_request = request >= 0
    start = tl.load(Bounds + request * bound_stride, mask=valid_request, other=0).to(
        tl.int64
    )
    end = tl.load(
        Bounds + (request + 1) * bound_stride, mask=valid_request, other=0
    ).to(tl.int64)
    slot = request
    if HAS_INDEX:
        slot = tl.load(
            Indices + request * index_stride, mask=valid_request, other=-1
        ).to(tl.int64)
    active = valid_request & (slot >= 0)
    if HAS_PAD:
        active &= slot != pad
    use_history = tl.full((), False, tl.int1)
    if HAS_HISTORY:
        use_history = tl.load(
            History + request * history_stride, mask=active, other=0
        ).to(tl.int1)
    chunk = tl.load(Offsets + tl.program_id(0) * offset_stride).to(tl.int64)
    local = chunk * CHUNK + tl.arange(0, BLOCK_T).to(tl.int64)
    channel = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C).to(tl.int64)
    live = (
        active
        & (local[:, None] + start < end)
        & (tl.arange(0, BLOCK_T)[:, None] < CHUNK)
        & (channel[None, :] < CHANNELS)
    )
    acc = tl.full((BLOCK_T, BLOCK_C), 0.0, tl.float32)
    if HAS_BIAS:
        acc += tl.load(
            Bias + channel * bias_stride, mask=channel < CHANNELS, other=0
        ).to(tl.float32)[None, :]
    for tap in tl.static_range(WIDTH):
        position = local + tap - (WIDTH - 1)
        current = tl.load(
            X + (start + position[:, None]) * xt + channel[None, :] * xc,
            mask=live & (position[:, None] >= 0),
            other=0,
        ).to(tl.float32)
        prior = tl.load(
            State
            + slot * sr
            + channel[None, :] * sc
            + (position[:, None] + WIDTH - 1) * st,
            mask=live
            & use_history
            & (position[:, None] < 0)
            & (position[:, None] >= -(WIDTH - 1)),
            other=0,
        ).to(tl.float32)
        weight = tl.load(
            Weight + channel * wc + tap * ww, mask=channel < CHANNELS, other=0
        ).to(tl.float32)
        acc += (current + prior) * weight[None, :]
    if SILU:
        acc *= tl.sigmoid(acc)
    tl.store(
        Out + (start + local[:, None]) * ot + channel[None, :] * oc, acc, mask=live
    )


@triton.jit(do_not_specialize=["xc", "xt", "sr", "sc", "st", "bs", "ixs", "hs", "pad"])
def _conv_prefill_state_kernel(
    X,
    State,
    Bounds,
    Indices,
    History,
    xc,
    xt,
    sr,
    sc,
    st,
    bs,
    ixs,
    hs,
    pad,
    CHANNELS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HAS_INDEX: tl.constexpr,
    HAS_HISTORY: tl.constexpr,
    HAS_PAD: tl.constexpr,
):
    request = tl.program_id(0).to(tl.int64)
    start = tl.load(Bounds + request * bs).to(tl.int64)
    end = tl.load(Bounds + (request + 1) * bs).to(tl.int64)
    slot = request
    if HAS_INDEX:
        slot = tl.load(Indices + request * ixs).to(tl.int64)
    active = (slot >= 0) & (end > start)
    if HAS_PAD:
        active &= slot != pad
    history = tl.full((), False, tl.int1)
    if HAS_HISTORY:
        history = tl.load(History + request * hs, mask=active, other=0).to(tl.int1)
    channel = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C).to(tl.int64)
    live = active & (channel < CHANNELS)
    # Increasing destinations read later old positions before they are overwritten.
    for position in tl.static_range(WIDTH - 1):
        relative = end - start - (WIDTH - 1) + position
        current = tl.load(
            X + (start + relative) * xt + channel * xc,
            mask=live & (relative >= 0),
            other=0,
        )
        old = tl.load(
            State + slot * sr + channel * sc + (relative + WIDTH - 1) * st,
            mask=live & history & (relative < 0),
            other=0,
        )
        result = tl.where(relative >= 0, current, old)
        tl.store(State + slot * sr + channel * sc + position * st, result, mask=live)


@triton.jit(
    do_not_specialize=[
        "xb",
        "xc",
        "wc",
        "ww",
        "sr",
        "sc",
        "st",
        "rs",
        "ws",
        "bias_stride",
        "pad",
    ]
)
def _conv_decode_kernel(
    X,
    Weight,
    Bias,
    State,
    Reads,
    Writes,
    xb,
    xc,
    wc,
    ww,
    sr,
    sc,
    st,
    rs,
    ws,
    bias_stride,
    pad,
    CHANNELS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HAS_READS: tl.constexpr,
    HAS_WRITES: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
    HAS_PAD: tl.constexpr,
):
    request = tl.program_id(0).to(tl.int64)
    source = request
    if HAS_READS:
        source = tl.load(Reads + request * rs).to(tl.int64)
    destination = source
    if HAS_WRITES:
        destination = tl.load(Writes + request * ws).to(tl.int64)
    active = source >= 0
    writable = destination >= 0
    if HAS_PAD:
        active &= source != pad
        writable &= destination != pad
    channel = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C).to(tl.int64)
    position = tl.arange(0, BLOCK_H).to(tl.int64)
    live = active & (channel < CHANNELS)
    history_mask = live[None, :] & (position[:, None] < WIDTH - 1)
    old = tl.load(
        State + source * sr + channel[None, :] * sc + position[:, None] * st,
        mask=history_mask,
        other=0,
    ).to(tl.float32)
    weights = tl.load(
        Weight + channel[None, :] * wc + position[:, None] * ww,
        mask=history_mask,
        other=0,
    ).to(tl.float32)
    current = tl.load(X + request * xb + channel * xc, mask=live, other=0)
    last_weight = tl.load(
        Weight + channel * wc + (WIDTH - 1) * ww, mask=live, other=0
    ).to(tl.float32)
    acc = tl.sum(old * weights, 0) + current.to(tl.float32) * last_weight
    if HAS_BIAS:
        acc += tl.load(Bias + channel * bias_stride, mask=live, other=0).to(tl.float32)
    if SILU:
        acc *= tl.sigmoid(acc)
    # Read every old history element before writing a possibly identical slot.
    shifted = tl.load(
        State + source * sr + channel[None, :] * sc + (position[:, None] + 1) * st,
        mask=history_mask & (position[:, None] + 1 < WIDTH - 1),
        other=0,
    )
    next_history = tl.where(position[:, None] == WIDTH - 2, current[None, :], shifted)
    tl.store(
        State + destination * sr + channel[None, :] * sc + position[:, None] * st,
        next_history,
        mask=history_mask & writable,
    )
    tl.store(X + request * xb + channel * xc, acc, mask=live)


_FLOAT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _validate(x, weight, state, bias):
    if x.device.type != "npu" or x.dtype not in _FLOAT_DTYPES:
        raise ValueError("Ascend convolution requires FP16, BF16 or FP32 NPU input")
    if weight.ndim != 2 or weight.shape[0] <= 0 or weight.shape[1] not in (2, 3, 4, 5):
        raise ValueError(
            "convolution weight must be [channels, width] with width 2 to 5"
        )
    channels, width = weight.shape
    if state.ndim != 3:
        raise ValueError("convolution state must be a three-dimensional tensor")
    if state.shape[1] == channels and state.shape[2] >= width - 1:
        normalized = state
    elif state.shape[2] == channels and state.shape[1] >= width - 1:
        normalized = state.transpose(1, 2)
    else:
        raise ValueError("convolution state must hold width-1 raw tokens per channel")
    if normalized.dtype != x.dtype:
        raise ValueError("convolution history must match the input dtype")
    for tensor in (weight, normalized) + (() if bias is None else (bias,)):
        if tensor.device != x.device or tensor.dtype not in _FLOAT_DTYPES:
            raise ValueError(
                "convolution tensors must be floating point on the input device"
            )
    if bias is not None and bias.shape != (channels,):
        raise ValueError("convolution bias must have one value per channel")
    return normalized


def _indices(x, indices, requests):
    if indices is not None and (
        indices.shape != (requests,)
        or indices.dtype not in (torch.int32, torch.int64)
        or indices.device != x.device
    ):
        raise ValueError(
            "convolution indices must be device integer [requests] vectors"
        )


def _activation(activation):
    if activation not in (None, False, True, "silu", "swish"):
        raise ValueError("convolution activation must be None, silu or swish")
    return activation in (True, "silu", "swish")


def _validate_values(state, indices, pad, requests):
    ids = list(range(requests)) if indices is None else indices.cpu().tolist()
    live = [index for index in ids if index >= 0 and index != pad]
    if any(index < 0 or index >= state.shape[0] for index in live) or len(live) != len(
        set(live)
    ):
        raise ValueError("convolution live state slots must be in bounds and unique")


def ascend_causal_conv1d_prefill(
    x,
    weight,
    bias,
    conv_states,
    query_start_loc,
    *,
    cache_indices,
    has_initial_state,
    activation,
    pad_slot_id,
    validate_data,
    prefill_metadata,
):
    """Convolve packed [channels,tokens] input and update raw history in place.

    Args:
        x: FP16/BF16/FP32 packed input [channels,tokens], read-only.
        weight: Per-channel [channels,width] taps ordered oldest to newest.
        bias: Optional [channels] bias.
        conv_states: History pool [slots,channels,history] or [slots,history,channels].
            Its dtype matches x. Only the first width-1 history positions are used.
        query_start_loc: Device integer packed boundaries [requests+1].
        cache_indices: Optional device integer [requests] cache slots, or row order.
        has_initial_state: Optional bool [requests]; omission means fresh zero history.
        activation: None/False for identity or True/silu/swish for SiLU.
        pad_slot_id: Optional sentinel slot id to skip.
        validate_data: Validate metadata contents with host synchronization when True.
        prefill_metadata: Read-only device program-to-request/chunk mapping.

    Returns:
        Convolution output shaped like x. Padding is zero. Live history slots
        are updated with raw input tokens, retaining history for short sequences.
        Empty sequences leave their slots untouched. Slots must be unique and
        writable; boundary/map contents are caller-owned metadata contracts.
    """
    state = _validate(x, weight, conv_states, bias)
    if x.ndim != 2 or x.shape[0] != weight.shape[0]:
        raise ValueError("convolution prefill input must be [channels,tokens]")
    requests = query_start_loc.numel() - 1
    _indices(x, query_start_loc, requests + 1)
    _indices(x, cache_indices, requests)
    if cache_indices is None and requests > state.shape[0]:
        raise ValueError("convolution state pool has fewer slots than requests")
    if has_initial_state is not None and (
        has_initial_state.shape != (requests,)
        or has_initial_state.dtype != torch.bool
        or has_initial_state.device != x.device
    ):
        raise ValueError("initial-history flags must be device bool [requests]")
    maps, offsets = prefill_metadata.batch_indices, prefill_metadata.chunk_offsets
    if prefill_metadata.block_m <= 0 or maps.ndim != 1 or offsets.shape != maps.shape:
        raise ValueError("invalid convolution scheduling metadata")
    for tensor in (maps, offsets):
        if tensor.device != x.device or tensor.dtype not in (torch.int32, torch.int64):
            raise ValueError("convolution maps must be device integer vectors")
    silu = _activation(activation)
    if validate_data:
        bounds = query_start_loc.cpu().tolist()
        if (
            not bounds
            or bounds[0] != 0
            or bounds[-1] > x.shape[1]
            or any(a > b for a, b in zip(bounds, bounds[1:]))
        ):
            raise ValueError("invalid convolution packed boundaries")
        _validate_values(state, cache_indices, pad_slot_id, requests)
        expected = [
            (row, chunk)
            for row in range(requests)
            for chunk in range(
                triton.cdiv(bounds[row + 1] - bounds[row], prefill_metadata.block_m)
            )
        ]
        actual = [
            (row, chunk)
            for row, chunk in zip(maps.cpu().tolist(), offsets.cpu().tolist())
            if row >= 0
        ]
        if sorted(actual) != expected:
            raise ValueError("convolution program maps do not cover packed sequences")
    out = torch.zeros_like(x)
    if requests == 0:
        return out
    flags = dict(
        CHANNELS=weight.shape[0],
        WIDTH=weight.shape[1],
        BLOCK_C=64,
        HAS_INDEX=cache_indices is not None,
        HAS_HISTORY=has_initial_state is not None,
        HAS_PAD=pad_slot_id is not None,
    )
    index_arg = query_start_loc if cache_indices is None else cache_indices
    history_arg = query_start_loc if has_initial_state is None else has_initial_state
    sentinel = -1 if pad_slot_id is None else pad_slot_id
    if maps.numel():
        _conv_prefill_kernel[(maps.numel(), triton.cdiv(weight.shape[0], 64))](
            x,
            weight,
            weight if bias is None else bias,
            state,
            query_start_loc,
            index_arg,
            history_arg,
            maps,
            offsets,
            out,
            *x.stride(),
            *weight.stride(),
            *state.stride(),
            *out.stride(),
            query_start_loc.stride(0),
            index_arg.stride(0),
            history_arg.stride(0),
            maps.stride(0),
            offsets.stride(0),
            0 if bias is None else bias.stride(0),
            sentinel,
            CHUNK=prefill_metadata.block_m,
            BLOCK_T=triton.next_power_of_2(prefill_metadata.block_m),
            HAS_BIAS=bias is not None,
            SILU=silu,
            **flags,
        )
    # The second launch orders all history reads before any slot is overwritten.
    _conv_prefill_state_kernel[(requests, triton.cdiv(weight.shape[0], 64))](
        x,
        state,
        query_start_loc,
        index_arg,
        history_arg,
        *x.stride(),
        *state.stride(),
        query_start_loc.stride(0),
        index_arg.stride(0),
        history_arg.stride(0),
        sentinel,
        **flags,
    )
    return out


def ascend_causal_conv1d_update(
    x,
    conv_state,
    weight,
    bias,
    activation,
    cache_seqlens,
    conv_state_indices,
    num_accepted_tokens,
    intermediate_conv_window,
    output_state_indices,
    pad_slot_id,
    validate_data,
    *,
    parent_indices,
):
    """Convolve one token in place and advance cache-managed raw history.

    Args:
        x: [requests,channels] or [requests,channels,1], overwritten with output.
        conv_state: History pool, in either supported prefill layout.
        weight: [channels,width] oldest-to-newest taps, width 2 to 5.
        bias: Optional [channels] bias.
        activation: None/False or True/silu/swish.
        cache_seqlens: Unsupported circular-history argument; must be None.
        conv_state_indices: Optional integer [requests] read slots, or row order.
        num_accepted_tokens: Unsupported speculative argument; must be None.
        intermediate_conv_window: Unsupported speculative output; must be None.
        output_state_indices: Optional [requests] or [requests,1] destinations.
            Omission updates each read slot in place; negative destinations skip writes.
        pad_slot_id: Optional sentinel; padded reads leave x/history untouched.
        validate_data: Validate index values with host synchronization when True.
        parent_indices: Unsupported tree argument; must be None.

    Returns:
        x itself, with raw history advanced before applying bias/activation.
        Destinations must be unique and cannot overwrite another request's
        input checkpoint. Source and destination may coincide for one request.
    """
    if any(
        value is not None
        for value in (
            cache_seqlens,
            num_accepted_tokens,
            intermediate_conv_window,
            parent_indices,
        )
    ):
        raise NotImplementedError(
            "Ascend convolution does not support circular or speculative updates"
        )
    state = _validate(x, weight, conv_state, bias)
    if (
        x.ndim not in (2, 3)
        or x.shape[1] != weight.shape[0]
        or (x.ndim == 3 and x.shape[2] != 1)
    ):
        raise ValueError("Ascend convolution decode requires one token per request")
    base = x if x.ndim == 2 else x[:, :, 0]
    requests = x.shape[0]
    _indices(x, conv_state_indices, requests)
    if conv_state_indices is None and requests > state.shape[0]:
        raise ValueError("convolution state pool has fewer slots than requests")
    writes = output_state_indices
    if writes is not None:
        if writes.shape == (requests, 1):
            writes = writes[:, 0]
        _indices(x, writes, requests)
    if validate_data:
        # Read slots may be shared checkpoints; write slots must be unique.
        read_ids = (
            list(range(requests))
            if conv_state_indices is None
            else conv_state_indices.cpu().tolist()
        )
        if any(index != pad_slot_id and index >= state.shape[0] for index in read_ids):
            raise ValueError("convolution read slots must be in bounds")
        _validate_values(
            state,
            conv_state_indices if writes is None else writes,
            pad_slot_id,
            requests,
        )
    silu = _activation(activation)
    if requests:
        reads_arg = base if conv_state_indices is None else conv_state_indices
        writes_arg = base if writes is None else writes
        _conv_decode_kernel[(requests, triton.cdiv(weight.shape[0], 64))](
            base,
            weight,
            weight if bias is None else bias,
            state,
            reads_arg,
            writes_arg,
            *base.stride(),
            *weight.stride(),
            *state.stride(),
            0 if conv_state_indices is None else conv_state_indices.stride(0),
            0 if writes is None else writes.stride(0),
            0 if bias is None else bias.stride(0),
            -1 if pad_slot_id is None else pad_slot_id,
            CHANNELS=weight.shape[0],
            WIDTH=weight.shape[1],
            BLOCK_H=triton.next_power_of_2(weight.shape[1] - 1),
            BLOCK_C=64,
            HAS_READS=conv_state_indices is not None,
            HAS_WRITES=writes is not None,
            HAS_BIAS=bias is not None,
            SILU=silu,
            HAS_PAD=pad_slot_id is not None,
        )
    return x
