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

"""Ascend cache-managed recurrent state preparation and writeback."""

import torch
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit(do_not_specialize=["cr", "cc", "ct", "sr", "sh", "sa", "sb", "ins", "outs"])
def _prepare_state_kernel(
    Conv,
    State,
    Inputs,
    Outputs,
    Gathered,
    History,
    cr,
    cc,
    ct,
    sr,
    sh,
    sa,
    sb,
    ins,
    outs,
    CONV_WIDTH: tl.constexpr,
    WINDOW: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    A: tl.constexpr,
    B: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    source = tl.load(Inputs + row * ins).to(tl.int64)
    destination = tl.load(Outputs + row * outs).to(tl.int64)
    history = (source > 0) & (destination > 0)
    if tl.program_id(1) == 0:
        tl.store(History + row, history)
    offset = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    conv_offset = offset // WINDOW * cc + offset % WINDOW * ct
    copy = history & (source != destination) & (offset < CONV_WIDTH)
    window = tl.load(Conv + source * cr + conv_offset, mask=copy, other=0)
    tl.store(Conv + destination * cr + conv_offset, window, mask=copy)
    state_offset = offset // (A * B) * sh + offset // B % A * sa + offset % B * sb
    state = tl.load(
        State + source * sr + state_offset,
        mask=history & (offset < STATE_WIDTH),
        other=0,
    )
    tl.store(Gathered + row * STATE_WIDTH + offset, state, mask=offset < STATE_WIDTH)


@triton.jit(do_not_specialize=["sr", "sh", "sa", "sb", "dr", "dh", "da", "db", "ids"])
def _store_state_kernel(
    State,
    Slab,
    Blocks,
    sr,
    sh,
    sa,
    sb,
    dr,
    dh,
    da,
    db,
    ids,
    WIDTH: tl.constexpr,
    A: tl.constexpr,
    B: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    destination = tl.load(Blocks + row * ids).to(tl.int64)
    offset = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    live = (destination >= 0) & (offset < WIDTH)
    head, a, b = offset // (A * B), offset // B % A, offset % B
    value = tl.load(State + row * sr + head * sh + a * sa + b * sb, mask=live, other=0)
    tl.store(Slab + destination * dr + head * dh + a * da + b * db, value, mask=live)


_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _indices(indices, device, rows):
    if (
        indices.shape != (rows,)
        or indices.device != device
        or indices.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            "state block indices must be device integer [requests] vectors"
        )


def prepare_prefill_state_inputs(
    conv_states, ssm_states, state_in_blocks, state_out_blocks
):
    """Prepare cache-managed inputs, copying only resumed live conv windows.

    Args:
        conv_states: Floating strided [blocks,channels,window] NPU cache.
        ssm_states: Floating strided [blocks,heads,A,B] NPU recurrent cache.
            A/B preserve the caller's ordering, including K-last GDN state.
        state_in_blocks: Integer [requests] source ids; <=0 means zero history.
        state_out_blocks: Integer [requests] destinations. Nonpositive rows
            are inactive padding and access no cache. Positive destinations
            must be unique and cannot overwrite another request's checkpoint.

    Returns:
        Contiguous [requests,heads,A,B] recurrent inputs and bool history flags.
        Fresh and padded recurrent rows are exactly zero, including with NaN
        null blocks. Fresh conv windows and all recurrent cache rows are unchanged.
        A request may evolve its own live row in place. Index values are
        caller-owned bounds/ownership contracts; no host read is performed.
    """
    if (
        conv_states.ndim != 3
        or ssm_states.ndim != 4
        or any(size <= 0 for size in (*conv_states.shape[1:], *ssm_states.shape[1:]))
    ):
        raise ValueError(
            "prefill caches require positive 3D conv and 4D state geometry"
        )
    if (
        conv_states.device.type != "npu"
        or ssm_states.device != conv_states.device
        or conv_states.dtype not in _DTYPES
        or ssm_states.dtype not in _DTYPES
    ):
        raise ValueError("prefill caches must be floating tensors on one Ascend device")
    rows = state_in_blocks.numel()
    _indices(state_in_blocks, conv_states.device, rows)
    _indices(state_out_blocks, conv_states.device, rows)
    recurrent = torch.empty(
        (rows, *ssm_states.shape[1:]), device=ssm_states.device, dtype=ssm_states.dtype
    )
    history = torch.empty(rows, device=ssm_states.device, dtype=torch.bool)
    if rows:
        conv_width = conv_states.shape[1] * conv_states.shape[2]
        state_width = ssm_states.shape[1] * ssm_states.shape[2] * ssm_states.shape[3]
        _prepare_state_kernel[(rows, triton.cdiv(max(conv_width, state_width), 256))](
            conv_states,
            ssm_states,
            state_in_blocks,
            state_out_blocks,
            recurrent,
            history,
            *conv_states.stride(),
            *ssm_states.stride(),
            state_in_blocks.stride(0),
            state_out_blocks.stride(0),
            CONV_WIDTH=conv_width,
            WINDOW=conv_states.shape[2],
            STATE_WIDTH=state_width,
            A=ssm_states.shape[2],
            B=ssm_states.shape[3],
            BLOCK=256
        )
    return recurrent, history


def store_state_rows(states, slab, blocks):
    """Write per-request state into live cache rows, skipping negative ids.

    Args:
        states: Floating strided [requests,heads,A,B] NPU states to store.
        slab: Floating strided [blocks,heads,A,B] NPU cache, mutated in place.
        blocks: Integer [requests] destinations. Nonnegative destinations must
            be unique, in bounds and writable; negative ids skip all access.
            Unlike the state-table resolver, this generic scatter permits id 0.

    Returns:
        None. Stored values cast to slab.dtype; other cache rows remain unchanged.
        Tensor/index metadata is validated without host reads of index values.
    """
    if states.ndim != 4 or slab.ndim != 4 or states.shape[1:] != slab.shape[1:]:
        raise ValueError("recurrent states and slab must have matching 4D geometry")
    if (
        states.device.type != "npu"
        or slab.device != states.device
        or states.dtype not in _DTYPES
        or slab.dtype not in _DTYPES
    ):
        raise ValueError("writeback requires floating tensors on one Ascend device")
    _indices(blocks, states.device, states.shape[0])
    width = states.shape[1] * states.shape[2] * states.shape[3]
    if states.shape[0] and width:
        _store_state_kernel[(states.shape[0], triton.cdiv(width, 256))](
            states,
            slab,
            blocks,
            *states.stride(),
            *slab.stride(),
            blocks.stride(0),
            WIDTH=width,
            A=states.shape[2],
            B=states.shape[3],
            BLOCK=256
        )
