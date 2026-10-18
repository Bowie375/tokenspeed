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

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_npu")

from tokenspeed_kernel.ops.attention.gdn.triton import (
    CAUSAL_CONV1D_BLOCK_M,
    build_causal_conv1d_prefill_metadata,
)

from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires Ascend")


def _reference(x, weight, bias, history, activation):
    joined = torch.cat((history.cpu().double(), x.cpu().double()), dim=-1)
    out = F.conv1d(
        joined.unsqueeze(0),
        weight.cpu().double().unsqueeze(1),
        bias=None if bias is None else bias.cpu().double(),
        groups=weight.shape[0],
    )[0]
    if activation in ("silu", "swish", True):
        out = F.silu(out)
    return out.to(x.dtype), joined[:, -(weight.shape[1] - 1) :].to(x.dtype)


def _check(actual, expected):
    tolerance = {torch.bfloat16: 2**-7, torch.float16: 2**-10, torch.float32: 2e-5}[
        actual.dtype
    ]
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=tolerance, atol=2e-6)


def _metadata(lengths):
    lens = torch.tensor(lengths, dtype=torch.int32)
    boundaries = torch.cat(
        (torch.zeros(1, dtype=torch.int32), lens.cumsum(0).int())
    ).to("npu")
    return boundaries, build_causal_conv1d_prefill_metadata(
        boundaries, lens, CAUSAL_CONV1D_BLOCK_M
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("width_major", [False, True])
@torch.inference_mode()
def test_prefill_outputs_and_history(dtype, activation, width_major):
    torch.manual_seed(9)
    lengths = [1, 0, 3, 9]
    channels = 7
    x = torch.randn(sum(lengths), channels + 5, device="npu", dtype=dtype)[
        :, :channels
    ].T
    original_x = x.clone()
    weight = torch.randn(channels, 4, device="npu", dtype=dtype) * 0.2
    bias = torch.randn(channels, device="npu", dtype=dtype)
    state = torch.randn(6, channels, 3, device="npu", dtype=dtype)
    if width_major:
        state = state.transpose(1, 2).contiguous().transpose(1, 2)
    original = state.clone()
    indices = torch.tensor([1, 2, 3, 4], device="npu", dtype=torch.int32)
    history = torch.tensor([True, False, False, True], device="npu")
    bounds, metadata = _metadata(lengths)
    cache = state.transpose(1, 2) if width_major else state
    actual = causal_conv1d_fn(
        x,
        weight,
        bias,
        cache,
        bounds,
        cache_indices=indices,
        has_initial_state=history,
        activation=activation,
        prefill_metadata=metadata,
    )
    offset = 0
    for row, length in enumerate(lengths):
        if length:
            initial = (
                original[row + 1]
                if row in (0, 3)
                else torch.zeros_like(original[row + 1])
            )
            expected, final = _reference(
                original_x[:, offset : offset + length],
                weight,
                bias,
                initial,
                activation,
            )
            _check(actual[:, offset : offset + length], expected)
            torch.testing.assert_close(
                state[row + 1],
                final.to("npu"),
                rtol=0,
                atol=0,
            )
        else:
            torch.testing.assert_close(
                state[row + 1], original[row + 1], rtol=0, atol=0
            )
        offset += length
    torch.testing.assert_close(state[[0, 5]], original[[0, 5]], rtol=0, atol=0)
    torch.testing.assert_close(x, original_x, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("activation", [None, "silu"])
@torch.inference_mode()
def test_decode_reads_checkpoint_writes_live_and_mutates_input(dtype, activation):
    torch.manual_seed(10)
    x = torch.randn(3, 12, device="npu", dtype=dtype)[:, :7]
    original_x = x.clone()
    weight = torch.randn(7, 4, device="npu", dtype=dtype) * 0.2
    state = torch.randn(6, 7, 3, device="npu", dtype=dtype)
    original = state.clone()
    reads = torch.tensor([1, -1, 2], device="npu", dtype=torch.int32)
    writes = torch.tensor([[3], [-1], [4]], device="npu", dtype=torch.int32)
    result = causal_conv1d_update(
        x,
        state,
        weight,
        activation=activation,
        conv_state_indices=reads,
        output_state_indices=writes,
        parent_indices=None,
    )
    assert result is x
    for row, source, destination in ((0, 1, 3), (2, 2, 4)):
        expected, final = _reference(
            original_x[row, :, None], weight, None, original[source], activation
        )
        _check(result[row], expected[:, 0])
        torch.testing.assert_close(state[destination], final.to("npu"), rtol=0, atol=0)
    torch.testing.assert_close(x[1], original_x[1], rtol=0, atol=0)
    torch.testing.assert_close(
        state[[0, 1, 2, 5]], original[[0, 1, 2, 5]], rtol=0, atol=0
    )


@pytest.mark.parametrize("channels", [7, 6144])
@torch.inference_mode()
def test_prefill_matches_repeated_decode(channels):
    torch.manual_seed(11)
    x = torch.randn(channels, 13, device="npu", dtype=torch.bfloat16)
    weights = torch.randn(channels, 4, device="npu", dtype=x.dtype) * 0.2
    initial = torch.randn(1, channels, 3, device="npu", dtype=x.dtype)
    prefill_state, decode_state = initial.clone(), initial.clone()
    bounds, metadata = _metadata([13])
    prefill = causal_conv1d_fn(
        x,
        weights,
        None,
        prefill_state,
        bounds,
        has_initial_state=torch.ones(1, device="npu", dtype=torch.bool),
        activation="silu",
        prefill_metadata=metadata,
    )
    outputs = []
    for token in range(13):
        step = x[:, token].reshape(1, channels).clone()
        outputs.append(
            causal_conv1d_update(
                step, decode_state, weights, activation="silu", parent_indices=None
            ).T
        )
    _check(torch.cat(outputs, dim=1), prefill)
    torch.testing.assert_close(prefill_state, decode_state, rtol=0, atol=0)


@pytest.mark.parametrize("width", [2, 3, 5])
@torch.inference_mode()
def test_kernel_width_and_bias_decode(width):
    channels = 9
    x = torch.randn(channels, 2, device="npu")
    weight = torch.randn(channels, width, device="npu")
    bias = torch.randn(channels, device="npu")
    state = torch.randn(1, channels, width - 1, device="npu")
    original = state.clone()
    bounds, metadata = _metadata([2])
    actual = causal_conv1d_fn(
        x,
        weight,
        bias,
        state,
        bounds,
        has_initial_state=torch.ones(1, device="npu", dtype=torch.bool),
        activation="silu",
        prefill_metadata=metadata,
        validate_data=True,
    )
    expected, final = _reference(x, weight, bias, original[0], "silu")
    _check(actual, expected)
    torch.testing.assert_close(state[0].cpu(), final, rtol=0, atol=0)
    step = x[:, :1].T.unsqueeze(-1).clone()
    expected, final = _reference(step[0], weight, bias, state[0], "silu")
    causal_conv1d_update(
        step, state, weight, bias=bias, activation="silu", parent_indices=None
    )
    _check(step[0], expected)
    torch.testing.assert_close(state[0].cpu(), final, rtol=0, atol=0)


@torch.inference_mode()
def test_empty_and_padded_prefill():
    for lengths in ([], [0, 0], [1, 2]):
        x = torch.randn(7, sum(lengths), device="npu")
        state = torch.randn(3, 7, 3, device="npu")
        original = state.clone()
        bounds, metadata = _metadata(lengths)
        ids = torch.full((len(lengths),), -1, device="npu", dtype=torch.int32)
        out = causal_conv1d_fn(
            x,
            torch.ones(7, 4, device="npu"),
            None,
            state,
            bounds,
            cache_indices=ids,
            prefill_metadata=metadata,
        )
        torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)
        torch.testing.assert_close(state, original, rtol=0, atol=0)


@pytest.mark.parametrize(
    "argument", ["cache_seqlens", "num_accepted_tokens", "intermediate_conv_window"]
)
@torch.inference_mode()
def test_unsupported_updates_reject_before_mutation(argument):
    x = torch.randn(1, 7, device="npu")
    state = torch.randn(1, 7, 3, device="npu")
    original_x, original_state = x.clone(), state.clone()
    with pytest.raises(NotImplementedError, match="circular or speculative"):
        causal_conv1d_update(
            x,
            state,
            torch.ones(7, 4, device="npu"),
            parent_indices=None,
            **{argument: torch.zeros(1, device="npu", dtype=torch.int32)},
        )
    torch.testing.assert_close(x, original_x, rtol=0, atol=0)
    torch.testing.assert_close(state, original_state, rtol=0, atol=0)


@pytest.mark.parametrize("decode", [False, True])
@torch.inference_mode()
def test_variable_batch_geometry_does_not_recompile(decode):
    import sys
    from pathlib import Path

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2] / "tokenspeed-kernel/test")
    )
    from tokenspeed_kernel_npu.ops.causal_conv1d import (
        _conv_decode_kernel,
        _conv_prefill_kernel,
        _conv_prefill_state_kernel,
    )
    from utils import assert_no_triton_compile

    def run(requests, tokens):
        channels = 7
        weights = torch.ones(channels, 4, device="npu", dtype=torch.bfloat16)
        state = torch.zeros(requests, channels, 3, device="npu", dtype=weights.dtype)
        if decode:
            x = torch.randn(
                requests, channels + requests, device="npu", dtype=weights.dtype
            )[:, :channels]
            causal_conv1d_update(x, state, weights, parent_indices=None)
        else:
            x = torch.randn(
                requests * tokens,
                channels + requests,
                device="npu",
                dtype=weights.dtype,
            )[:, :channels].T
            bounds, metadata = _metadata([tokens] * requests)
            causal_conv1d_fn(x, weights, None, state, bounds, prefill_metadata=metadata)
        torch.npu.synchronize()

    run(1, 1)
    run(3, 7)
    kernels = (
        [_conv_decode_kernel]
        if decode
        else [_conv_prefill_kernel, _conv_prefill_state_kernel]
    )
    with assert_no_triton_compile(*kernels):
        for requests, tokens in ((2, 3), (5, 17), (4, 65)):
            run(requests, tokens)


@torch.inference_mode()
def test_negative_destination_skips_write_with_validation():
    x = torch.ones(1, 7, device="npu")
    state = torch.ones(1, 7, 3, device="npu")
    original = state.clone()
    out = causal_conv1d_update(
        x,
        state,
        torch.ones(7, 4, device="npu"),
        output_state_indices=torch.tensor([-2], device="npu", dtype=torch.int32),
        validate_data=True,
        parent_indices=None,
    )
    torch.testing.assert_close(out, torch.full_like(out, 4), rtol=0, atol=0)
    torch.testing.assert_close(state, original, rtol=0, atol=0)
