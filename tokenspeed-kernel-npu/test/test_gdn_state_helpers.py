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

pytest.importorskip("torch_npu")

from tokenspeed_kernel.ops.attention.gdn import (
    build_causal_conv1d_prefill_metadata,
    prepare_prefill_state_inputs,
)
from tokenspeed_kernel.ops.kvcache import store_state_rows

from tokenspeed.runtime.execution.breakable_cuda_graph import scrub_padding_tail
from tokenspeed.runtime.layers.attention.backends.state.checkpoint import (
    compute_state_block_indices,
)

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires Ascend")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@torch.inference_mode()
def test_gather_resumed_fresh_and_padded_rows(dtype):
    conv = torch.randn(16, 14, 6, device="npu", dtype=dtype)[::2, ::2, ::2]
    slab = torch.randn(16, 4, 10, 14, device="npu")[::2, ::2, ::2, ::2].transpose(
        -1, -2
    )
    conv[0] = float("nan")
    slab[0] = float("nan")
    conv_before, slab_before = conv.cpu().clone(), slab.cpu().clone()
    sources = torch.tensor([0, 2, 3, 2, 2], device="npu", dtype=torch.int32)
    destinations = torch.tensor([1, 4, 3, 5, -1], device="npu", dtype=torch.int64)
    gathered, history = prepare_prefill_state_inputs(conv, slab, sources, destinations)
    expected = torch.zeros(5, *slab.shape[1:])
    expected[1], expected[2], expected[3] = (
        slab_before[2],
        slab_before[3],
        slab_before[2],
    )
    expected_conv = conv_before.clone()
    expected_conv[4], expected_conv[5] = conv_before[2], conv_before[2]
    torch.testing.assert_close(gathered.cpu(), expected, rtol=0, atol=0)
    assert history.cpu().tolist() == [False, True, True, True, False]
    assert gathered.is_contiguous()
    torch.testing.assert_close(
        conv.cpu(), expected_conv, rtol=0, atol=0, equal_nan=True
    )
    torch.testing.assert_close(slab.cpu(), slab_before, rtol=0, atol=0, equal_nan=True)


@torch.inference_mode()
def test_writeback_strided_rows_skips_negative_destinations():
    states = torch.randn(5, 2, 7, 5, device="npu").transpose(-1, -2)
    states[2] = float("nan")
    slab = torch.randn(14, 4, 10, 14, device="npu")[::2, ::2, ::2, ::2]
    original = slab.cpu().clone()
    blocks = torch.tensor([3, 1, -1, 5, -2], device="npu", dtype=torch.int32)
    store_state_rows(states, slab, blocks)
    expected = original.clone()
    for row, destination in ((0, 3), (1, 1), (3, 5)):
        expected[destination] = states[row].cpu()
    torch.testing.assert_close(slab.cpu(), expected, rtol=0, atol=0)


@torch.inference_mode()
def test_two_slot_block_resolution():
    table = torch.tensor(
        [[0, 1, -1, -1], [2, 4, -1, -1], [0, 3, -1, -1], [-1, -1, -1, -1]],
        device="npu",
        dtype=torch.int64,
    )
    before = torch.tensor([0, 7, 8, 0], device="npu", dtype=torch.int32)
    sources, destinations = compute_state_block_indices(
        table, before, validate=False, group_id="gdn"
    )
    assert sources.cpu().tolist() == [0, 2, 3, 0]
    assert destinations.cpu().tolist() == [1, 4, 3, -1]
    assert sources.dtype == destinations.dtype == torch.int32
    assert sources.is_contiguous() and destinations.is_contiguous()


@torch.inference_mode()
def test_block_validation_rejects_cross_request_checkpoint_overwrite():
    table = torch.tensor([[2, 3], [3, 4]], device="npu", dtype=torch.int32)
    with pytest.raises(ValueError, match="checkpoint|input"):
        compute_state_block_indices(
            table,
            torch.ones(2, device="npu", dtype=torch.int32),
            validate=True,
            group_id="gdn",
        )


@pytest.mark.parametrize("device", ["cpu", "npu"])
@pytest.mark.parametrize("lengths", [[], [0, 0], [1, 0, 8, 9, 17]])
@torch.inference_mode()
def test_prefill_maps_exact_ragged_sequence_coverage(lengths, device):
    lens = torch.tensor(lengths, dtype=torch.int64)
    host = torch.cat((torch.zeros(1, dtype=torch.int64), lens.cumsum(0)))
    metadata = build_causal_conv1d_prefill_metadata(host.to(device), lens, 8)
    expected = [
        (request, chunk)
        for request, length in enumerate(lengths)
        for chunk in range((length + 7) // 8)
    ]
    assert (
        list(
            zip(
                metadata.batch_indices.cpu().tolist(),
                metadata.chunk_offsets.cpu().tolist(),
            )
        )
        == expected
    )
    torch.testing.assert_close(metadata.cu_seqlens_cpu, host, rtol=0, atol=0)


@pytest.mark.parametrize("real", [0, 2, 5, None])
@torch.inference_mode()
def test_padding_scrub_preserves_live_rows_and_zeroes_nan_tail(real):
    backing = torch.randn(5, 6, device="npu")
    view = backing[:, ::2]
    if real is not None:
        view[real:] = float("nan")
    original = backing.cpu().clone()
    scrub_padding_tail(real, view, None)
    expected = original.clone()
    if real is not None:
        expected[real:, ::2] = 0
    torch.testing.assert_close(backing.cpu(), expected, rtol=0, atol=0, equal_nan=True)


@torch.inference_mode()
def test_empty_state_requests():
    conv = torch.zeros(2, 7, 3, device="npu")
    slab = torch.zeros(2, 2, 7, 5, device="npu")
    ids = torch.empty(0, device="npu", dtype=torch.int32)
    gathered, history = prepare_prefill_state_inputs(conv, slab, ids, ids)
    assert gathered.shape == (0, 2, 7, 5) and history.shape == (0,)
    store_state_rows(gathered, slab, ids)
    torch.testing.assert_close(slab, torch.zeros_like(slab), rtol=0, atol=0)


@torch.inference_mode()
def test_writeback_cast_and_strided_indices():
    states = torch.randn(2, 2, 7, 5, device="npu")
    slab = torch.zeros(4, 2, 7, 5, device="npu", dtype=torch.bfloat16)
    blocks = torch.tensor([2, 99, 3, 99], device="npu", dtype=torch.int32)[::2]
    store_state_rows(states, slab, blocks)
    torch.testing.assert_close(slab[2:], states.bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(slab[:2], torch.zeros_like(slab[:2]), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["prepare", "store", "metadata"])
@torch.inference_mode()
def test_variable_request_counts_and_strides_do_not_recompile(kind):
    import sys
    from pathlib import Path

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2] / "tokenspeed-kernel/test")
    )
    from tokenspeed_kernel.ops.attention.gdn._triton.causal_conv1d_metadata import (
        _build_causal_conv1d_metadata_kernel,
    )
    from tokenspeed_kernel_npu.ops.state import (
        _prepare_state_kernel,
        _store_state_kernel,
    )
    from utils import assert_no_triton_compile

    def run(requests, tokens):
        if kind == "metadata":
            lengths = torch.full((requests,), tokens, dtype=torch.int64)
            bounds = torch.cat(
                (torch.zeros(1, dtype=torch.int64), lengths.cumsum(0))
            ).to("npu")
            metadata = build_causal_conv1d_prefill_metadata(bounds, lengths, 8)
            expected = [
                (row, chunk)
                for row in range(requests)
                for chunk in range((tokens + 7) // 8)
            ]
            assert (
                list(
                    zip(
                        metadata.batch_indices.cpu().tolist(),
                        metadata.chunk_offsets.cpu().tolist(),
                    )
                )
                == expected
            )
        elif kind == "prepare":
            conv = torch.randn(2 * requests + 1, 7, 3 + requests, device="npu")[
                :, :, :3
            ]
            slab = torch.randn(2 * requests + 1, 2, 7, 5 + requests, device="npu")[
                :, :, :, :5
            ]
            sources = torch.arange(1, requests + 1, device="npu", dtype=torch.int32)
            destinations = sources + requests
            before = slab.cpu().clone()
            recurrent, flags = prepare_prefill_state_inputs(
                conv, slab, sources, destinations
            )
            torch.testing.assert_close(
                recurrent.cpu(), before[1 : requests + 1], rtol=0, atol=0
            )
            assert flags.cpu().tolist() == [True] * requests
        else:
            states = torch.randn(requests, 2, 7, 5 + requests, device="npu")[
                :, :, :, :5
            ]
            slab = torch.zeros(requests + 1, 2, 7, 5 + requests, device="npu")[
                :, :, :, :5
            ]
            destinations = torch.arange(
                1, requests + 1, device="npu", dtype=torch.int64
            )
            store_state_rows(states, slab, destinations)
            torch.testing.assert_close(slab[1:], states, rtol=0, atol=0)
        torch.npu.synchronize()

    kernel = {
        "prepare": _prepare_state_kernel,
        "store": _store_state_kernel,
        "metadata": _build_causal_conv1d_metadata_kernel,
    }[kind]
    run(1, 1)
    with assert_no_triton_compile(kernel):
        for requests, tokens in ((2, 9), (3, 17), (5, 25)):
            run(requests, tokens)


@torch.inference_mode()
def test_negative_padding_extent_rejected_without_modification():
    x = torch.ones(3, 4, device="npu")
    with pytest.raises(ValueError, match="nonnegative"):
        scrub_padding_tail(-1, x)
    torch.testing.assert_close(x, torch.ones_like(x), rtol=0, atol=0)


@torch.inference_mode()
def test_chunked_prefill_resumes_checkpoint_without_overwriting_it():
    from tokenspeed_kernel.ops.attention.gdn import (
        fused_qkv_split_gdn_prefill,
        gdn_chunk_prefill,
    )

    from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
        causal_conv1d_fn,
    )
    from tokenspeed.runtime.layers.attention.linear.gdn import fused_gdn_gating

    torch.manual_seed(72)
    raw = torch.randn(5, 72, device="npu") * 0.1
    a, b = torch.randn(5, 4, device="npu"), torch.randn(5, 4, device="npu")
    rates, bias = torch.zeros(4, device="npu"), torch.zeros(4, device="npu")
    weights = torch.randn(72, 4, device="npu") * 0.2
    conv = torch.full((3, 72, 3), float("nan"), device="npu")
    slab = torch.full((3, 4, 11, 7), float("nan"), device="npu")
    outputs = []

    def scan(inputs, recurrent, history, conv_pool, slot, start, length):
        bounds = torch.tensor([0, length], device="npu", dtype=torch.int32)
        metadata = build_causal_conv1d_prefill_metadata(
            bounds, torch.tensor([length], dtype=torch.int64), 8
        )
        processed = causal_conv1d_fn(
            inputs.T,
            weights,
            None,
            conv_pool,
            bounds,
            cache_indices=torch.tensor([slot], device="npu", dtype=torch.int32),
            has_initial_state=history,
            activation="silu",
            prefill_metadata=metadata,
        )
        q, k, v = fused_qkv_split_gdn_prefill(
            processed.T, 2, 2, 4, 7, 7, 11, fuse_l2norm=False
        )
        return gdn_chunk_prefill(
            q,
            k,
            v,
            fused_gdn_gating(rates, a[start : start + length], bias).unsqueeze(0),
            b[start : start + length].sigmoid().unsqueeze(0),
            scale=None,
            initial_state=recurrent,
            cu_seqlens=bounds,
            qk_l2norm=True,
            output_final_state=True,
            output_h=False,
        )

    snapshot_conv, snapshot_state = None, None
    for start, length, checkpoint, live in ((0, 3, 0, 1), (3, 2, 1, 2)):
        table = torch.tensor([[checkpoint, live]], device="npu", dtype=torch.int32)
        source, destination = compute_state_block_indices(
            table,
            torch.tensor([start], device="npu", dtype=torch.int32),
            validate=True,
            group_id="gdn",
        )
        recurrent, history = prepare_prefill_state_inputs(
            conv, slab, source, destination
        )
        result = scan(
            raw[start : start + length], recurrent, history, conv, live, start, length
        )
        store_state_rows(result.final_state, slab, destination)
        outputs.append(result.out)
        if live == 1:
            snapshot_conv, snapshot_state = conv[1].clone(), slab[1].clone()
    full_conv = torch.zeros(1, 72, 3, device="npu")
    expected = scan(
        raw,
        torch.zeros(1, 4, 11, 7, device="npu"),
        torch.zeros(1, device="npu", dtype=torch.bool),
        full_conv,
        0,
        0,
        5,
    )
    torch.testing.assert_close(
        torch.cat(outputs, dim=1), expected.out, rtol=3e-5, atol=3e-6
    )
    torch.testing.assert_close(slab[2], expected.final_state[0], rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(conv[2], full_conv[0], rtol=0, atol=0)
    torch.testing.assert_close(conv[1], snapshot_conv, rtol=0, atol=0)
    torch.testing.assert_close(slab[1], snapshot_state, rtol=0, atol=0)
    assert torch.isnan(conv[0]).all() and torch.isnan(slab[0]).all()
