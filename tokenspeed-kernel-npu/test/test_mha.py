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

import math

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from tokenspeed_kernel_npu.ops.mha import (
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
    mha_prefill,
)

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend MHA tests require an NPU"
)


def _attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool
) -> torch.Tensor:
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    scores = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(q.shape[-1])
    if causal:
        q_positions = (
            k.shape[0] - q.shape[0] + torch.arange(q.shape[0], device=q.device)
        )
        k_positions = torch.arange(k.shape[0], device=q.device)
        scores.masked_fill_(
            ~(k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)).unsqueeze(0),
            torch.finfo(q.dtype).min,
        )
    probabilities = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.einsum("hqk,khd->qhd", probabilities, v)


def _paged_rows(
    cache: torch.Tensor, table: torch.Tensor, batch: int, length: int
) -> torch.Tensor:
    pages = cache[table[batch].long()].reshape(-1, cache.shape[-2], cache.shape[-1])
    return pages[:length]


@pytest.mark.parametrize("head_dim", [64, 256])
def test_mha_prefill(head_dim) -> None:
    lengths = [3, 5]
    cumulative = torch.tensor([0, 3, 8], dtype=torch.int32, device="npu")
    q = torch.randn(8, 4, head_dim, dtype=torch.bfloat16, device="npu")
    k = torch.randn(8, 2, head_dim, dtype=torch.bfloat16, device="npu")
    v = torch.randn_like(k)

    output = mha_prefill(q, k, v, cumulative, [0, 3, 8], 5)
    expected = torch.cat(
        [
            _attention(q[:3], k[:3], v[:3], causal=True),
            _attention(q[3:], k[3:], v[3:], causal=True),
        ]
    )

    torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("head_dim", [64, 256])
def test_mha_extend_with_paged_cache(head_dim) -> None:
    page_size = 128
    q_lengths = [2, 3]
    cache_lengths = torch.tensor([5, 70], dtype=torch.int32, device="npu")
    cumulative_q = torch.tensor([0, 2, 5], dtype=torch.int32, device="npu")
    cumulative_kv = torch.tensor([0, 5, 75], dtype=torch.int32, device="npu")
    q = torch.randn(5, 4, head_dim, dtype=torch.bfloat16, device="npu")
    k_cache = torch.randn(4, page_size, 2, head_dim, dtype=torch.bfloat16, device="npu")
    v_cache = torch.randn_like(k_cache)
    page_table = torch.tensor([[0, 0], [1, 2]], dtype=torch.int32, device="npu")

    output = mha_extend_with_kvcache(
        q,
        cumulative_q,
        cumulative_kv,
        k_cache,
        v_cache,
        page_table,
        cache_lengths,
        max(q_lengths),
        128,
        is_causal=True,
    )
    expected = []
    start = 0
    for batch, q_length in enumerate(q_lengths):
        length = int(cache_lengths[batch])
        expected.append(
            _attention(
                q[start : start + q_length],
                _paged_rows(k_cache, page_table, batch, length),
                _paged_rows(v_cache, page_table, batch, length),
                causal=True,
            )
        )
        start += q_length

    torch.testing.assert_close(output, torch.cat(expected), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("head_dim", [64, 256])
def test_mha_decode_with_paged_cache(head_dim) -> None:
    page_size = 128
    q = torch.randn(2, 4, head_dim, dtype=torch.bfloat16, device="npu")
    k_cache = torch.randn(4, page_size, 2, head_dim, dtype=torch.bfloat16, device="npu")
    v_cache = torch.randn_like(k_cache)
    page_table = torch.tensor([[0, 0], [1, 2]], dtype=torch.int32, device="npu")
    cache_lengths = torch.tensor([5, 70], dtype=torch.int32, device="npu")

    output = mha_decode_with_kvcache(
        q, k_cache, v_cache, page_table, cache_lengths, max_seqlen_k=128
    )
    expected = torch.stack(
        [
            _attention(
                q[batch : batch + 1],
                _paged_rows(k_cache, page_table, batch, int(length)),
                _paged_rows(v_cache, page_table, batch, int(length)),
                causal=False,
            )[0]
            for batch, length in enumerate(cache_lengths)
        ]
    )

    torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("head_dim", [64, 256])
def test_mha_decode_npugraph_updates_sequence_lengths(head_dim) -> None:
    page_size = 128
    q = torch.randn(2, 4, head_dim, dtype=torch.bfloat16, device="npu")
    k_cache = torch.randn(4, page_size, 2, head_dim, dtype=torch.bfloat16, device="npu")
    v_cache = torch.randn_like(k_cache)
    page_table = torch.tensor([[0, 0], [1, 2]], dtype=torch.int32, device="npu")
    capture_lengths = torch.ones(2, dtype=torch.int32, device="npu")

    for _ in range(3):
        mha_decode_with_kvcache(
            q, k_cache, v_cache, page_table, capture_lengths, max_seqlen_k=128
        )
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream(), auto_dispatch_capture=True):
        output = mha_decode_with_kvcache(
            q, k_cache, v_cache, page_table, capture_lengths, max_seqlen_k=128
        )
    torch.npu.synchronize()

    for lengths in ([5, 70], [12, 80]):
        graph.update(cpu_update_input=[{"actual_seq_lengths_kv": lengths}])
        graph.replay()
        torch.npu.synchronize()

        expected = torch.stack(
            [
                _attention(
                    q[batch : batch + 1],
                    _paged_rows(k_cache, page_table, batch, length),
                    _paged_rows(v_cache, page_table, batch, length),
                    causal=False,
                )[0]
                for batch, length in enumerate(lengths)
            ]
        )
        torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_qwen35_head256_ragged_prefill_matches_cpu_reference(dtype):
    from tokenspeed_kernel.ops.attention.mha import mha_prefill as shared_prefill

    torch.manual_seed(59)
    lengths = [1, 0, 3, 5]
    bounds_cpu = [0, 1, 1, 4, 9]
    bounds = torch.tensor(bounds_cpu, device="npu", dtype=torch.int32)
    # Q is a view into the query/gate projection, with a wider token stride.
    q = torch.randn(9, 8, 512, device="npu", dtype=dtype)[:, :, :256]
    k = torch.randn(9, 2, 256, device="npu", dtype=dtype)
    v = torch.randn_like(k)
    original = [tensor.clone() for tensor in (q, k, v)]
    actual = shared_prefill(
        q=q,
        k=k,
        v=v,
        cu_seqlens=bounds,
        cu_seqlens_cpu=bounds_cpu,
        max_seqlen=5,
        softmax_scale=0.07,
    )
    expected = []
    start = 0
    for length in lengths:
        if length:
            query = q[start : start + length].cpu().double().transpose(0, 1)
            key = (
                k[start : start + length]
                .cpu()
                .double()
                .repeat_interleave(4, dim=1)
                .transpose(0, 1)
            )
            value = (
                v[start : start + length]
                .cpu()
                .double()
                .repeat_interleave(4, dim=1)
                .transpose(0, 1)
            )
            scores = (query @ key.transpose(-1, -2)) * 0.07
            scores.masked_fill_(
                torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1),
                float("-inf"),
            )
            expected.append((scores.softmax(-1) @ value).transpose(0, 1))
        start += length
    _check_native_attention(actual, torch.cat(expected))
    for tensor, saved in zip((q, k, v), original):
        torch.testing.assert_close(tensor, saved, rtol=0, atol=0)


def _check_native_attention(actual, expected):
    # Native attention uses mixed-precision arithmetic. Allow two output ULPs
    # and a dtype-sized absolute floor near cancellation, plus a global L2 bound.
    bf16 = actual.dtype == torch.bfloat16
    rtol, atol = (2**-6, 2**-8) if bf16 else (2**-9, 2**-11)
    result = actual.cpu().double()
    expected = expected.cpu().to(actual.dtype).double()
    torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)
    relative_l2 = torch.linalg.vector_norm(
        result - expected
    ) / torch.linalg.vector_norm(expected)
    assert relative_l2 <= (2**-8 if bf16 else 2**-11)


@torch.inference_mode()
def test_bsh_conversion_is_exact_and_does_not_recompile():
    import sys
    from pathlib import Path

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2] / "tokenspeed-kernel/test")
    )
    from tokenspeed_kernel_npu.ops.mha import (
        _pack_bsh,
        _pack_bsh_kernel,
        _unpack_bsh,
        _unpack_bsh_kernel,
    )
    from utils import assert_no_triton_compile

    def run(requests, length):
        lengths = [length] * requests
        bounds = torch.arange(requests + 1, device="npu", dtype=torch.int32) * length
        source = torch.randn(
            requests * length, 8, 256 + requests, device="npu", dtype=torch.bfloat16
        )[:, :, :256]
        packed = _pack_bsh(source, bounds, length + 3)
        for request in range(requests):
            torch.testing.assert_close(
                packed[request, :length].reshape(length, 8, 256),
                source[request * length : (request + 1) * length],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                packed[request, length:],
                torch.zeros_like(packed[request, length:]),
                rtol=0,
                atol=0,
            )
        restored = _unpack_bsh(packed, bounds, source)
        torch.testing.assert_close(restored, source, rtol=0, atol=0)
        assert restored.is_contiguous()
        torch.npu.synchronize()

    run(1, 1)
    with assert_no_triton_compile(_pack_bsh_kernel, _unpack_bsh_kernel):
        for requests, length in ((2, 3), (3, 17), (5, 9)):
            run(requests, length)


@torch.inference_mode()
def test_qwen35_head256_cached_prefix_matches_cpu_reference():
    page_size = 128
    torch.manual_seed(61)
    q = torch.randn(4, 8, 256, device="npu", dtype=torch.bfloat16)
    k = torch.randn(6, page_size, 2, 256, device="npu", dtype=q.dtype)
    v = torch.randn_like(k)
    table = torch.tensor([[4, 0, 0], [3, 1, 5]], device="npu", dtype=torch.int32)
    lengths = [3, 2 * page_size + 5]
    output = mha_extend_with_kvcache(
        q,
        torch.tensor([0, 1, 4], device="npu", dtype=torch.int32),
        torch.tensor([0, 3, 3 + lengths[1]], device="npu", dtype=torch.int32),
        k,
        v,
        table,
        torch.tensor(lengths, device="npu", dtype=torch.int32),
        3,
        max(lengths),
        is_causal=True,
        softmax_scale=0.07,
    )
    expected = []
    for request, (start, end) in enumerate(((0, 1), (1, 4))):
        query = q[start:end].cpu().double().transpose(0, 1)
        key = (
            _paged_rows(k, table, request, lengths[request])
            .cpu()
            .double()
            .repeat_interleave(4, dim=1)
            .transpose(0, 1)
        )
        value = (
            _paged_rows(v, table, request, lengths[request])
            .cpu()
            .double()
            .repeat_interleave(4, dim=1)
            .transpose(0, 1)
        )
        scores = query @ key.transpose(-1, -2) * 0.07
        qpositions = torch.arange(end - start) + lengths[request] - (end - start)
        mask = torch.arange(lengths[request])[None, :] > qpositions[:, None]
        scores.masked_fill_(mask, float("-inf"))
        expected.append((scores.softmax(-1) @ value).transpose(0, 1))
    _check_native_attention(output, torch.cat(expected))


@torch.inference_mode()
def test_empty_prefill():
    q = torch.empty(0, 8, 256, device="npu", dtype=torch.bfloat16)
    k = torch.empty(0, 2, 256, device="npu", dtype=q.dtype)
    output = mha_prefill(
        q, k, k, torch.tensor([0, 0], device="npu", dtype=torch.int32), [0, 0], 0
    )
    assert output.shape == q.shape and output.dtype == q.dtype


@torch.inference_mode()
def test_cached_gqa_rejects_64_token_pages():
    q = torch.randn(3, 8, 256, device="npu", dtype=torch.bfloat16)
    cache = torch.randn(1, 64, 2, 256, device="npu", dtype=q.dtype)
    original = cache.clone()
    with pytest.raises(NotImplementedError, match="page size divisible by 128"):
        mha_extend_with_kvcache(
            q,
            torch.tensor([0, 3], device="npu", dtype=torch.int32),
            torch.tensor([0, 5], device="npu", dtype=torch.int32),
            cache,
            cache,
            torch.zeros(1, 1, device="npu", dtype=torch.int32),
            torch.tensor([5], device="npu", dtype=torch.int32),
            3,
            5,
            is_causal=True,
        )
    torch.testing.assert_close(cache, original, rtol=0, atol=0)


@torch.inference_mode()
def test_prefill_rejects_capacity_smaller_than_live_sequence():
    q = torch.randn(3, 8, 256, device="npu", dtype=torch.bfloat16)
    k = torch.randn(3, 2, 256, device="npu", dtype=q.dtype)
    with pytest.raises(ValueError, match="max_seqlen"):
        mha_prefill(
            q, k, k, torch.tensor([0, 3], device="npu", dtype=torch.int32), [0, 3], 2
        )
