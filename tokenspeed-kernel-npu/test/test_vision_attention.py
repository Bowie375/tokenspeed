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

from tokenspeed.runtime.layers.attention.mm_encoder_attention import _resolve_backend

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires Ascend")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("lengths", [[1], [3, 5], [2, 0, 4], [], [0, 0]])
@torch.inference_mode()
def test_noncausal_encoder_attention_matches_cpu(dtype, lengths):
    tokens = sum(lengths)
    storage = torch.randn(tokens, 12, 64 * 6, device="npu", dtype=dtype)
    q, k, v = storage[..., :64], storage[..., 128:192], storage[..., 256:320]
    before = storage.clone()
    bounds_cpu = [0]
    for length in lengths:
        bounds_cpu.append(bounds_cpu[-1] + length)
    bounds = torch.tensor(bounds_cpu, device="npu", dtype=torch.int32)
    backend = _resolve_backend("ascend")
    actual = backend(
        q,
        k,
        v,
        cu_seqlens=bounds,
        cu_seqlens_cpu=bounds_cpu,
        bsz=1,
        seq_len=tokens,
        softmax_scale=0.13,
        max_seqlen=max(lengths, default=0),
        sequence_lengths=None,
        workspace_buffer=None,
    )
    expected = torch.empty(tokens, 12, 64, dtype=torch.float64)
    for start, end in zip(bounds_cpu, bounds_cpu[1:]):
        if start == end:
            continue
        query, key, value = [
            tensor[start:end].cpu().double().transpose(0, 1) for tensor in (q, k, v)
        ]
        expected[start:end] = (
            (query @ key.transpose(-1, -2) * 0.13).softmax(-1) @ value
        ).transpose(0, 1)
    assert actual.shape == q.shape and actual.dtype == dtype
    tolerance = (2**-6, 2**-8) if dtype == torch.bfloat16 else (2**-9, 2**-11)
    torch.testing.assert_close(
        actual.cpu(), expected.to(dtype), rtol=tolerance[0], atol=tolerance[1]
    )
    torch.testing.assert_close(storage, before, rtol=0, atol=0)


@torch.inference_mode()
def test_encoder_visibility_is_bidirectional_and_images_are_isolated():
    q = torch.zeros(5, 2, 64, device="npu", dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.empty_like(q)
    for row, value in enumerate((1, 3, 5, 10, 14)):
        v[row].fill_(value)
    options = dict(
        cu_seqlens=torch.tensor([0, 3, 5], device="npu", dtype=torch.int32),
        cu_seqlens_cpu=[0, 3, 5],
        bsz=1,
        seq_len=5,
        softmax_scale=0.5,
        max_seqlen=3,
        sequence_lengths=None,
        workspace_buffer=None,
    )
    backend = _resolve_backend(None)
    result = backend(q, k, v, **options)
    torch.testing.assert_close(
        result[:3], torch.full_like(result[:3], 3), rtol=0, atol=0
    )
    torch.testing.assert_close(
        result[3:], torch.full_like(result[3:], 12), rtol=0, atol=0
    )
    v[3:].fill_(100)
    changed = backend(q, k, v, **options)
    torch.testing.assert_close(changed[:3], result[:3], rtol=0, atol=0)


@torch.inference_mode()
def test_uniform_batched_rank_four_and_eager_metadata():
    q = torch.zeros(2, 3, 2, 64, device="npu", dtype=torch.float16)
    v = torch.arange(6, device="npu", dtype=q.dtype).reshape(2, 3, 1, 1).expand_as(q)
    result = _resolve_backend("ascend")(q, q, v, cu_seqlens=None, bsz=2, seq_len=3)
    expected = torch.empty_like(q)
    expected[0].fill_(1)
    expected[1].fill_(4)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("argument", ["sequence_lengths", "workspace_buffer"])
@torch.inference_mode()
def test_unsupported_external_metadata_rejected(argument):
    q = torch.zeros(1, 2, 64, device="npu", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="cuDNN"):
        _resolve_backend("ascend")(
            q,
            q,
            q,
            cu_seqlens=None,
            bsz=1,
            seq_len=1,
            **{argument: torch.ones(1, device="npu")}
        )
    torch.testing.assert_close(q, torch.zeros_like(q), rtol=0, atol=0)


@torch.inference_mode()
def test_encoder_module_preserves_explicit_softmax_scale():
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.layers.attention.mm_encoder_attention import (
        MultimodalEncoderAttention,
    )
    from tokenspeed.runtime.model_loader.utils import set_default_torch_dtype

    with torch.device("npu"), set_default_torch_dtype(torch.bfloat16):
        module = MultimodalEncoderAttention(
            embed_dim=128,
            num_heads=2,
            mapping=Mapping(rank=0, world_size=1),
            mm_attention_backend="ascend",
        ).eval()
        identity = torch.eye(128, dtype=torch.bfloat16)
        module.qkv_proj.weight.copy_(identity.repeat(3, 1))
        module.qkv_proj.bias.zero_()
        module.proj.weight.copy_(identity)
        module.proj.bias.zero_()
    generator = torch.Generator().manual_seed(42)
    x = (torch.randn(1, 3, 128, generator=generator) * 0.5).bfloat16()
    scale = 0.75
    actual = module(x.npu(), softmax_scale=scale)
    rows = x.double().reshape(3, 2, 64).transpose(0, 1)
    expected = (
        ((rows @ rows.transpose(-1, -2) * scale).softmax(-1) @ rows)
        .transpose(0, 1)
        .reshape_as(x)
    )
    torch.testing.assert_close(
        actual.cpu(), expected.bfloat16(), rtol=2**-6, atol=2**-8
    )
