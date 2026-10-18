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

from tokenspeed_kernel.ops.attention.gdn import (
    fused_qkv_split_gdn_prefill,
    pack_gdn_projection,
)

from tokenspeed.runtime.layers.attention.linear.gdn import fused_gdn_gating

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires Ascend")


@pytest.mark.parametrize("beta,threshold", [(1.0, 20.0), (2.0, 5.0)])
@torch.inference_mode()
def test_decay_gate_is_stable_and_preserves_options(beta, threshold):
    a = torch.tensor([[-40.0, -20.0, 0.0, 40.0]], device="npu")
    A_log = torch.tensor([40.0, 20.0, 0.0, -3.0], device="npu")
    bias = torch.zeros_like(A_log)
    expected = -A_log.cpu().double().exp() * F.softplus(
        a.cpu().double(), beta=beta, threshold=threshold
    )
    actual = fused_gdn_gating(A_log, a, bias, beta=beta, threshold=threshold)
    torch.testing.assert_close(actual.cpu(), expected.float(), rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("rows", [0, 1, 17])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@torch.inference_mode()
def test_pack_projection_dense_independent_copies(rows, dtype):
    source = torch.randn(rows, 55, device="npu", dtype=dtype)
    qkv, a, b = source[:, :37], source[:, 41:45], source[:, 48:52]
    saved = tuple(t.clone() for t in (qkv, a, b))
    outputs = pack_gdn_projection(qkv, a, b)
    for original, packed, expected in zip((qkv, a, b), outputs, saved):
        torch.testing.assert_close(packed, original, rtol=0, atol=0)
        packed.zero_()
        if rows:
            torch.testing.assert_close(original, expected, rtol=0, atol=0)
            assert packed.data_ptr() != original.data_ptr()


@pytest.mark.parametrize("rows", [0, 1, 17])
@pytest.mark.parametrize("normalize", [False, True])
@torch.inference_mode()
def test_split_postconv_layout_and_normalization(rows, normalize):
    qwidth, kwidth, vwidth = 2 * 7, 2 * 7, 4 * 11
    x = torch.randn(
        rows, qwidth + kwidth + vwidth + 9, device="npu", dtype=torch.bfloat16
    )[:, : qwidth + kwidth + vwidth]
    q, k, v = fused_qkv_split_gdn_prefill(x, 2, 2, 4, 7, 7, 11, fuse_l2norm=normalize)
    for actual, part, heads, width in zip(
        (q, k, v), x.split((qwidth, kwidth, vwidth), dim=-1), (2, 2, 4), (7, 7, 11)
    ):
        expected = part.cpu().float().reshape(1, rows, heads, width)
        if normalize and actual is not v:
            expected *= torch.rsqrt(expected.square().sum(-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(
            actual.cpu(), expected.to(x.dtype), rtol=2**-7, atol=1e-6
        )
        assert actual.is_contiguous()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@torch.inference_mode()
def test_gate_strided_parameters(dtype):
    a = torch.randn(9, 12, device="npu", dtype=dtype)[:, ::2]
    rates = torch.randn(12, device="npu")[::2]
    bias = torch.randn(12, device="npu")[::2]
    expected = -rates.cpu().double().exp() * F.softplus(
        a.cpu().double() + bias.cpu().double(), beta=1.7, threshold=8
    )
    out = fused_gdn_gating(rates, a, bias, beta=1.7, threshold=8)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out.cpu(), expected.float(), rtol=3e-5, atol=1e-7)


@pytest.mark.parametrize("normalize", [False, True])
@torch.inference_mode()
def test_split_qwen_geometry_and_device_padding(normalize):
    x = torch.randn(5, 6144, device="npu", dtype=torch.bfloat16)
    x[3:] = float("nan")
    valid = torch.tensor(3, device="npu", dtype=torch.int32)
    for count in (3, 1, 0):
        valid.fill_(count)
        outputs = fused_qkv_split_gdn_prefill(
            x, 16, 16, 16, 128, 128, 128, fuse_l2norm=normalize, valid_tokens=valid
        )
        for index, actual in enumerate(outputs):
            expected = (
                x[:, index * 2048 : (index + 1) * 2048]
                .cpu()
                .float()
                .view(1, 5, 16, 128)
            )
            expected[:, count:] = 0
            if normalize and index < 2:
                expected *= torch.rsqrt(expected.square().sum(-1, keepdim=True) + 1e-6)
            torch.testing.assert_close(
                actual.cpu(), expected.to(x.dtype), rtol=2**-7, atol=1e-6
            )
            assert actual.is_contiguous()
            assert actual.data_ptr() != x.data_ptr()


@torch.inference_mode()
def test_split_rejects_replay():
    x = torch.randn(1, 3, device="npu")
    original = x.clone()
    with pytest.raises(NotImplementedError, match="replay"):
        fused_qkv_split_gdn_prefill(
            x, 1, 1, 1, 1, 1, 1, fuse_l2norm=False, replay=(x, x, x)
        )
    torch.testing.assert_close(x, original, rtol=0, atol=0)


@torch.inference_mode()
def test_preparation_and_recurrence_prefill_matches_decode():
    from tokenspeed_kernel.ops.attention.gdn import gdn_chunk_prefill, gdn_decode_step
    from tokenspeed_kernel.ops.attention.gdn.triton import (
        CAUSAL_CONV1D_BLOCK_M,
        build_causal_conv1d_prefill_metadata,
    )

    from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
        causal_conv1d_fn,
        causal_conv1d_update,
    )

    torch.manual_seed(101)
    tokens, channels = 9, 72  # Q/K: 2x7, V: 4x11.
    projected = torch.randn(tokens, channels + 8, device="npu") * 0.15
    raw, a, b = pack_gdn_projection(
        projected[:, :channels],
        projected[:, channels : channels + 4],
        projected[:, channels + 4 :],
    )
    weights = torch.randn(channels, 4, device="npu") * 0.2
    conv_state = torch.randn(1, channels, 3, device="npu") * 0.1
    decode_conv = conv_state.clone()
    recurrent = torch.randn(1, 4, 11, 7, device="npu") * 0.1
    decode_recurrent = recurrent.clone()
    rates = torch.randn(4, device="npu") * 0.1
    bias = torch.randn(4, device="npu") * 0.1
    bounds = torch.tensor([0, tokens], device="npu", dtype=torch.int32)
    metadata = build_causal_conv1d_prefill_metadata(
        bounds, torch.tensor([tokens], dtype=torch.int32), CAUSAL_CONV1D_BLOCK_M
    )
    conv = causal_conv1d_fn(
        raw.T,
        weights,
        None,
        conv_state,
        bounds,
        has_initial_state=torch.ones(1, device="npu", dtype=torch.bool),
        activation="silu",
        prefill_metadata=metadata,
    )
    q, k, v = fused_qkv_split_gdn_prefill(conv.T, 2, 2, 4, 7, 7, 11, fuse_l2norm=False)
    prefill = gdn_chunk_prefill(
        q,
        k,
        v,
        fused_gdn_gating(rates, a, bias).unsqueeze(0),
        b.float().sigmoid().unsqueeze(0),
        scale=None,
        initial_state=recurrent,
        cu_seqlens=bounds,
        qk_l2norm=True,
        output_final_state=True,
        output_h=False,
    )
    outputs = []
    for token in range(tokens):
        step = raw[token : token + 1].clone()
        causal_conv1d_update(
            step, decode_conv, weights, activation="silu", parent_indices=None
        )
        q, k, v = fused_qkv_split_gdn_prefill(
            step, 2, 2, 4, 7, 7, 11, fuse_l2norm=False
        )
        outputs.append(
            gdn_decode_step(
                q,
                k,
                v,
                A_log=rates,
                a=a[token : token + 1].unsqueeze(0),
                dt_bias=bias,
                b=b[token : token + 1].unsqueeze(0),
                initial_state=decode_recurrent,
                initial_state_indices=torch.zeros(1, device="npu", dtype=torch.int32),
                scale=None,
                output_state_indices=None,
                use_qk_l2norm=True,
            )
        )
    torch.testing.assert_close(
        torch.cat(outputs, dim=1), prefill.out, rtol=3e-5, atol=3e-6
    )
    torch.testing.assert_close(
        decode_recurrent, prefill.final_state, rtol=3e-5, atol=3e-6
    )
    torch.testing.assert_close(decode_conv, conv_state, rtol=0, atol=0)
