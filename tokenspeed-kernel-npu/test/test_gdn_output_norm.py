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

from tokenspeed.runtime.layers.attention.linear.layernorm_gated import (
    RMSNorm,
    rmsnorm_fn,
)

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires Ascend")


def _reference(x, weight, z, *, eps, group_size, norm_before_gate, sigmoid_gate):
    # CPU FP64 reference; each head has its own RMS statistic.
    values = x.cpu().double()
    gate = None
    if z is not None:
        gate = z.cpu().double().sigmoid() if sigmoid_gate else F.silu(z.cpu().double())
        if not norm_before_gate:
            values = values * gate
    width = values.shape[-1]
    size = width if group_size is None else group_size
    grouped = values.reshape(*values.shape[:-1], width // size, size)
    normalized = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + eps)
    weights = weight.cpu().double()
    if weights.numel() == width:
        weights = weights.reshape(width // size, size)
    result = (normalized * weights).reshape_as(values)
    if gate is not None and norm_before_gate:
        result = result * gate
    return result.to(x.dtype)


def _check(actual, expected):
    tolerance = {torch.bfloat16: 2**-7, torch.float16: 2**-10, torch.float32: 3e-5}[
        actual.dtype
    ]
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=2e-6)


@pytest.mark.parametrize("tokens", [0, 1, 3, 17])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@torch.inference_mode()
def test_qwen_gated_norm_with_shared_head_weight_and_strided_gate(tokens, dtype):
    torch.manual_seed(37)
    x = torch.randn(tokens, 2048, device="npu", dtype=dtype)
    projection = torch.randn(tokens, 8200, device="npu", dtype=dtype)
    z = projection[:, 6144:8192]
    norm = RMSNorm(
        128, eps=1e-6, group_size=128, norm_before_gate=True, device="npu", dtype=dtype
    )
    norm.weight.copy_(torch.randn_like(norm.weight))
    originals = [tensor.clone() for tensor in (x, projection, norm.weight)]
    expected = _reference(
        x,
        norm.weight,
        z,
        eps=1e-6,
        group_size=128,
        norm_before_gate=True,
        sigmoid_gate=False,
    )
    actual = norm(x, z)
    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    assert actual.device == x.device
    _check(actual, expected)
    for tensor, original in zip((x, projection, norm.weight), originals):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)


@pytest.mark.parametrize("norm_before_gate", [False, True])
@pytest.mark.parametrize("sigmoid_gate", [False, True])
@pytest.mark.parametrize("shared_weight", [False, True])
@torch.inference_mode()
def test_gate_order_and_group_weights(norm_before_gate, sigmoid_gate, shared_weight):
    # Nonuniform groups and negative/extreme logits distinguish both gate orders.
    x = torch.tensor([[1.0, 2.0, -3.0, 4.0, 0.01, 0.02, 0.04, -0.08]], device="npu")
    z = torch.tensor([[-20.0, -2.0, 0.0, 2.0, 20.0, -3.0, 1.0, -1.0]], device="npu")
    weight = torch.arange(
        1, 5 if shared_weight else 9, device="npu", dtype=torch.float32
    )
    options = dict(
        eps=0.01,
        group_size=4,
        norm_before_gate=norm_before_gate,
        sigmoid_gate=sigmoid_gate,
    )
    actual = rmsnorm_fn(x, weight, z=z, weights_independent=False, **options)
    _check(actual, _reference(x, weight, z, **options))


@pytest.mark.parametrize("group_size", [None, 7])
@pytest.mark.parametrize("has_gate", [False, True])
@torch.inference_mode()
def test_arbitrary_shapes_and_strides(group_size, has_gate):
    x = torch.randn(2, 3, 42, device="npu", dtype=torch.float16)[..., ::2]
    z = torch.randn(2, 3, 42, device="npu")[..., ::2] if has_gate else None
    weight = torch.randn(42, device="npu")[::2]
    options = dict(
        eps=0.03, group_size=group_size, norm_before_gate=False, sigmoid_gate=False
    )
    actual = rmsnorm_fn(x, weight, z=z, weights_independent=True, **options)
    _check(actual, _reference(x, weight, z, **options))


@torch.inference_mode()
def test_fp8_request_rejected_without_mutation():
    x = torch.randn(1, 128, device="npu", dtype=torch.bfloat16)
    weight = torch.ones(128, device="npu", dtype=x.dtype)
    original = x.clone()
    with pytest.raises(NotImplementedError, match="FP8"):
        rmsnorm_fn(
            x,
            weight,
            z=x,
            group_size=128,
            norm_before_gate=True,
            sigmoid_gate=False,
            weights_independent=True,
            fp8_scale=torch.ones(1, device="npu"),
        )
    torch.testing.assert_close(x, original, rtol=0, atol=0)


@pytest.mark.parametrize("norm_before_gate", [False, True])
@torch.inference_mode()
def test_batch_sizes_and_projection_strides_do_not_recompile(norm_before_gate):
    import sys
    from pathlib import Path

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2] / "tokenspeed-kernel/test")
    )
    from tokenspeed_kernel_npu.ops.layernorm import _gated_rmsnorm_kernel
    from utils import assert_no_triton_compile

    weight = torch.randn(128, device="npu", dtype=torch.bfloat16)

    def run(rows):
        x = torch.randn(rows, 2048 + rows, device="npu", dtype=weight.dtype)[:, :2048]
        z = torch.randn(rows, 8200 + rows, device="npu", dtype=weight.dtype)[
            :, 6144:8192
        ]
        options = dict(
            eps=1e-6,
            group_size=128,
            norm_before_gate=norm_before_gate,
            sigmoid_gate=False,
        )
        result = rmsnorm_fn(x, weight, z=z, weights_independent=True, **options)
        _check(result, _reference(x, weight, z, **options))
        torch.npu.synchronize()

    run(1)
    with assert_no_triton_compile(_gated_rmsnorm_kernel):
        for rows in (0, 2, 4, 17, 65):
            run(rows)


@torch.inference_mode()
def test_group_statistics_are_independent():
    x = torch.ones(1, 256, device="npu")
    x[:, 128:] = 1000
    weight = torch.ones(128, device="npu")
    actual = rmsnorm_fn(
        x,
        weight,
        z=None,
        eps=1e-6,
        group_size=128,
        norm_before_gate=True,
        sigmoid_gate=False,
        weights_independent=True,
    )
    torch.testing.assert_close(actual, torch.ones_like(actual), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("group_size,weight_width", [(0, 128), (127, 128), (128, 127)])
@torch.inference_mode()
def test_invalid_groups_or_weights_rejected(group_size, weight_width):
    x = torch.randn(1, 256, device="npu")
    with pytest.raises(ValueError, match="group_size|weight"):
        rmsnorm_fn(
            x,
            torch.ones(weight_width, device="npu"),
            z=None,
            group_size=group_size,
            norm_before_gate=True,
            sigmoid_gate=False,
            weights_independent=True,
        )
