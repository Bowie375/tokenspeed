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

from tokenspeed_kernel.ops.activation import sigmoid_mul

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires Ascend")


def _reference(x, gate, bias):
    values = gate.cpu().double()
    if bias is not None:
        values = (
            values.reshape(x.shape[0], bias.numel(), x.shape[1] // bias.numel())
            + bias.cpu().double()[None, :, None]
        )
    return (x.cpu().double() * values.sigmoid().reshape_as(x)).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("tokens", [0, 1, 3, 17])
@pytest.mark.parametrize("with_bias", [False, True])
@torch.inference_mode()
def test_full_attention_gate_mutates_output_preserves_projection(
    dtype, tokens, with_bias
):
    torch.manual_seed(23)
    # Q and gate are alternating head-sized slices of one wide QKV projection.
    projection = torch.randn(tokens, 8 * 2 * 256 + 512, device="npu", dtype=dtype)
    heads = projection[:, :4096].view(tokens, 8, 2, 256)
    query, gate = heads.unbind(dim=2)
    x = torch.randn(tokens, 2048, device="npu", dtype=dtype)
    original, saved_projection = x.clone(), projection.clone()
    bias = torch.linspace(-2, 2, 8, device="npu", dtype=dtype) if with_bias else None
    saved_bias = None if bias is None else bias.clone()
    expected = _reference(x, gate, bias)
    actual = sigmoid_mul(x, gate, head_bias=bias)
    assert actual is x
    tolerance = {torch.bfloat16: 2**-7, torch.float16: 2**-10, torch.float32: 3e-5}[
        dtype
    ]
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=1e-7)
    torch.testing.assert_close(projection, saved_projection, rtol=0, atol=0)
    if bias is not None:
        torch.testing.assert_close(bias, saved_bias, rtol=0, atol=0)
    if tokens:
        assert not torch.equal(x, original)


@pytest.mark.parametrize("with_bias", [False, True])
@torch.inference_mode()
def test_flat_gate_extreme_values(with_bias):
    gate = torch.tensor([[-80.0, -20.0, -2.0, 0.0, 2.0, 20.0, 80.0, 0.0]], device="npu")
    x = torch.tensor([[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, 8.0]], device="npu")
    bias = torch.tensor([-0.5, 0.5], device="npu") if with_bias else None
    expected = _reference(x, gate, bias)
    actual = sigmoid_mul(x, gate, head_bias=bias)
    torch.testing.assert_close(actual.cpu(), expected, rtol=3e-5, atol=1e-9)


@torch.inference_mode()
def test_variable_token_counts_and_projection_strides_do_not_recompile():
    import sys
    from pathlib import Path

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2] / "tokenspeed-kernel/test")
    )
    from tokenspeed_kernel_npu.ops.activation import _sigmoid_mul_kernel
    from utils import assert_no_triton_compile

    def run(tokens):
        projection = torch.randn(
            tokens, 4096 + tokens, device="npu", dtype=torch.bfloat16
        )
        gate = projection[:, :4096].view(tokens, 8, 2, 256)[:, :, 1, :]
        x = torch.randn(tokens, 2048, device="npu", dtype=torch.bfloat16)
        expected = _reference(x, gate, None)
        actual = sigmoid_mul(x, gate)
        torch.testing.assert_close(actual.cpu(), expected, rtol=2**-7, atol=1e-7)
        torch.npu.synchronize()

    run(1)
    with assert_no_triton_compile(_sigmoid_mul_kernel):
        for tokens in (2, 7, 17):
            run(tokens)


@pytest.mark.parametrize("with_bias", [False, True])
@torch.inference_mode()
def test_flat_gate_row_stride(with_bias):
    x = torch.randn(3, 14, device="npu")
    gate = torch.randn(3, 23, device="npu")[:, 2:16]
    bias = torch.randn(2, device="npu") if with_bias else None
    expected = _reference(x, gate, bias)
    sigmoid_mul(x, gate, head_bias=bias)
    torch.testing.assert_close(x.cpu(), expected, rtol=3e-5, atol=1e-7)


@pytest.mark.parametrize(
    "invalid",
    [
        "output_stride",
        "gate_shape",
        "gate_columns",
        "gate_dtype",
        "bias_heads",
        "bias_stride",
    ],
)
@torch.inference_mode()
def test_invalid_operands_reject_before_mutation(invalid):
    x = torch.randn(2, 16, device="npu")
    gate = torch.randn_like(x)
    bias = None
    if invalid == "output_stride":
        x = torch.randn(2, 32, device="npu")[:, ::2]
    elif invalid == "gate_shape":
        gate = gate[:, :8]
    elif invalid == "gate_columns":
        gate = torch.randn(2, 32, device="npu")[:, ::2]
    elif invalid == "gate_dtype":
        gate = gate.half()
    elif invalid == "bias_heads":
        bias = torch.zeros(3, device="npu")
    elif invalid == "bias_stride":
        bias = torch.zeros(4, device="npu")[::2]
    original = x.clone()
    with pytest.raises(ValueError):
        sigmoid_mul(x, gate, head_bias=bias)
    torch.testing.assert_close(x, original, rtol=0, atol=0)
