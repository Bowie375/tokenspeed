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

import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("torch_npu")

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tokenspeed-kernel/test"))
from tokenspeed_kernel.ops.layernorm import add_rmsnorm
from tokenspeed_kernel_npu.ops.layernorm import _gemma_rmsnorm_kernel
from utils import assert_no_triton_compile

from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="requires an Ascend NPU"
)


def _reference(total, weight, eps):
    total = total.cpu().double()
    scale = 1.0 + weight.cpu().double()
    return total * torch.rsqrt(total.square().mean(-1, keepdim=True) + eps) * scale


def _check(actual, expected):
    # At most one activation-dtype rounding unit, with a tighter FP32 bound.
    rtol = {torch.bfloat16: 2**-7, torch.float16: 2**-10, torch.float32: 2e-6}[
        actual.dtype
    ]
    torch.testing.assert_close(
        actual.cpu(), expected.to(actual.dtype), rtol=rtol, atol=1e-6
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("rows,width", [(0, 1024), (1, 37), (7, 1024), (129, 1024)])
@pytest.mark.parametrize("with_residual", [False, True])
@torch.inference_mode()
def test_gemma_norm_forward(dtype, rows, width, with_residual):
    torch.manual_seed(42)
    norm = GemmaRMSNorm(width, eps=1e-6).to(device="npu", dtype=dtype)
    weight = (torch.randn(width) * 0.2).to(device="npu", dtype=dtype)
    norm.weight.weight_loader(norm.weight, weight)
    x = torch.randn(rows, width * 2, dtype=dtype, device="npu")[:, :width]
    original = x.clone()
    residual = torch.randn_like(x) if with_residual else None
    total = x.float() if residual is None else x.float() + residual.float()
    expected = _reference(total, weight, 1e-6)

    result = norm(x, residual)

    if with_residual:
        actual, updated = result
        assert actual is x
        assert updated is residual
        torch.testing.assert_close(updated, total.to(dtype), rtol=0, atol=0)
    else:
        actual = result
        torch.testing.assert_close(x, original, rtol=0, atol=0)
    _check(actual, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("with_x2", [False, True])
@pytest.mark.parametrize("alias_output", [False, True])
@torch.inference_mode()
def test_shared_add_norm_keeps_unrounded_sum_and_aliases_output(
    dtype, gemma, with_x2, alias_output
):
    torch.manual_seed(43)
    x = torch.randn(7, 2048, dtype=dtype, device="npu")[:, :1024]
    residual = torch.randn(7, 2048, dtype=dtype, device="npu")[:, :1024]
    weight = (torch.randn(1024) * 0.2).to(device="npu", dtype=dtype)
    x2 = torch.randn_like(x) if with_x2 else None
    original = x.clone()
    out = x if alias_output else torch.empty_like(x)
    addend = x if x2 is None else x + x2
    total = addend.float() + residual.float()
    # Avoid an activation-dtype weight-offset roundtrip in the plain norm reference.
    expected = _reference(total, weight, 1e-6)
    if not gemma:
        values = total.cpu().double()
        expected = (
            values
            * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
            * weight.cpu().double()
        )

    add_rmsnorm(
        x,
        residual,
        weight,
        1e-6,
        x2=x2,
        out=out,
        out_fp8=None,
        fp8_scale=None,
        out_fp4=None,
        fp4_scale=None,
        gemma=gemma,
    )

    torch.testing.assert_close(residual, total.to(dtype), rtol=0, atol=0)
    _check(out, expected)
    if not alias_output:
        torch.testing.assert_close(x, original, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["fp8", "fp4"])
@torch.inference_mode()
def test_gemma_layer_fused_norm_without_quantization(method):
    torch.manual_seed(44)
    norm = GemmaRMSNorm(1024, eps=1e-6).to(device="npu", dtype=torch.bfloat16)
    norm.weight.weight_loader(norm.weight, torch.randn_like(norm.weight) * 0.2)
    x = torch.randn(3, 1024, device="npu", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    total = x.float() + residual.float()
    expected = _reference(total, norm.weight, 1e-6)
    if method == "fp8":
        actual, quantized, updated = norm.add_norm_with_fp8(x, residual, None)
    else:
        actual, quantized, updated = norm.add_norm_with_fp4(x, residual, None)
    assert actual is x
    assert updated is residual
    assert quantized is None
    torch.testing.assert_close(residual, total.to(x.dtype), rtol=0, atol=0)
    _check(actual, expected)


@torch.inference_mode()
def test_fused_gemma_norm_uses_sum_before_bfloat16_rounding():
    x = torch.tensor([[1.0, 1.0]], device="npu", dtype=torch.bfloat16)
    residual = torch.tensor([[1 / 256, 3 / 256]], device="npu", dtype=x.dtype)
    weight = torch.zeros(2, device="npu", dtype=x.dtype)
    add_rmsnorm(
        x,
        residual,
        weight,
        1e-6,
        x2=None,
        out=x,
        out_fp8=None,
        fp8_scale=None,
        out_fp4=None,
        fp4_scale=None,
        gemma=True,
    )
    # Normalizing the stored BF16 residual instead would give [0.9921875, 1.0078125].
    torch.testing.assert_close(
        x.cpu(), torch.tensor([[255 / 256, 1.0]], dtype=x.dtype), rtol=0, atol=0
    )
    torch.testing.assert_close(
        residual.cpu(), torch.tensor([[1.0, 65 / 64]], dtype=x.dtype), rtol=0, atol=0
    )


@pytest.mark.parametrize("method", ["fp8", "fp4"])
@torch.inference_mode()
def test_gemma_layer_rejects_quantized_outputs_without_mutation(method):
    norm = GemmaRMSNorm(37).to(device="npu", dtype=torch.bfloat16)
    x = torch.ones(1, 37, device="npu", dtype=torch.bfloat16)
    residual = torch.ones_like(x)
    scale = torch.ones(1, device="npu")
    with pytest.raises(NotImplementedError, match="FP8 or NVFP4"):
        if method == "fp8":
            norm.add_norm_with_fp8(x, residual, scale)
        else:
            norm.add_norm_with_fp4(x, residual, scale)
    torch.testing.assert_close(x, torch.ones_like(x), rtol=0, atol=0)
    torch.testing.assert_close(residual, torch.ones_like(residual), rtol=0, atol=0)


@pytest.mark.parametrize("with_residual", [False, True])
@torch.inference_mode()
def test_gemma_norm_reuses_compilation_across_batch_sizes(with_residual):
    norm = GemmaRMSNorm(1024).to(device="npu", dtype=torch.bfloat16)

    def run(rows):
        x = torch.ones(rows, 1024, device="npu", dtype=torch.bfloat16)
        if with_residual:
            actual, residual = norm(x, torch.ones_like(x))
            torch.testing.assert_close(
                residual, torch.full_like(residual, 2), rtol=0, atol=0
            )
        else:
            actual = norm(x)
        torch.testing.assert_close(actual, torch.ones_like(actual), rtol=0, atol=0)

    run(1)
    with assert_no_triton_compile(_gemma_rmsnorm_kernel):
        for rows in (3, 16, 129):
            run(rows)
