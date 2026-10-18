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

from tokenspeed_kernel.ops.attention.gdn import gdn_chunk_prefill, gdn_decode_step

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="GDN tests require an Ascend NPU"
)


def _strided(tensor):
    storage = torch.empty(
        *tensor.shape[:-1], tensor.shape[-1] * 2, device="npu", dtype=tensor.dtype
    )
    view = storage[..., ::2]
    view.copy_(tensor)
    return view


def _inputs(lengths, dtype, *, heads, value_heads, key_width, value_width):
    torch.manual_seed(19)
    tokens = sum(lengths)
    q = _strided(torch.randn(1, tokens, heads, key_width).to(dtype) * 0.25)
    k = _strided(torch.randn(1, tokens, heads, key_width).to(dtype) * 0.25)
    v = _strided(torch.randn(1, tokens, value_heads, value_width).to(dtype) * 0.25)
    g = _strided(-torch.rand(1, tokens, value_heads))
    beta = _strided(torch.rand(1, tokens, value_heads).to(dtype))
    state = _strided(
        torch.randn(len(lengths), value_heads, value_width, key_width) * 0.1
    )
    return q, k, v, g, beta, state


def _reference(q, k, v, g, beta, state, lengths, *, normalize, scale):
    # An independent CPU FP64, value-major recurrence; no production helpers.
    q, k, v, g, beta, state = [x.cpu().double() for x in (q, k, v, g, beta, state)]
    q = q.repeat_interleave(v.shape[2] // q.shape[2], dim=2)
    k = k.repeat_interleave(v.shape[2] // k.shape[2], dim=2)
    if normalize:
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    out = torch.empty_like(v)
    offset = 0
    final = state.clone()
    for request, length in enumerate(lengths):
        memory = state[request].clone()
        for position in range(offset, offset + length):
            key = k[0, position]
            memory = memory * g[0, position].exp()[:, None, None]
            prediction = torch.einsum("hvk,hk->hv", memory, key)
            delta = beta[0, position, :, None] * (v[0, position] - prediction)
            memory = memory + delta[:, :, None] * key[:, None, :]
            out[0, position] = torch.einsum(
                "hvk,hk->hv", memory, q[0, position] * scale
            )
        final[request] = memory
        offset += length
    return out, final


def _check_output(actual, expected):
    rtol = {torch.bfloat16: 2**-7, torch.float16: 2**-10, torch.float32: 3e-5}[
        actual.dtype
    ]
    torch.testing.assert_close(
        actual.cpu(), expected.to(actual.dtype), atol=3e-6, rtol=rtol
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("zero_state", [False, True])
@torch.inference_mode()
def test_prefill_varlen_matches_reference(dtype, normalize, zero_state):
    lengths = [0, 1, 3, 0, 5]
    q, k, v, g, beta, state = _inputs(
        lengths, dtype, heads=2, value_heads=4, key_width=7, value_width=11
    )
    if zero_state:
        state.zero_()
    else:
        state = state.transpose(-1, -2).contiguous().transpose(-1, -2)
    original = state.clone()
    boundaries = _strided(torch.tensor([0, 0, 1, 4, 4, 9], dtype=torch.int32))
    expected, expected_state = _reference(
        q, k, v, g, beta, state, lengths, normalize=normalize, scale=0.3
    )
    result = gdn_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=0.3,
        initial_state=state,
        cu_seqlens=boundaries,
        qk_l2norm=normalize,
        output_final_state=True,
        output_h=False,
    )
    _check_output(result.out, expected)
    torch.testing.assert_close(
        result.final_state.cpu(), expected_state.float(), rtol=3e-5, atol=3e-6
    )
    torch.testing.assert_close(state, original, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [1, 65, 129])
@torch.inference_mode()
def test_prefill_qwen35_geometry(tokens):
    args = _inputs(
        [tokens],
        torch.bfloat16,
        heads=16,
        value_heads=16,
        key_width=128,
        value_width=128,
    )
    q, k, v, g, beta, state = args
    expected, expected_state = _reference(
        *args, [tokens], normalize=True, scale=128**-0.5
    )
    result = gdn_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=None,
        initial_state=state,
        cu_seqlens=torch.tensor([0, tokens], device="npu", dtype=torch.int64),
        qk_l2norm=True,
        output_final_state=True,
        output_h=False,
    )
    _check_output(result.out, expected)
    torch.testing.assert_close(
        result.final_state.cpu(), expected_state.float(), rtol=3e-5, atol=3e-6
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("separate_output", [False, True])
@torch.inference_mode()
def test_decode_indexed_state_and_padding(dtype, normalize, separate_output):
    args = _inputs([1] * 5, dtype, heads=2, value_heads=4, key_width=7, value_width=11)
    q, k, v = [x.transpose(0, 1) for x in args[:3]]
    pool = _strided(torch.randn(7, 4, 11, 7) * 0.1)
    original = pool.clone()
    reads = [2, -1, 0, 4, -1]
    writes = [1, -1, 3, -1, 5] if separate_output else reads
    a = _strided(torch.linspace(-80, 80, 20).view(5, 1, 4))
    b = _strided(torch.linspace(80, -80, 20).view(5, 1, 4))
    A_log = _strided(torch.linspace(-2, 1, 4))
    bias = _strided(torch.linspace(-1, 1, 4))
    g = -A_log.cpu().exp() * F.softplus(a.cpu() + bias.cpu())
    beta = b.cpu().sigmoid()
    expected_pool = original.cpu().clone()
    expected = torch.empty(5, 1, 4, 11)
    for row, (source, destination) in enumerate(zip(reads, writes)):
        initial = (
            original[source : source + 1]
            if source >= 0
            else torch.zeros_like(original[:1])
        )
        out, final = _reference(
            q[row : row + 1],
            k[row : row + 1],
            v[row : row + 1],
            g[row : row + 1],
            beta[row : row + 1],
            initial,
            [1],
            normalize=normalize,
            scale=7**-0.5,
        )
        expected[row : row + 1] = out
        if destination >= 0:
            expected_pool[destination] = final[0]
    result = gdn_decode_step(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=bias,
        b=b,
        initial_state=pool,
        initial_state_indices=_strided(torch.tensor(reads, dtype=torch.int64)),
        output_state_indices=(
            _strided(torch.tensor(writes, dtype=torch.int32))
            if separate_output
            else None
        ),
        use_qk_l2norm=normalize,
    )
    live = torch.tensor(
        [source >= 0 or destination >= 0 for source, destination in zip(reads, writes)]
    )
    _check_output(result.cpu()[live], expected[live])
    torch.testing.assert_close(pool.cpu(), expected_pool, rtol=3e-5, atol=3e-6)


@torch.inference_mode()
def test_prefill_and_multistep_decode_continuity():
    q, k, v, _, _, state = _inputs(
        [17], torch.bfloat16, heads=2, value_heads=4, key_width=7, value_width=11
    )
    torch.manual_seed(20)
    a = torch.randn(1, 17, 4, device="npu")
    b = torch.randn_like(a)
    A_log = torch.randn(4, device="npu")
    bias = torch.randn(4, device="npu")
    g = -A_log.exp() * F.softplus(a + bias)
    beta = b.sigmoid()
    expected, expected_state = _reference(
        q, k, v, g, beta, state, [17], normalize=True, scale=7**-0.5
    )
    result = gdn_chunk_prefill(
        q[:, :5],
        k[:, :5],
        v[:, :5],
        g[:, :5],
        beta[:, :5],
        scale=None,
        initial_state=state,
        cu_seqlens=torch.tensor([0, 5], device="npu", dtype=torch.int32),
        qk_l2norm=True,
        output_final_state=True,
        output_h=False,
    )
    outputs = [result.out.cpu()]
    pool = torch.zeros(2, 4, 11, 7, device="npu")
    pool[0].copy_(result.final_state[0])
    for token in range(5, 17):
        source, destination = (token - 5) % 2, (token - 4) % 2
        out = gdn_decode_step(
            q[:, token : token + 1],
            k[:, token : token + 1],
            v[:, token : token + 1],
            A_log=A_log,
            a=a[:, token : token + 1],
            dt_bias=bias,
            b=b[:, token : token + 1],
            initial_state=pool,
            initial_state_indices=torch.tensor(
                [source], device="npu", dtype=torch.int32
            ),
            output_state_indices=torch.tensor(
                [destination], device="npu", dtype=torch.int32
            ),
            use_qk_l2norm=True,
        )
        outputs.append(out.cpu())
    _check_output(torch.cat(outputs, dim=1), expected)
    torch.testing.assert_close(
        pool[0].cpu(), expected_state[0].float(), rtol=3e-5, atol=3e-6
    )


@pytest.mark.parametrize("lengths", [[], [0], [0, 0], [3]])
@pytest.mark.parametrize("return_state", [False, True])
@torch.inference_mode()
def test_prefill_empty_sequences_and_optional_state(lengths, return_state):
    args = _inputs(
        lengths, torch.float32, heads=2, value_heads=4, key_width=7, value_width=11
    )
    q, k, v, g, beta, initial = args
    original = initial.clone()
    boundaries = torch.tensor(
        [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
        device="npu",
        dtype=torch.int32,
    )
    expected, final = _reference(*args, lengths, normalize=True, scale=7**-0.5)
    result = gdn_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=None,
        initial_state=initial,
        cu_seqlens=boundaries,
        qk_l2norm=True,
        output_final_state=return_state,
        output_h=False,
    )
    _check_output(result.out, expected)
    if return_state:
        torch.testing.assert_close(
            result.final_state.cpu(), final.float(), rtol=3e-5, atol=3e-6
        )
    else:
        assert result.final_state is None
    torch.testing.assert_close(initial, original, rtol=0, atol=0)


@torch.inference_mode()
def test_chunked_prefill_matches_uninterrupted_recurrence():
    args = _inputs(
        [17], torch.bfloat16, heads=2, value_heads=4, key_width=7, value_width=11
    )
    q, k, v, g, beta, state = args
    expected, final = _reference(*args, [17], normalize=True, scale=0.2)
    outputs = []
    start = 0
    for length in (3, 0, 5, 9):
        end = start + length
        result = gdn_chunk_prefill(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            scale=0.2,
            initial_state=state,
            cu_seqlens=torch.tensor([0, length], device="npu", dtype=torch.int32),
            qk_l2norm=True,
            output_final_state=True,
            output_h=False,
        )
        outputs.append(result.out)
        state = result.final_state
        start = end
    _check_output(torch.cat(outputs, dim=1), expected)
    torch.testing.assert_close(state.cpu(), final.float(), rtol=3e-5, atol=3e-6)


@torch.inference_mode()
def test_decode_qwen35_geometry():
    args = _inputs(
        [1, 1], torch.bfloat16, heads=16, value_heads=16, key_width=128, value_width=128
    )
    q, k, v = [x.transpose(0, 1) for x in args[:3]]
    state = args[-1]
    a = torch.randn(2, 1, 16, device="npu", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    A_log = torch.randn(16, device="npu")
    bias = torch.randn(16, device="npu")
    g = -A_log.exp() * F.softplus(a.float() + bias)
    beta = b.float().sigmoid()
    expected, final = _reference(
        args[0],
        args[1],
        args[2],
        g.transpose(0, 1),
        beta.transpose(0, 1),
        state,
        [1, 1],
        normalize=True,
        scale=128**-0.5,
    )
    result = gdn_decode_step(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=bias,
        b=b,
        initial_state=state,
        initial_state_indices=torch.arange(2, device="npu", dtype=torch.int32),
        use_qk_l2norm=True,
    )
    _check_output(result.transpose(0, 1), expected)
    torch.testing.assert_close(state.cpu(), final.float(), rtol=3e-5, atol=3e-6)


@torch.inference_mode()
def test_prefill_rejects_intermediate_checkpoints():
    q, k, v, g, beta, state = _inputs(
        [1], torch.bfloat16, heads=2, value_heads=4, key_width=7, value_width=11
    )
    with pytest.raises(NotImplementedError, match="intermediate checkpoints"):
        gdn_chunk_prefill(
            q,
            k,
            v,
            g,
            beta,
            scale=None,
            initial_state=state,
            cu_seqlens=torch.tensor([0, 1], device="npu", dtype=torch.int32),
            output_h=True,
            override="ascend_gdn_chunk_prefill",
        )


@pytest.mark.parametrize("decode", [False, True])
@torch.inference_mode()
def test_variable_batch_geometry_does_not_recompile(decode):
    import sys
    from pathlib import Path

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2] / "tokenspeed-kernel/test")
    )
    from tokenspeed_kernel_npu.ops.gdn import _gdn_recurrence_kernel
    from utils import assert_no_triton_compile

    def run(requests, tokens):
        args = _inputs(
            [tokens] * requests,
            torch.bfloat16,
            heads=2,
            value_heads=4,
            key_width=7,
            value_width=11,
        )
        q, k, v, g, beta, state = args
        if decode:
            q, k, v = [x.transpose(0, 1) for x in (q, k, v)]
            a, b = g.transpose(0, 1), beta.transpose(0, 1)
            gdn_decode_step(
                q,
                k,
                v,
                A_log=torch.zeros(4, device="npu"),
                a=a,
                dt_bias=torch.zeros(4, device="npu"),
                b=b,
                initial_state=state,
                initial_state_indices=torch.arange(
                    requests, device="npu", dtype=torch.int32
                ),
                use_qk_l2norm=True,
            )
        else:
            gdn_chunk_prefill(
                q,
                k,
                v,
                g,
                beta,
                scale=0.3,
                initial_state=state,
                cu_seqlens=torch.arange(requests + 1, device="npu", dtype=torch.int32)
                * tokens,
                qk_l2norm=True,
                output_final_state=True,
                output_h=False,
            )

    run(1, 1)
    with assert_no_triton_compile(_gdn_recurrence_kernel):
        for requests in (2, 3, 5):
            for tokens in ((1,) if decode else (3, 16, 65)):
                run(requests, tokens)


@pytest.mark.parametrize(
    "decay_logit,log_rate",
    [(-40.0, 40.0), (-20.0, 20.0), (-3.0, 0.0), (0.0, 0.0), (40.0, -3.0)],
)
@torch.inference_mode()
def test_decode_stable_decay_gates(decay_logit, log_rate):
    args = _inputs(
        [1], torch.float32, heads=2, value_heads=4, key_width=7, value_width=11
    )
    q, k, v, _, _, state = args
    a = torch.full((1, 1, 4), decay_logit, device="npu")
    b = torch.zeros_like(a)
    A_log = torch.full((4,), log_rate, device="npu")
    bias = torch.zeros_like(A_log)
    g = -A_log.cpu().double().exp() * F.softplus(a.cpu().double())
    expected, final = _reference(
        q, k, v, g, torch.full_like(g, 0.5), state, [1], normalize=True, scale=0.3
    )
    result = gdn_decode_step(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=bias,
        b=b,
        initial_state=state,
        initial_state_indices=torch.zeros(1, device="npu", dtype=torch.int32),
        scale=0.3,
        use_qk_l2norm=True,
    )
    _check_output(result, expected)
    torch.testing.assert_close(state.cpu(), final.float(), rtol=3e-5, atol=3e-6)


@torch.inference_mode()
def test_zero_keys_and_queries_preserve_state_with_unit_decay():
    args = _inputs(
        [3], torch.bfloat16, heads=2, value_heads=4, key_width=7, value_width=11
    )
    q, k, v, g, beta, state = args
    q.zero_()
    k.zero_()
    g.zero_()
    result = gdn_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=None,
        initial_state=state,
        cu_seqlens=torch.tensor([0, 3], device="npu", dtype=torch.int32),
        qk_l2norm=True,
        output_final_state=True,
        output_h=False,
    )
    torch.testing.assert_close(result.out, torch.zeros_like(v), rtol=0, atol=0)
    torch.testing.assert_close(result.final_state, state, rtol=0, atol=0)


@pytest.mark.parametrize("decode", [False, True])
@torch.inference_mode()
def test_rejects_non_fp32_state_before_mutation(decode):
    q, k, v, g, beta, state = _inputs(
        [1], torch.bfloat16, heads=2, value_heads=4, key_width=7, value_width=11
    )
    state = state.to(torch.bfloat16)
    original = state.clone()
    indices = torch.zeros(1, device="npu", dtype=torch.int32)
    with pytest.raises(ValueError, match="FP32 recurrent state"):
        if decode:
            gdn_decode_step(
                q,
                k,
                v,
                A_log=torch.zeros(4, device="npu"),
                a=g,
                dt_bias=torch.zeros(4, device="npu"),
                b=beta,
                initial_state=state,
                initial_state_indices=indices,
            )
        else:
            gdn_chunk_prefill(
                q,
                k,
                v,
                g,
                beta,
                scale=None,
                initial_state=state,
                cu_seqlens=torch.tensor([0, 1], device="npu", dtype=torch.int32),
            )
    torch.testing.assert_close(state, original, rtol=0, atol=0)
