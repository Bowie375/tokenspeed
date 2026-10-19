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

"""Prefill sampling must preserve the cache frontier before accepted advancement."""

import pytest
import torch

pytest.importorskip("torch_npu")

from tokenspeed_kernel.ops.metadata import advance_accepted_frontier

from tokenspeed.runtime.execution.cache_loc_kernel import fused_decode_input_prep
from tokenspeed.runtime.execution.model_executor import _with_prefill_sample_offsets
from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo


@pytest.mark.parametrize("device", ["cpu", "npu"])
@pytest.mark.parametrize("slots,lengths", [([4], [3]), ([5, 2, 4], [1, 3, 2048])])
@torch.inference_mode()
def test_sampling_offsets_preserve_cache_frontier(device, slots, lengths):
    if device == "npu" and not torch.npu.is_available():
        pytest.skip("requires Ascend")
    original = torch.tensor([0, 19, 7, 31, 0, 128], dtype=torch.int32)
    cache_lengths = original.to(device).clone()
    indices = torch.tensor(slots, dtype=torch.int64, device=device)
    input_lengths = torch.tensor(lengths, dtype=torch.int32, device=device)
    info = SamplingBatchInfo(
        valid_cache_lengths=cache_lengths, req_pool_indices=indices, device=device
    )
    adjusted = _with_prefill_sample_offsets(info, input_lengths)
    expected_offsets = original.clone()
    expected_frontier = original.clone()
    for slot, length in zip(slots, lengths):
        expected_offsets[slot] += length - 1
        expected_frontier[slot] += length
    torch.testing.assert_close(
        adjusted.valid_cache_lengths.cpu(), expected_offsets, rtol=0, atol=0
    )
    torch.testing.assert_close(cache_lengths.cpu(), original, rtol=0, atol=0)
    advance_accepted_frontier(
        indices,
        input_lengths,
        torch.ones_like(input_lengths),
        cache_lengths,
        len(slots),
        0,
        ngram_tail=None,
        ngram_previous_tokens=None,
        ngram_token_mask=None,
        input_ids=None,
    )
    torch.testing.assert_close(cache_lengths.cpu(), expected_frontier, rtol=0, atol=0)
    if device == "npu":
        positions = torch.empty(len(slots), dtype=torch.int64, device=device)
        seq_lens = torch.empty(len(slots), dtype=torch.int32, device=device)
        fused_decode_input_prep(
            positions, seq_lens, indices, cache_lengths, uniform_input_length=1
        )
        torch.testing.assert_close(
            positions.cpu(), expected_frontier[slots].long(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            seq_lens.cpu(), expected_frontier[slots] + 1, rtol=0, atol=0
        )
