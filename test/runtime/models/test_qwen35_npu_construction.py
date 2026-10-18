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

"""Dense Qwen3.5 construction and embeddings on Ascend without CUDA."""

import pytest
import torch

pytest.importorskip("torch_npu")

from tokenspeed.runtime.configs.qwen3_5_config import Qwen3_5TextConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.models.qwen3_5 import Qwen3_5ForCausalLM

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="requires an Ascend NPU"
)


@pytest.mark.parametrize("full_attention_interval", [1, 2])
def test_dense_model_constructs_and_embeds_on_npu(full_attention_interval: int):
    config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        full_attention_interval=full_attention_interval,
        dtype=torch.bfloat16,
    )
    with torch.device("npu"), torch.no_grad():
        model = Qwen3_5ForCausalLM(config, Mapping(rank=0, world_size=1))
        embedding = model.get_input_embeddings().to(dtype=torch.bfloat16)
        columns = torch.arange(32, dtype=torch.bfloat16)
        weights = columns[:, None] + columns[None, :]
        embedding.weight[:32].copy_(weights)
        tokens = torch.tensor([0, 7, 31], dtype=torch.int64)
        actual = embedding(tokens)
        expected = torch.tensor([0, 7, 31], dtype=torch.bfloat16)[:, None] + columns

    torch.npu.synchronize()
    assert actual.device.type == "npu"
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
