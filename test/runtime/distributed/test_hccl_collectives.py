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

"""Real two-process arithmetic coverage of the HCCL backend's portable API.

Gloo exercises distributed tensor semantics on CPU. The homework communication
runner additionally executes the same backend on actual HCCL/NPU tensors.
"""

from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tokenspeed.runtime.distributed.comm_backend.hccl import HcclBackend
from tokenspeed.runtime.distributed.process_group_manager import process_group_manager


def _collectives_worker(rank: int, rendezvous: str):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=40),
    )
    try:
        group = (0, 1)
        process_group_manager.register_process_group("hccl", group, dist.group.WORLD)
        from experiments.homework2.communication import install_collective_trace
        from experiments.homework2.hooks import TPCapture

        capture = TPCapture(
            Path(rendezvous.removeprefix("file://")).parent / f"rank{rank}"
        )
        capture.pending = {}
        install_collective_trace(capture)
        backend = HcclBackend()
        local = torch.arange(6, dtype=torch.float32).reshape(2, 3) + rank * 10
        peers = [
            torch.arange(6, dtype=torch.float32).reshape(2, 3) + r * 10 for r in group
        ]
        for dim in (0, 1, -1):
            torch.testing.assert_close(
                backend.all_gather(local, group, dim=dim),
                torch.cat(peers, dim=dim),
                rtol=0,
                atol=0,
            )
        reduced = backend.all_reduce(local.clone(), group)
        torch.testing.assert_close(reduced, sum(peers), rtol=0, atol=0)
        merged = torch.arange(12, dtype=torch.float32).reshape(4, 3) + rank
        expected = (torch.arange(12, dtype=torch.float32).reshape(4, 3) * 2 + 1).chunk(
            2
        )[rank]
        torch.testing.assert_close(
            backend.reduce_scatter(merged, group), expected, rtol=0, atol=0
        )
        for counts in ([1, 3], [0, 2], [2, 2], [0, 0]):
            shards = [
                torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) + r * 100
                for r, n in enumerate(counts)
            ]
            torch.testing.assert_close(
                backend.token_all_gather(shards[rank], group, counts),
                torch.cat(shards),
                rtol=0,
                atol=0,
            )
            source = (
                torch.arange(sum(counts) * 3, dtype=torch.float32).reshape(-1, 3) + rank
            )
            offset = sum(counts[:rank])
            expected = (
                torch.arange(sum(counts) * 3, dtype=torch.float32).reshape(-1, 3) * 2
                + 1
            )[offset : offset + counts[rank]]
            torch.testing.assert_close(
                backend.token_reduce_scatter(source, group, counts),
                expected,
                rtol=0,
                atol=0,
            )
        from transformers import PretrainedConfig

        from tokenspeed.runtime.layers.logits_processor import LogitsProcessor

        processor = LogitsProcessor(
            PretrainedConfig(vocab_size=6),
            skip_all_gather=False,
            do_argmax=False,
            logit_scale=None,
            tp_rank=rank,
            tp_size=2,
            tp_group=group,
            dp_lm_head_tp=False,
        )
        head = torch.nn.Linear(3, 3, bias=False, dtype=torch.bfloat16)
        head.requires_grad_(False)
        weights = torch.arange(18, dtype=torch.bfloat16).reshape(6, 3)
        with torch.no_grad():
            head.weight.copy_(weights.chunk(2)[rank])
        hidden = torch.tensor([[1, 2, 3]], dtype=torch.bfloat16)
        actual = processor._get_logits(
            hidden,
            head,
            logits_metadata=None,
            embedding_bias=None,
            plan=None,
            require_full_vocab=False,
        )
        torch.testing.assert_close(
            actual, (hidden @ weights.T).to(actual.dtype), rtol=0, atol=0
        )
        events = capture.collective_events
        assert any(
            e["operation"] == "all_reduce"
            and e["shape"] == [2, 3]
            and e["payload_bytes"] == 24
            for e in events
        )
        assert any(
            e["operation"] == "all_gather_into_tensor"
            and e["shape"] == [1, 3]
            and e["payload_bytes"] == 6
            for e in events
        )
        assert any(e["operation"] == "reduce_scatter_tensor" for e in events)
    finally:
        dist.destroy_process_group()


def test_collectives_preserve_real_peer_values(tmp_path):
    mp.spawn(_collectives_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2)
