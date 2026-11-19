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


"""Two-rank HCCL arithmetic validation and reproducible AllReduce microbenchmark."""

import argparse
import json
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from tokenspeed.runtime.distributed.comm_backend.hccl import HcclBackend
from tokenspeed.runtime.distributed.process_group_manager import process_group_manager


def bandwidth_gbps(*, payload_bytes: int, latency_us: float) -> float:
    """Return per-rank payload / latency, in decimal GB/s; no ring traffic multiplier."""
    if payload_bytes < 0 or latency_us <= 0:
        raise ValueError("payload must be nonnegative and latency positive")
    return payload_bytes / (latency_us * 1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    options = parser.parse_args()
    if options.warmups < 1 or options.repetitions < 1:
        parser.error("warmups and repetitions must be positive")
    import torch_npu  # Registers the NPU device and HCCL backend.

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 2:
        raise ValueError("this experiment requires exactly two ranks")
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    group = (0, 1)
    process_group_manager.register_process_group("hccl", group, dist.group.WORLD)
    backend = HcclBackend()
    options.directory.mkdir(parents=True, exist_ok=True)
    try:
        local = (
            torch.arange(6, device="npu", dtype=torch.float32).reshape(2, 3) + rank * 10
        )
        peers = [torch.arange(6).reshape(2, 3).float() + r * 10 for r in group]
        torch.testing.assert_close(
            backend.all_reduce(local.clone(), group).cpu(), sum(peers), rtol=0, atol=0
        )
        for dim in (0, 1, -1):
            torch.testing.assert_close(
                backend.all_gather(local, group, dim=dim).cpu(),
                torch.cat(peers, dim=dim),
                rtol=0,
                atol=0,
            )
        source = (
            torch.arange(12, device="npu", dtype=torch.float32).reshape(4, 3) + rank
        )
        expected = (torch.arange(12).reshape(4, 3).float() * 2 + 1).chunk(2)[rank]
        torch.testing.assert_close(
            backend.reduce_scatter(source, group).cpu(), expected, rtol=0, atol=0
        )
        for counts in ([2, 2], [1, 3], [0, 2]):
            shards = [
                torch.arange(n * 3).reshape(n, 3).float() + r * 100
                for r, n in enumerate(counts)
            ]
            torch.testing.assert_close(
                backend.token_all_gather(shards[rank].to("npu"), group, counts).cpu(),
                torch.cat(shards),
                rtol=0,
                atol=0,
            )
            source = torch.arange(sum(counts) * 3, dtype=torch.float32).reshape(-1, 3)
            start = sum(counts[:rank])
            expected = (source * 2 + 1)[start : start + counts[rank]]
            torch.testing.assert_close(
                backend.token_reduce_scatter(
                    (source + rank).to("npu"), group, counts
                ).cpu(),
                expected,
                rtol=0,
                atol=0,
            )
        benchmarks = []
        for rows, width in (
            (1, 1024),
            (128, 1024),
            (512, 1024),
            (1, 5120),
            (128, 5120),
        ):
            tensor = torch.empty((rows, width), dtype=torch.bfloat16, device="npu")
            samples = []
            for index in range(options.warmups + options.repetitions):
                tensor.fill_(rank + 1)
                torch.npu.synchronize()
                start = time.perf_counter_ns()
                backend.all_reduce(tensor, group)
                torch.npu.synchronize()
                elapsed = (time.perf_counter_ns() - start) / 1000
                if index >= options.warmups:
                    samples.append(elapsed)
            torch.testing.assert_close(
                tensor.cpu(),
                torch.full((rows, width), 3, dtype=torch.bfloat16),
                rtol=0,
                atol=0,
            )
            median = statistics.median(samples)
            benchmarks.append(
                dict(
                    shape=[rows, width],
                    dtype="BF16",
                    payload_bytes=tensor.numel() * tensor.element_size(),
                    latency_us=dict(median=median, min=min(samples), max=max(samples)),
                    samples_us=samples,
                    payload_bandwidth_gbps=bandwidth_gbps(
                        payload_bytes=tensor.numel() * tensor.element_size(),
                        latency_us=median,
                    ),
                )
            )
        (options.directory / f"rank{rank}.json").write_text(
            json.dumps(
                dict(
                    rank=rank,
                    device=local_rank,
                    arithmetic_passed=True,
                    bandwidth_definition="per-rank payload bytes / synchronized wall time; decimal GB/s; not link bus bandwidth",
                    benchmarks=benchmarks,
                ),
                indent=2,
            )
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
