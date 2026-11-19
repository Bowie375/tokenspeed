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


"""Observe real HCCL calls without synchronizing or changing their arguments."""

import sys
from functools import wraps

import torch


def install_collective_trace(capture):
    """Record actual input payloads and runtime callers during correctness forwards.

    Wrappers live only in disposable correctness workers. They preserve supplied
    arguments and return values; records describe payload bytes, not link traffic
    or measured communication duration.
    """
    from tokenspeed.runtime.distributed.comm_backend.hccl import HcclBackend

    def record(kind, tensor, group):
        if capture.pending is None or len(group) < 2:
            return
        caller = "unknown"
        frame = sys._getframe(1)
        try:
            while frame is not None:
                filename = frame.f_code.co_filename
                if any(
                    part in filename
                    for part in (
                        "comm_manager.py",
                        "comm_ops.py",
                        "logits_processor.py",
                        "vocab_parallel_embedding.py",
                        "linear.py",
                    )
                ):
                    caller = f"{filename.split('/tokenspeed/runtime/')[-1]}:{frame.f_code.co_name}:{frame.f_lineno}"
                    break
                frame = frame.f_back
        finally:
            del frame
        capture.collective_events.append(
            dict(
                operation=kind,
                shape=list(tensor.shape),
                dtype=str(tensor.dtype),
                payload_bytes=tensor.numel() * tensor.element_size(),
                ranks=list(group),
                caller=caller,
            )
        )

    reduce = HcclBackend.all_reduce
    gather = HcclBackend.all_gather_single
    scatter = HcclBackend.reduce_scatter

    @wraps(reduce)
    def all_reduce(self, tensor, group, *args, **kwargs):
        result = reduce(self, tensor, group, *args, **kwargs)
        if isinstance(tensor, torch.Tensor):
            record("all_reduce", tensor, group)
        return result

    @wraps(gather)
    def all_gather_single(self, output, input, group, *args, **kwargs):
        result = gather(self, output, input, group, *args, **kwargs)
        record("all_gather_into_tensor", input, group)
        return result

    @wraps(scatter)
    def reduce_scatter(self, tensor, group, *args, **kwargs):
        result = scatter(self, tensor, group, *args, **kwargs)
        record("reduce_scatter_tensor", tensor, group)
        return result

    HcclBackend.all_reduce = all_reduce
    HcclBackend.all_gather_single = all_gather_single
    HcclBackend.reduce_scatter = reduce_scatter


def validate_model_communication(directory):
    """Check real dense TP2 teacher forwards against layer and payload geometry."""
    import json
    from collections import Counter

    config = json.loads((directory / "config.json").read_text())
    model = json.loads((directory / "manifest.json").read_text())["model_config"]
    fixtures = json.loads((directory / "fixtures.json").read_text())
    checks = []
    for fixture in fixtures:
        for step in range(config["steps"]):
            traces = []
            for rank in (0, 1):
                filename = (
                    directory
                    / f"tp2/rank{rank}/teacher_{fixture['name']}/communication_{fixture['name']}_{step:03d}.json"
                )
                events = json.loads(filename.read_text()) if filename.exists() else []
                traces.append(events)
                rows = len(fixture["input_ids"]) if step == 0 else 1
                counts = Counter(e["operation"] for e in events)
                valid = counts == {
                    "all_reduce": 2 * model["num_hidden_layers"] + 1,
                    "all_gather_into_tensor": 1,
                }
                for event in events:
                    width = (
                        model["hidden_size"]
                        if event["operation"] == "all_reduce"
                        else model["vocab_size"] // 2
                    )
                    valid &= event["shape"] == [
                        rows if event["operation"] == "all_reduce" else 1,
                        width,
                    ]
                    valid &= event["dtype"] == "torch.bfloat16"
                    valid &= event["payload_bytes"] == 2 * width * (
                        rows if event["operation"] == "all_reduce" else 1
                    )
                    valid &= event["ranks"] == [0, 1]
                checks.append(
                    dict(
                        fixture=fixture["name"],
                        step=step,
                        rank=rank,
                        passed=valid,
                        counts=dict(counts),
                    )
                )
            checks.append(
                dict(
                    fixture=fixture["name"],
                    step=step,
                    passed=bool(traces[0]) and traces[0] == traces[1],
                    scope="exact peer trace agreement",
                )
            )
    result = dict(
        passed=bool(checks) and all(c["passed"] for c in checks),
        total=len(checks),
        checks=checks,
    )
    (directory / "communication_validation.json").write_text(
        json.dumps(result, indent=2)
    )
    return result


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    options = parser.parse_args()
    result = validate_model_communication(options.directory)
    print(f"communication checks: {result['total']}; passed={result['passed']}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
