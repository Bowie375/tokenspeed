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


"""Conservative CPU/NPU timeline analysis; duration sums are not wall time."""

import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path


def _group(name: str) -> str:
    text = name.lower()
    for group, words in (
        (
            "copies_layout",
            (
                "copy",
                "memcpy",
                "transpose",
                "contiguous",
                "gather",
                "scatter",
                "index_",
                "pack",
                "split",
            ),
        ),
        ("convolution", ("conv",)),
        ("gdn", ("gdn", "delta_rule")),
        ("full_attention", ("attention", "mha", "softmax")),
        (
            "normalization_activation",
            ("norm", "silu", "sigmoid", "gelu", "rotary", "rope"),
        ),
        ("projections_mlp", ("matmul", "gemm", "linear", "mm", "muladd")),
    ):
        if any(word in text for word in words):
            return group
    return "unclassified"


def summarize_events(events: list[dict]) -> dict:
    """Separate CPU/device complete events; union device intervals across streams."""
    # CANN emits complete hardware events without a category. Process metadata
    # distinguishes them from its overlapping "Overlap Analysis" summary track.
    hardware_processes = {
        event.get("pid")
        for event in events
        if event.get("ph") == "M"
        and event.get("name") == "process_name"
        and event.get("args", {}).get("name") == "Ascend Hardware"
    }
    groups = defaultdict(float)
    cpu, intervals, names = 0.0, [], defaultdict(float)
    for event in events:
        if event.get("ph") != "X" or "dur" not in event or "ts" not in event:
            continue
        duration = float(event["dur"])
        if duration < 0:
            continue
        category = str(event.get("cat", "")).lower()
        if event.get("pid") in hardware_processes or category in (
            "kernel",
            "npu",
            "device",
            "ascend_hardware",
            "aicore",
            "aicpu",
        ):
            name = event.get("name", "unknown")
            groups[_group(name)] += duration
            names[name] += duration
            timestamp = Decimal(str(event["ts"]))
            intervals.append((timestamp, timestamp + Decimal(str(duration))))
        elif category in ("cpu_op", "user_annotation"):
            cpu += duration
    if intervals:
        origin = min(start for start, _ in intervals)
        intervals = [
            (float(start - origin), float(stop - origin)) for start, stop in intervals
        ]
    busy, end = 0.0, float("-inf")
    for start, stop in sorted(intervals):
        busy += max(0, stop - max(start, end))
        end = max(end, stop)
    span = max((stop for _, stop in intervals), default=0) - min(
        (start for start, _ in intervals), default=0
    )
    return dict(
        status="observed" if intervals else "no_device_events",
        device_events=len(intervals),
        device_duration_sum_us=sum(groups.values()),
        device_busy_union_us=busy,
        device_span_us=span,
        device_idle_gaps_us=max(0, span - busy),
        cpu_duration_sum_us=cpu,
        device_groups_us=dict(groups),
        dominant_observed_group=max(groups, key=groups.get) if groups else None,
        top_device_names=sorted(names.items(), key=lambda item: item[1], reverse=True)[
            :20
        ],
        interpretation="Name-based categories; overlapping duration sums are not wall time. Idle gaps alone do not establish a host or memory-bandwidth bottleneck.",
    )


def analyze_profiles(directory: Path) -> list[dict]:
    """Analyze exported Chrome device timelines; preserve unsupported/empty traces."""
    rows = []
    for path in sorted(directory.rglob("*")):
        if (
            not path.is_file()
            or path.name not in ("trace_view.json", "trace.json")
            and not path.name.endswith(".pt.trace.json")
        ):
            continue
        try:
            value = json.loads(path.read_text())
            events = value["traceEvents"] if isinstance(value, dict) else value
            rows.append(dict(path=str(path), **summarize_events(events)))
        except (ValueError, KeyError, TypeError) as error:
            rows.append(dict(path=str(path), status="parse_error", error=str(error)))
    return rows
