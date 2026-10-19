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


"""Disposable worker instrumentation around the real eager forward step.

Primary latency requests have no added device synchronization. Only the
separate forward-diagnostic pass synchronizes at complete forward boundaries.
No tensor contents, sampling choices, or cache ownership are modified.
"""

import functools
import inspect
import json
import time
from pathlib import Path

import torch

MEMORY_FIELDS = (
    "memory_allocated",
    "memory_reserved",
    "max_memory_allocated",
    "max_memory_reserved",
)


def memory_snapshot() -> dict:
    """Read framework allocator bytes; missing APIs remain explicitly unavailable."""
    result = {}
    for name in MEMORY_FIELDS:
        function = getattr(torch.npu, name, None)
        try:
            result[name] = int(function()) if function is not None else None
        except (RuntimeError, AttributeError) as error:
            result[name] = None
            result[f"{name}_error"] = str(error)
    return result


class ForwardCapture:
    """Keep one live request's scalar records and flush at the next prompt."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.active: dict | None = None
        self.events: list[dict] = []
        self.before: dict = {}
        self.after: dict = {}
        self.profiler = None
        self.startup_recorded = False
        self.memory_reset_supported = False

    def begin(self):
        requested = json.loads((self.directory / "active.json").read_text())
        if self.active is not None and requested.get("id") == self.active["id"]:
            return
        self.flush()
        self.active = requested if requested["enabled"] else None
        self.events = []
        if self.active is None:
            return
        if torch.npu.device_count() != 1:
            raise RuntimeError(
                "Performance measurement requires exactly one visible NPU"
            )
        self.before = memory_snapshot()
        if not self.startup_recorded:
            startup = dict(
                allocator_bytes=self.before,
                device_name=torch.npu.get_device_name(0),
                visible_devices=torch.npu.device_count(),
            )
            (self.directory / "startup_memory.json").write_text(
                json.dumps(startup, indent=2)
            )
            self.startup_recorded = True
        reset = getattr(torch.npu, "reset_peak_memory_stats", None)
        self.memory_reset_supported = reset is not None
        if reset is not None:
            reset()
        if self.active["phase"] == "profile":
            import torch_npu

            torch.npu.synchronize()
            self.profiler = torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                record_shapes=True,
                profile_memory=False,
                schedule=torch_npu.profiler.schedule(
                    wait=0, warmup=0, active=1, repeat=1
                ),
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    export_type="text"
                ),
                with_stack=False,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    str(self.directory / "profiles" / self.active["case"])
                ),
            )
            self.profiler.start()

    def flush(self):
        """Finish a profile and persist the preceding request outside its latency."""
        if self.profiler is not None:
            torch.npu.synchronize()
            # One logical profiling step spans the complete request. Advance to
            # NONE to save it explicitly before stopping the profiler.
            self.profiler.step()
            self.profiler.stop()
            self.profiler = None
        if self.active is None:
            return
        payload = dict(
            request=self.active,
            events=self.events,
            memory_before_bytes=self.before,
            memory_after_bytes=self.after,
            peak_reset_supported=self.memory_reset_supported,
        )
        target = self.directory / "instrumentation" / f"{self.active['id']}.json"
        target.write_text(json.dumps(payload, indent=2))
        self.active = None


def install(directory: Path):
    """Install once in an isolated Engine process, never in a running server."""
    from tokenspeed.runtime.execution.model_executor import ModelExecutor

    capture = ForwardCapture(directory)
    original = ModelExecutor._forward_step
    signature = inspect.signature(original)

    @functools.wraps(original)
    def forward(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs).arguments
        ctx = bound["ctx"]
        if ctx.num_extends:
            capture.begin()
        active = capture.active
        if active is None:
            return original(self, *args, **kwargs)
        if ctx.bs != 1:
            raise RuntimeError("benchmark request unexpectedly executed in a batch")
        kind = "prefill" if ctx.num_extends else "decode"
        elapsed = None
        if active["phase"] == "forward":
            torch.npu.synchronize()
            start = time.perf_counter()
            result = original(self, *args, **kwargs)
            torch.npu.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
        elif active["phase"] == "profile":
            with torch.profiler.record_function(f"homework.performance.{kind}"):
                result = original(self, *args, **kwargs)
        else:
            result = original(self, *args, **kwargs)
        capture.events.append(
            dict(kind=kind, duration_ms=elapsed, input_tokens=ctx.input_num_tokens)
        )
        capture.after = memory_snapshot()
        return result

    ModelExecutor._forward_step = forward
    return capture
