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


"""Run fresh streaming Engine requests; all model work stays in its NPU process."""

import argparse
import dataclasses
import json
import subprocess
import threading
import time
import traceback
from pathlib import Path

from experiments.homework1.evaluation.fixtures import load_fixtures
from experiments.homework1.performance.config import (
    PerformanceConfig,
    engine_options,
    workload_matrix,
)
from experiments.homework1.performance.measurements import StreamMeasurements


def device_snapshot(directory: Path, name: str):
    """Save device-wide npu-smi output and its actual collection interval."""
    start = time.perf_counter()
    try:
        result = subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True, timeout=20, check=False
        )
        value = dict(
            returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        value = dict(error=str(error))
    value.update(
        start_monotonic=start,
        end_monotonic=time.perf_counter(),
        scope="device-wide; includes other processes, not framework allocator bytes",
    )
    (directory / f"npu_smi_{name}.json").write_text(json.dumps(value, indent=2))


def set_active(directory: Path, value: dict):
    temporary = directory / "active.next.json"
    temporary.write_text(json.dumps(value))
    temporary.replace(directory / "active.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    options = parser.parse_args()
    directory = options.directory
    value = json.loads((directory / "config.json").read_text())
    value["input_lengths"] = tuple(value["input_lengths"])
    value["output_lengths"] = tuple(value["output_lengths"])
    config = PerformanceConfig(**value)
    checkpoint = json.loads((directory / "launch_config.json").read_text())["model"]
    workloads = workload_matrix(config, load_fixtures(directory / "fixtures.json"))
    eos_ids = json.loads((directory / "generation_policy.json").read_text())[
        "eos_token_ids"
    ]
    device_snapshot(directory, "idle")
    from tokenspeed.runtime.entrypoints.engine import Engine

    engine = None
    completed = False
    request_index = 0
    try:
        engine = Engine(**engine_options(config, checkpoint))
        (directory / "effective_server_args.json").write_text(
            json.dumps(dataclasses.asdict(engine.server_args), indent=2, default=str)
        )
        device_snapshot(directory, "allocated")

        def request(workload, *, phase, warmup, repetition):
            nonlocal request_index
            request_index += 1
            tag = dict(
                id=f"request_{request_index:05}",
                enabled=True,
                case=workload.name,
                phase=phase,
                warmup=warmup,
                repetition=repetition,
                input_tokens=len(workload.input_ids),
            )
            set_active(directory, tag)
            start = time.perf_counter()
            stream = StreamMeasurements(start=start)
            sample = dict(
                **tag,
                target_output_tokens=workload.output_tokens,
                actual_output_tokens=0,
                valid=False,
            )
            snapshot = None
            try:
                for event in engine.generate(
                    input_ids=list(workload.input_ids),
                    sampling_params=dict(
                        temperature=0,
                        max_new_tokens=workload.output_tokens,
                        ignore_eos=config.eos_policy == "fixed_length",
                        stop_token_ids=(
                            [] if config.eos_policy == "fixed_length" else eos_ids
                        ),
                    ),
                    stream=True,
                ):
                    stream.observe(time.perf_counter(), event)
                    if phase == "memory" and snapshot is None:
                        snapshot = threading.Thread(
                            target=device_snapshot,
                            args=(directory, workload.name),
                            daemon=True,
                        )
                        snapshot.start()
                sample.update(
                    stream.finish(
                        end=time.perf_counter(),
                        target_tokens=workload.output_tokens,
                        eos_policy=config.eos_policy,
                    )
                )
            except Exception as error:
                sample.update(
                    stream.fail(
                        end=time.perf_counter(),
                        target_tokens=workload.output_tokens,
                        eos_policy=config.eos_policy,
                        error=str(error),
                    )
                )
                sample["traceback"] = traceback.format_exc()
                raise
            finally:
                sample["request_start_monotonic"] = start
                sample["request_end_monotonic"] = time.perf_counter()
                with (directory / "requests.jsonl").open("a") as stream_file:
                    stream_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                if snapshot is not None:
                    snapshot.join(timeout=25)
            print(
                f"{phase} {workload.name} warmup={warmup} repetition={repetition} tokens={sample['actual_output_tokens']} valid={sample['valid']}",
                flush=True,
            )

            # Export the previous request's buffered records/profile outside all
            # measured intervals. A one-token fresh control request triggers the
            # flush on the same forward thread, before the next client timer.
            set_active(directory, dict(id=f"flush_{request_index:05}", enabled=False))
            engine.generate(
                input_ids=[workloads[0].input_ids[0]],
                sampling_params=dict(temperature=0, max_new_tokens=1, ignore_eos=True),
            )

        for phase in ("latency", "forward"):
            # Every matrix shape is warmed before any measurement in this pass.
            for workload in workloads:
                for iteration in range(config.warmups):
                    request(workload, phase=phase, warmup=True, repetition=iteration)
            for workload in workloads:
                for iteration in range(config.repetitions):
                    request(workload, phase=phase, warmup=False, repetition=iteration)
        profile_workloads = [
            next(
                w
                for w in workloads
                if len(w.input_ids) == length
                and w.output_tokens == min(config.output_lengths)
            )
            for length in config.input_lengths
        ]
        for workload in profile_workloads:
            request(workload, phase="memory", warmup=False, repetition=0)
        if config.profile:
            for workload in profile_workloads:
                request(workload, phase="profile", warmup=False, repetition=0)
        completed = True
    finally:
        (directory / "worker_status.json").write_text(
            json.dumps(dict(completed=completed, requests=request_index), indent=2)
        )
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
