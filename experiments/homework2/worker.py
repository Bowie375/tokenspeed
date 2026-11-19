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


"""Disposable native TP worker; no external server is required or terminated."""

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.homework2.measurements import TokenMeasurements


def set_active(directory: Path, payload: dict):
    temporary = directory / "active.next.json"
    temporary.write_text(json.dumps(payload))
    temporary.replace(directory / "active.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    options = parser.parse_args()
    directory = options.directory
    config = json.loads((directory / "config.json").read_text())
    launch = json.loads((directory / "launch.json").read_text())
    from tokenspeed.runtime.entrypoints.engine import Engine

    engine = None
    completed = False
    try:
        engine = Engine(**launch)
        if config["task"] == "correctness":
            for job in json.loads((directory / "jobs.json").read_text()):
                set_active(directory, dict(job, capture_enabled=True))
                result = engine.generate(
                    input_ids=[f["input_ids"] for f in job["fixtures"]],
                    sampling_params=dict(
                        temperature=0, max_new_tokens=job["steps"], ignore_eos=True
                    ),
                )
                (directory / f"{job['name']}_response.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2)
                )
        else:
            records = []

            def request(ids, steps):
                start = time.perf_counter()
                measure = TokenMeasurements(start=start)
                for event in engine.generate(
                    input_ids=ids,
                    sampling_params=dict(
                        temperature=0, max_new_tokens=steps, ignore_eos=True
                    ),
                    stream=True,
                ):
                    measure.observe(time.perf_counter(), event)
                return measure.finish(
                    end=time.perf_counter(),
                    target_tokens=steps,
                    eos_policy="fixed_length",
                )

            for case in config["workloads"]:
                ids = [42] * case["input_tokens"]
                steps = case["output_tokens"]
                for _ in range(config["warmups"]):
                    request(ids, steps)
                for parallel in (False, True):
                    trials = []
                    for iteration in range(config["repetitions"]):
                        start = time.perf_counter()
                        if parallel:
                            with ThreadPoolExecutor(
                                max_workers=config["concurrency"]
                            ) as pool:
                                samples = list(
                                    pool.map(
                                        lambda _: request(ids, steps),
                                        range(config["concurrency"]),
                                    )
                                )
                        else:
                            samples = [
                                request(ids, steps)
                                for _ in range(config["concurrency"])
                            ]
                        elapsed = time.perf_counter() - start
                        trial = dict(
                            repetition=iteration,
                            requests=samples,
                            wall_seconds=elapsed,
                            throughput_tokens_per_second=sum(
                                s["actual_output_tokens"] for s in samples
                            )
                            / elapsed,
                        )
                        trials.append(trial)
                        with (directory / "performance.jsonl").open("a") as stream:
                            stream.write(
                                json.dumps(dict(case=case, parallel=parallel, **trial))
                                + "\n"
                            )
                    records.append(
                        dict(
                            case=case,
                            parallel=parallel,
                            trials=trials,
                            throughput_median=statistics.median(
                                t["throughput_tokens_per_second"] for t in trials
                            ),
                        )
                    )
            (directory / "performance.json").write_text(json.dumps(records, indent=2))
            set_active(directory, dict(capture_enabled=False, phase="memory"))
            request([42] * max(w["input_tokens"] for w in config["workloads"]), 2)
        completed = True
    finally:
        (directory / "worker_status.json").write_text(
            json.dumps(dict(completed=completed))
        )
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
