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


"""Execute one owned native engine; emit responses and quiescent block gauges."""

import argparse
import json
import time
from pathlib import Path


def atomic_json(path: Path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory
    launch = json.loads((directory / "launch.json").read_text())
    jobs = json.loads((directory / "jobs.json").read_text())
    from tokenspeed.runtime.entrypoints.engine import Engine

    engine = None
    results = []

    def control(action, name):
        atomic_json(directory / "command.json", dict(id=name, action=action))
        deadline = time.monotonic() + 90
        paths = [
            directory / f"rank{rank}" / f"ack_{name}.json"
            for rank in range(launch["world_size"])
        ]
        while not all(path.exists() for path in paths):
            if time.monotonic() > deadline:
                raise TimeoutError(f"quiescent {action}: {name}")
            time.sleep(0.02)
        return [json.loads(path.read_text()) for path in paths]

    try:
        atomic_json(directory / "active.json", dict(name="warmup", capture=False))
        engine = Engine(**launch)
        for job in jobs:
            if job["action"] != "generate":
                result = dict(
                    name=job["name"],
                    action=job["action"],
                    ranks=control(job["action"], job["name"]),
                )
            else:
                atomic_json(
                    directory / "active.json",
                    dict(name=job["name"], capture=job["capture"]),
                )
                start = time.perf_counter()
                response = engine.generate(
                    input_ids=job["ids"],
                    sampling_params=dict(
                        temperature=0, max_new_tokens=job["steps"], ignore_eos=True
                    ),
                )
                elapsed = time.perf_counter() - start
                result = dict(
                    name=job["name"],
                    action="generate",
                    response=response,
                    seconds=elapsed,
                    input_tokens=len(job["ids"]),
                    output_tokens=job["steps"],
                    ranks=control("stats", job["name"]),
                )
            results.append(result)
            atomic_json(directory / "results.json", results)
        atomic_json(directory / "complete.json", dict(completed=True))
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
