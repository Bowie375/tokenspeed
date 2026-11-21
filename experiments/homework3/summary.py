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


"""Compare complete measured workloads only under identical Device resources."""

import argparse
import json
from pathlib import Path

from experiments.homework3.analyze import analyze


def check_resources(left: dict, right: dict):
    host_options = {"disable_kvstore", "kvstore_ratio", "kvstore_size"}
    for key in sorted((left.keys() | right.keys()) - host_options):
        if key not in left or key not in right or left[key] != right[key]:
            raise ValueError(f"unfair performance comparison: {key} differs")


def summarize_matrix(directories: list[Path]):
    models = {}
    jobs_by_model = {}
    for directory in directories:
        launch = json.loads((directory / "launch.json").read_text())
        jobs = json.loads((directory / "jobs.json").read_text())
        reference = jobs_by_model.setdefault(launch["model"], jobs)
        if jobs != reference:
            raise ValueError("unfair performance comparison: workload differs")
        analysis = analyze(directory, task="performance", require_eviction=False)
        if set(analysis["workloads"]) != {"high", "low"}:
            raise ValueError("performance workload needs both high and low reuse")
        entry = models.setdefault(launch["model"], dict(launch=launch, rows=[]))
        check_resources(entry["launch"], launch)
        # Ensure the actual resolved allocator capacity agrees, too.
        if (
            entry["rows"]
            and entry["rows"][0]["ranks"][0]["device_total_blocks"]
            != analysis["ranks"][0]["device_total_blocks"]
        ):
            raise ValueError("resolved Device block capacities differ")
        entry["rows"].append(
            dict(
                host_ratio=launch["kvstore_ratio"],
                ranks=analysis["ranks"],
                workloads=analysis["workloads"],
            )
        )
    for entry in models.values():
        ratios = [r["host_ratio"] for r in entry["rows"]]
        if len(set(ratios)) != 3 or 0 not in ratios:
            raise ValueError("need Host off and two distinct Host budgets")
    return dict(passed=True, models=models)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    options.output.write_text(
        json.dumps(summarize_matrix(options.directories), indent=2)
    )


if __name__ == "__main__":
    main()
