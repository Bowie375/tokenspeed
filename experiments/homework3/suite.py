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


"""One-command native Homework 3 acceptance and capacity/performance matrix."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from experiments.homework3.analyze import analyze, compare_capture
from experiments.homework3.summary import summarize_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-model", type=Path, required=True)
    parser.add_argument("--large-model", type=Path, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument(
        "--task", choices=["functional", "performance", "all"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    chips = args.devices.split(",")
    if len(set(chips)) != 2:
        raise ValueError("select two distinct available physical chips")
    args.output.mkdir(parents=True, exist_ok=False)

    def run(name, model, tp, task, ratio, capacity, forced):
        directory = args.output / name
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.homework3.run",
                "--model",
                str(model),
                "--devices",
                ",".join(chips[:tp]),
                "--tp",
                str(tp),
                "--task",
                task,
                "--host-ratio",
                str(ratio),
                "--device-tokens",
                str(capacity),
                "--force-retraction",
                str(forced),
                "--output",
                str(directory),
            ],
            check=True,
        )
        analyze(directory, task=task, require_eviction=ratio == 0.5)
        return directory

    if args.task in ("functional", "all"):
        for name, model, capacity in [
            ("small", args.small_model, 8192),
            ("large", args.large_model, 16384),
        ]:
            for tp in (1, 2):
                run(
                    f"{name}_tp{tp}_functional", model, tp, "functional", 4, capacity, 0
                )
        run("small_host_budget", args.small_model, 1, "functional", 0.5, 8192, 0)
        for tp in (1, 2):
            actual = run(
                f"small_tp{tp}_retraction",
                args.small_model,
                tp,
                "retraction",
                4,
                8192,
                3,
            )
            reference = run(
                f"small_tp{tp}_uninterrupted",
                args.small_model,
                tp,
                "retraction",
                4,
                8192,
                0,
            )
            actual_outputs = [
                r["response"]["output_ids"]
                for r in json.loads((actual / "results.json").read_text())
                if r["action"] == "generate"
            ]
            reference_outputs = [
                r["response"]["output_ids"]
                for r in json.loads((reference / "results.json").read_text())
                if r["action"] == "generate"
            ]
            if actual_outputs != reference_outputs:
                raise AssertionError(
                    "active retraction changed native continuation tokens"
                )
            checks = [
                compare_capture(
                    reference / f"rank{rank}/uninterrupted",
                    actual / f"rank{rank}/uninterrupted",
                    exact_logits=True,
                )
                for rank in range(tp)
            ]
            (actual / "continuation_comparison.json").write_text(
                json.dumps(checks, indent=2)
            )
    if args.task in ("performance", "all"):
        directories = []
        for name, model, capacity in [
            ("small", args.small_model, 8192),
            ("large", args.large_model, 16384),
        ]:
            for ratio in (0, 0.5, 4):
                directories.append(
                    run(
                        f"{name}_performance_{ratio}",
                        model,
                        1,
                        "performance",
                        ratio,
                        capacity,
                        0,
                    )
                )
        result = summarize_matrix(directories)
        (args.output / "performance_matrix.json").write_text(
            json.dumps(result, indent=2)
        )
    (args.output / "complete.json").write_text(json.dumps(dict(passed=True)))


if __name__ == "__main__":
    main()
