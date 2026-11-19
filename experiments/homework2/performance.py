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


"""Validate complete TP performance evidence and summarize recorded measurements."""

import argparse
import json
import statistics
from pathlib import Path

from experiments.homework2.validation import (
    validate_performance,
    validate_performance_resources,
)


def validate_rank_memory(directory: Path, *, world_size: int) -> dict:
    """Require real post-load and post-forward counters from every device process."""
    errors = []
    for rank in range(world_size):
        rd = directory / f"rank{rank}"
        try:
            loaded = json.loads((rd / "load_memory.json").read_text())
            memory = json.loads((rd / "memory.json").read_text())
            model = json.loads((rd / "model.json").read_text())
            valid = (
                loaded["allocated"] > 0
                and loaded["reserved"] >= loaded["allocated"]
                and memory["allocated"] > 0
                and memory["reserved"] >= memory["allocated"]
                and memory["peak_allocated"] >= memory["allocated"]
                and memory["peak_reserved"] >= memory["reserved"]
                and model["world_size"] == world_size
                and model["rank"] == rank
                and bool(model["parameters"])
                and all(
                    p["device"].startswith("npu:") for p in model["parameters"].values()
                )
            )
            if not valid:
                errors.append(f"invalid rank {rank} model/memory evidence")
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"missing/invalid rank {rank} evidence: {error}")
    return dict(passed=not errors, errors=errors)


def analyze_performance(directory: Path) -> dict:
    """Enforce equal resources, complete trials and every rank's memory evidence."""
    config = json.loads((directory / "config.json").read_text())
    checks = {}
    summary = {}
    for tp in (1, 2):
        rd = directory / f"tp{tp}"
        local = json.loads((rd / "config.json").read_text())
        validate_performance_resources(config, local)
        records = json.loads((rd / "performance.json").read_text())
        checks[f"tp{tp}_requests"] = validate_performance(records, config)
        checks[f"tp{tp}_device_evidence"] = validate_rank_memory(rd, world_size=tp)
        if not checks[f"tp{tp}_requests"]["passed"]:
            summary[f"tp{tp}"] = dict(
                skipped="invalid request matrix; inspect validation.json"
            )
            continue
        rows = []
        for record in records:
            samples = [s for trial in record["trials"] for s in trial["requests"]]
            rows.append(
                dict(
                    case=record["case"],
                    parallel=record["parallel"],
                    requests=len(samples),
                    latency_median_ms={
                        k: statistics.median(s[k] for s in samples)
                        for k in ("client_ttft_ms", "client_tpot_ms", "e2e_ms")
                    },
                    throughput_median_tokens_per_second=statistics.median(
                        t["throughput_tokens_per_second"] for t in record["trials"]
                    ),
                )
            )
        summary[f"tp{tp}"] = rows
    result = dict(
        passed=all(c["passed"] for c in checks.values()),
        checks=checks,
        scope="complete performance matrix, equal configured resources, actual per-rank model/memory evidence",
    )
    (directory / "validation.json").write_text(json.dumps(result, indent=2))
    (directory / "performance_summary.json").write_text(json.dumps(summary, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    options = parser.parse_args()
    result = analyze_performance(options.directory)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
