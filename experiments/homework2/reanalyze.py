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


"""Reanalyze completed native evidence without rerunning inference or discarding history."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def reanalyze(directory: Path, *, reference: str, communication: str) -> dict:
    """Require complete native workers and preserve prior orchestration status."""
    from experiments.homework2.analyze import analyze

    if reference not in ("hf", "none"):
        raise ValueError("reference must be hf or none")
    for tp in (1, 2):
        status = json.loads((directory / f"tp{tp}/worker_status.json").read_text())
        if status["completed"] is not True:
            raise ValueError(f"TP{tp} native worker did not complete")
    config = json.loads((directory / "config.json").read_text())
    if config["task"] != "correctness":
        raise ValueError("only correctness captures can be reanalyzed")
    manifest = json.loads((directory / "manifest.json").read_text())
    result = analyze(
        directory,
        config,
        json.loads((directory / "fixtures.json").read_text()),
        manifest["model_config"],
        reference=reference,
    )
    if communication not in ("check", "unavailable"):
        raise ValueError("communication must be check or unavailable")
    if communication == "check":
        from experiments.homework2.communication import validate_model_communication

        communication_result = validate_model_communication(directory)
        result["communication_passed"] = communication_result["passed"]
        result["passed"] &= communication_result["passed"]
    else:
        result["communication_passed"] = None
    status_path = directory / "status.json"
    previous = json.loads(status_path.read_text()) if status_path.exists() else None
    record = dict(
        timestamp=datetime.now(timezone.utc).isoformat(),
        previous_status=previous,
        passed=result["passed"],
        reference=reference,
        total=result["total"],
        communication=communication,
    )
    with (directory / "reanalysis_history.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    status_path.write_text(
        json.dumps(
            dict(
                completed=result["passed"], reanalyzed=True, communication=communication
            )
        )
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--reference", choices=["hf", "none"], required=True)
    parser.add_argument(
        "--communication", choices=["check", "unavailable"], required=True
    )
    options = parser.parse_args()
    result = reanalyze(
        options.directory,
        reference=options.reference,
        communication=options.communication,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
