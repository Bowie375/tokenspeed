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


"""Run existing operator suites and summarize persistent evaluation evidence."""

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from experiments.homework1.evaluation.metrics import RunArtifacts

OPERATOR_TESTS = [
    "tokenspeed-kernel-npu/test/test_gdn.py",
    "tokenspeed-kernel-npu/test/test_mha.py",
    "tokenspeed-kernel-npu/test/test_attention_output_gate.py",
    "tokenspeed-kernel-npu/test/test_causal_conv1d.py",
    "tokenspeed-kernel-npu/test/test_gdn_preparation.py",
    "tokenspeed-kernel-npu/test/test_gdn_output_norm.py",
    "tokenspeed-kernel-npu/test/test_gdn_state_helpers.py",
    "test/runtime/models/test_gemma_rmsnorm_npu.py",
]


def operator_tests(root: Path, run: RunArtifacts, *, timeout_seconds: int) -> dict:
    """Execute the actual operator tests; persist stdout, JUnit, skips and failures."""
    xml = run.path / "operators.xml"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *OPERATOR_TESTS,
        f"--junitxml={xml.resolve()}",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        os.pathsep.join(
            str(p)
            for p in [
                root,
                root / "python",
                root / "tokenspeed-kernel/python",
                root / "tokenspeed-kernel-npu/python",
            ]
        )
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    with (run.path / "operators.log").open("w") as log:
        result = subprocess.run(
            command,
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
    suites = ET.parse(xml).getroot().iter("testsuite")
    counts = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
    for suite in suites:
        for key in counts:
            counts[key] += int(suite.attrib.get(key, 0))
    counts["passed"] = (
        counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    )
    summary = dict(
        command=command,
        exit_code=result.returncode,
        **counts,
        all_passed=result.returncode == 0
        and counts["passed"] > 0
        and counts["skipped"] == 0,
    )
    run.write_json("operators.json", summary)
    return summary


def finish(
    run: RunArtifacts, *, operators: dict, generation: list[dict], lifecycle: dict
) -> dict:
    """Write final counts without concealing failed comparisons or unverified reuse."""
    summary = dict(
        numerics=run.summary(),
        unavailable_checks=run.failures,
        operators=operators,
        generation_exact_matches=sum(row["exact_match"] for row in generation),
        generation_cases=len(generation),
        physical_reuse_status=lifecycle["physical_reuse_status"],
        batched_execution_observed=lifecycle["batched_execution_observed"],
    )
    summary["all_required_passed"] = (
        summary["numerics"]["all_passed"]
        and operators["all_passed"]
        and lifecycle["physical_reuse_observed"]
        and lifecycle["batched_execution_observed"]
    )
    run.write_json("summary.json", summary)
    lines = [
        "# Step 11 results",
        "",
        f"Run: `{run.path.name}`",
        "",
        "| Comparison | Max absolute | RMSE | Relative L2 | Passed |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in run.rows:
        lines.append(
            f"| {row['name']} | {row['max_abs']} | {row['rmse']} | {row['relative_l2']} | {row['passed']} |"
        )
    for failure in run.failures:
        lines.append(
            f"| {failure['name']} | unavailable | unavailable | unavailable | FAIL: {failure['reason']} |"
        )
    lines.extend(
        [
            "",
            f"Operators: {operators['passed']} passed; {operators['skipped']} skipped; {operators['failures']} failures; {operators['errors']} errors.",
            f"Free generation: {summary['generation_exact_matches']}/{len(generation)} exact matches. See generation.json for first differences and margins.",
            f"Physical state-block reuse: {lifecycle['physical_reuse_status']}.",
            "",
            "Graphs, prefix caching, transfer/retraction, and performance are outside this eager correctness run.",
        ]
    )
    (run.path / "report.md").write_text("\n".join(lines) + "\n")
    return summary
