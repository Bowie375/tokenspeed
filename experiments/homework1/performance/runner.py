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


"""Prepare reproducible performance artifacts and launch an isolated native worker."""

import argparse
import dataclasses
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from experiments.homework1.evaluation.fixtures import (
    environment_manifest,
    load_fixtures,
    save_fixtures,
)
from experiments.homework1.performance.config import (
    PerformanceConfig,
    engine_options,
    workload_matrix,
)
from experiments.homework1.performance.results import finish


def find_fixture_source(root: Path, *, input_lengths: tuple[int, int]) -> Path:
    """Find the newest saved correctness run covering both requested raw lengths."""
    candidates = sorted(
        (root / "outputs/homework1/correctness/runs").glob("*/fixtures.json"),
        reverse=True,
    )
    for path in candidates:
        fixtures = load_fixtures(path)
        if set(input_lengths) <= {
            len(f.input_ids) for f in fixtures if f.kind == "raw"
        }:
            return path
    raise FileNotFoundError(
        f"Run the correctness notebook first with raw lengths {input_lengths}, or explicitly select saved fixtures"
    )


def prepare_run(
    root: Path,
    checkpoint: Path,
    fixtures_path: Path,
    config: PerformanceConfig,
    *,
    correctness_summary: Path | None,
) -> Path:
    """Freeze saved token IDs, checkpoint/config identity, EOS policy and source state."""
    fixtures = load_fixtures(fixtures_path)
    workload_matrix(config, fixtures)
    manifest = environment_manifest(
        root, checkpoint, seed=config.seed, settings=dataclasses.asdict(config)
    )
    source_manifest = fixtures_path.parent / "manifest.json"
    if source_manifest.exists():
        previous = json.loads(source_manifest.read_text())
        if previous.get("checkpoint_revision") != manifest["checkpoint_revision"]:
            raise ValueError(
                "saved fixture checkpoint revision differs from selected checkpoint"
            )
        if previous["checkpoint_file_hashes"] != manifest["checkpoint_file_hashes"]:
            raise ValueError(
                "saved fixture checkpoint/tokenizer hashes differ from selected checkpoint"
            )
    policy_path = fixtures_path.parent / "generation_policy.json"
    if not policy_path.exists():
        raise FileNotFoundError(
            "saved fixtures need their correctness generation_policy.json with explicit EOS IDs"
        )
    policy = json.loads(policy_path.read_text())
    if not policy["eos_token_ids"]:
        raise ValueError("saved EOS token ID set is empty")
    for package in (
        "triton",
        "triton-ascend",
        "tokenspeed-kernel",
        "tokenspeed-kernel-npu",
    ):
        try:
            manifest["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            manifest["packages"][package] = None
    manifest["profiler"] = (
        "torch_npu.profiler CPU+NPU, explicit one-request schedule, text export"
    )
    manifest.update(
        reference="native performance; numerical correctness linked separately",
        fixture_source=str(fixtures_path.resolve()),
        precision=dict(
            weights="BF16 with model-specific FP32 parameters",
            activations="BF16",
            recurrent_state="FP32",
        ),
    )
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    directory = root / "outputs/homework1/performance/runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "instrumentation").mkdir()
    (directory / "profiles").mkdir()
    save_fixtures(directory / "fixtures.json", fixtures)
    evidence = dict(status="unavailable", source=None)
    if correctness_summary is not None:
        evidence = dict(
            source=str(correctness_summary.resolve()),
            summary=json.loads(correctness_summary.read_text()),
        )
    for name, value in (
        ("manifest", manifest),
        ("config", dataclasses.asdict(config)),
        ("launch_config", engine_options(config, str(checkpoint.resolve()))),
        ("generation_policy", dict(policy, eos_policy=config.eos_policy)),
        ("correctness_status", evidence),
        ("active", dict(id="startup", enabled=False)),
    ):
        (directory / f"{name}.json").write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n"
        )
    return directory


def run_benchmark(
    root: Path, directory: Path, *, timeout_seconds: int, failure_policy: str
) -> dict:
    """Run the real Engine; preserve evidence on worker failure or timeout."""
    if failure_policy not in ("record", "raise"):
        raise ValueError("failure_policy must be record or raise")
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise ValueError("select exactly one free physical NPU before launching")
    if (directory / "worker.log").exists():
        raise FileExistsError("run already launched; prepare a new unique run")
    value = json.loads((directory / "config.json").read_text())
    value["input_lengths"], value["output_lengths"] = tuple(
        value["input_lengths"]
    ), tuple(value["output_lengths"])
    config = PerformanceConfig(**value)
    bootstrap = directory / "bootstrap"
    bootstrap.mkdir()
    (bootstrap / "sitecustomize.py").write_text(
        "import os\nfrom pathlib import Path\nfrom experiments.homework1.performance.hooks import install\ninstall(Path(os.environ['HOMEWORK_PERF_DIRECTORY']))\n"
    )
    env = os.environ.copy()
    env.update(
        HOMEWORK_PERF_DIRECTORY=str(directory.resolve()),
        OMP_NUM_THREADS="8",
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
        GRPC_ENABLE_HTTP_PROXY="0",
    )
    env["PYTHONPATH"] = (
        os.pathsep.join(
            str(p.resolve())
            for p in (
                bootstrap,
                root,
                root / "python",
                root / "tokenspeed-kernel/python",
                root / "tokenspeed-kernel-npu/python",
            )
        )
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    command = [
        sys.executable,
        "-m",
        "experiments.homework1.performance.worker",
        "--directory",
        str(directory.resolve()),
    ]
    (directory / "command.json").write_text(json.dumps(command, indent=2))
    status = dict(exit_code=None, timed_out=False, timeout_seconds=timeout_seconds)
    with (directory / "worker.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            status["exit_code"] = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            status["timed_out"] = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            status["exit_code"] = process.returncode
    (directory / "process_status.json").write_text(json.dumps(status, indent=2))
    result = finish(directory, config, worker_status=status)
    if failure_policy == "raise" and not result["all_cases_measured"]:
        raise RuntimeError(
            f"Incomplete performance matrix; inspect {directory / 'report.md'}"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        required=True,
        help="unique prepared run directory from prepare_run",
    )
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--failure-policy", choices=("record", "raise"), required=True)
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    print(
        json.dumps(
            run_benchmark(
                root,
                options.directory,
                timeout_seconds=options.timeout_seconds,
                failure_policy=options.failure_policy,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
