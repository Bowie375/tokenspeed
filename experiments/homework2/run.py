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


"""Run isolated TP1/TP2 jobs with explicit configuration and nonzero failure status."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def worker_environment(root: Path, directory: Path, devices: list[str]) -> dict:
    """Prepare one child environment without changing the calling shell or other services."""
    bootstrap = directory / "bootstrap"
    bootstrap.mkdir()
    (bootstrap / "sitecustomize.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "from experiments.homework2.hooks import install\n"
        "install(Path(os.environ['HOMEWORK_TP_CAPTURE']))\n"
    )
    env = os.environ.copy()
    env.update(
        HOMEWORK_TP_CAPTURE=str(directory.resolve()),
        ASCEND_RT_VISIBLE_DEVICES=",".join(sorted(devices, key=int)),
        OMP_NUM_THREADS="8",
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
        GRPC_ENABLE_HTTP_PROXY="0",
    )
    env["PYTHONPATH"] = (
        os.pathsep.join(
            str(p)
            for p in [
                bootstrap.resolve(),
                root,
                root / "python",
                root / "tokenspeed-kernel/python",
                root / "tokenspeed-kernel-npu/python",
            ]
        )
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    return env


def execute(
    command: list[str], *, directory: Path, env: dict, timeout: int, root: Path
):
    """Limit the worker lifetime and terminate only its own process group on timeout/error."""
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
            code = process.wait(timeout=timeout)
        finally:
            # A parent may exit before its children. Address the entire owned
            # group even after wait() returns, and do not touch other services.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    if code:
        raise RuntimeError(
            f"worker exited with status {code}: {directory / 'worker.log'}"
        )


def build_jobs(fixtures: list[dict], *, steps: int, reuse_attempts: int) -> list[dict]:
    jobs = []
    for f in fixtures:
        for teacher in (True, False):
            jobs.append(
                dict(
                    name=f"{'teacher' if teacher else 'free'}_{f['name']}",
                    fixtures=[f],
                    steps=steps,
                    teacher=teacher,
                )
            )
    jobs.append(dict(name="batch", fixtures=fixtures[:2], steps=steps, teacher=True))
    jobs.extend(
        dict(name=f"repeat_{i}", fixtures=fixtures[:1], steps=steps, teacher=True)
        for i in range(reuse_attempts)
    )
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--devices", required=True, help="two distinct available physical chip IDs"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", choices=["correctness", "performance"], required=True)
    parser.add_argument("--reference", choices=["hf", "none"], required=True)
    parser.add_argument("--output-root", type=Path)
    options = parser.parse_args()
    devices = options.devices.split(",")
    if (
        len(devices) != 2
        or len(set(devices)) != 2
        or any(not d.isdigit() for d in devices)
    ):
        parser.error("devices must name two distinct physical chips")
    if options.task == "performance" and options.reference != "none":
        parser.error(
            "performance does not run numerical references; use --reference none"
        )
    root = Path(__file__).resolve().parents[2]
    checkpoint = options.model.resolve()
    config = json.loads(options.config.read_text())
    if options.task == "performance":
        from experiments.homework2.validation import validate_performance_resources

        validate_performance_resources(
            config, config["tp1_overrides"].get(checkpoint.name, {})
        )
    for name in (
        "steps",
        "reuse_attempts",
        "warmups",
        "repetitions",
        "concurrency",
        "context_length",
        "capacity",
        "chunked_prefill_size",
        "max_num_seqs",
        "timeout_seconds",
    ):
        if config[name] < 1:
            raise ValueError(f"{name} must be positive")
    if (
        len(config["lengths"]) < 2
        or min(config["lengths"]) < 1
        or len(set(config["lengths"])) != len(config["lengths"])
    ):
        raise ValueError("at least two distinct positive correctness lengths required")
    model = json.loads((checkpoint / "config.json").read_text())["text_config"]
    if any(
        model[k] % 2
        for k in (
            "num_attention_heads",
            "num_key_value_heads",
            "linear_num_key_heads",
            "linear_num_value_heads",
        )
    ):
        raise ValueError(
            "this model's head geometry does not support the two-rank baseline"
        )
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    )
    directory = (
        options.output_root or root / "outputs/homework2/runs"
    ) / f"{checkpoint.name}_{options.task}_{run_id}"
    directory.mkdir(parents=True)
    fixtures = [
        dict(
            name=f"raw_{n}",
            input_ids=[42 + index] * n,
            continuation_ids=[60 + i for i in range(config["steps"])],
        )
        for index, n in enumerate(config["lengths"])
    ]
    config = dict(config, task=options.task)
    (directory / "config.json").write_text(json.dumps(config, indent=2))
    (directory / "fixtures.json").write_text(json.dumps(fixtures, indent=2))
    (directory / "manifest.json").write_text(
        json.dumps(
            dict(
                checkpoint=str(checkpoint),
                checkpoint_revision=(
                    (checkpoint / ".cache/huggingface/download/config.json.metadata")
                    .read_text()
                    .splitlines()[0]
                    if (
                        checkpoint / ".cache/huggingface/download/config.json.metadata"
                    ).exists()
                    else None
                ),
                checkpoint_config_sha256=hashlib.sha256(
                    (checkpoint / "config.json").read_bytes()
                ).hexdigest(),
                source_hashes={
                    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted((root / "experiments/homework2").rglob("*.py"))
                },
                packages={
                    name: importlib.metadata.version(name)
                    for name in (
                        "torch",
                        "torch_npu",
                        "transformers",
                        "tokenspeed-scheduler",
                    )
                },
                model_config=model,
                devices=devices,
                rank_devices={"tp1": devices[:1], "tp2": sorted(devices, key=int)},
                source_revision=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=root, text=True
                ).strip(),
                worktree_status=subprocess.check_output(
                    ["git", "status", "--short"], cwd=root, text=True
                ),
                reference=options.reference,
                python=sys.version,
                measurement="in-process streaming; separate state capture and performance; shared chips may have other services",
            ),
            indent=2,
        )
    )
    print(directory, flush=True)
    try:
        for tp in (1, 2):
            rd = directory / f"tp{tp}"
            rd.mkdir()
            local = dict(config)
            if tp == 1:
                local.update(config["tp1_overrides"].get(checkpoint.name, {}))
            if max(config["lengths"]) + config["steps"] > local["context_length"]:
                raise ValueError("correctness fixture exceeds context capacity")
            if (
                options.task == "performance"
                and max(
                    w["input_tokens"] + w["output_tokens"] for w in config["workloads"]
                )
                > local["context_length"]
            ):
                raise ValueError("performance workload exceeds context capacity")
            if (
                options.task == "correctness"
                and max(config["lengths"]) > local["chunked_prefill_size"]
            ):
                raise ValueError(
                    "correctness capture requires each initial prompt to fit one prefill chunk"
                )
            launch = dict(
                model=str(checkpoint),
                world_size=tp,
                device="npu",
                dtype="bfloat16",
                language_model_only=True,
                attention_backend="mha",
                sampling_backend="greedy",
                enforce_eager=True,
                disable_prefill_graph=True,
                disable_pdl=True,
                disable_autotune=True,
                enable_prefix_caching=False,
                disable_kvstore=True,
                disable_overlap_schedule=True,
                retraction_snapshot_ratio=0,
                max_model_len=local["context_length"],
                max_num_seqs=local["max_num_seqs"],
                max_total_tokens=local["capacity"],
                max_prefill_tokens=local["chunked_prefill_size"],
                chunked_prefill_size=local["chunked_prefill_size"],
                seed=local["seed"],
                mamba_ssm_dtype="float32",
                stream_interval=1,
                log_level="warning",
            )
            from experiments.homework2.capacity import check_cache_capacity

            admission = check_cache_capacity(
                launch,
                requests=2 if options.task == "correctness" else 1,
                request_tokens=(
                    max(config["lengths"]) + config["steps"]
                    if options.task == "correctness"
                    else max(
                        w["input_tokens"] + w["output_tokens"]
                        for w in config["workloads"]
                    )
                ),
            )
            (rd / "cache_admission.json").write_text(json.dumps(admission, indent=2))
            for filename, value in [
                ("config.json", local),
                ("launch.json", launch),
                (
                    "jobs.json",
                    build_jobs(
                        fixtures,
                        steps=config["steps"],
                        reuse_attempts=config["reuse_attempts"],
                    ),
                ),
            ]:
                (rd / filename).write_text(json.dumps(value, indent=2))
            (rd / "active.json").write_text(json.dumps(dict(capture_enabled=False)))
            env = worker_environment(root, rd, devices[:tp])
            execute(
                [
                    sys.executable,
                    "-m",
                    "experiments.homework2.worker",
                    "--directory",
                    str(rd),
                ],
                directory=rd,
                env=env,
                timeout=config["timeout_seconds"],
                root=root,
            )
        if options.task == "correctness":
            if options.reference == "hf":
                from experiments.homework2.reference import generate_reference

                generate_reference(directory, dtype=config["reference_dtype"])
            from experiments.homework2.analyze import analyze

            result = analyze(
                directory, config, fixtures, model, reference=options.reference
            )
            print(
                json.dumps(
                    {k: v for k, v in result.items() if k != "checks"}, indent=2
                ),
                flush=True,
            )
            if not result["passed"]:
                raise RuntimeError(
                    "TP correctness criteria failed; inspect validation.json"
                )
            from experiments.homework2.communication import validate_model_communication

            communication = validate_model_communication(directory)
            if not communication["passed"]:
                raise RuntimeError(
                    "model communication criteria failed; inspect communication_validation.json"
                )
        else:
            from experiments.homework2.performance import analyze_performance

            result = analyze_performance(directory)
            if not result["passed"]:
                raise RuntimeError("invalid or incomplete performance evidence")
        (directory / "status.json").write_text(json.dumps(dict(completed=True)))
    except BaseException as error:
        (directory / "status.json").write_text(
            json.dumps(dict(completed=False, error=str(error)))
        )
        raise


if __name__ == "__main__":
    main()
