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


"""Native serving runs and independent HF CPU reference evaluation."""

import dataclasses
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import torch

from experiments.homework1.evaluation.fixtures import Fixture
from experiments.homework1.evaluation.metrics import Limits, RunArtifacts


def native_run(
    root: Path,
    checkpoint: Path,
    directory: Path,
    fixtures: list[Fixture],
    *,
    teacher_steps: int,
    free_steps: int,
    capacity: int,
    context_length: int,
    timeout_seconds: int,
    lifecycle: bool,
    reuse_attempts: int,
    failure_policy: str,
    seed: int,
    eos_token_ids: list[int],
):
    """Run real Engine jobs with cached decode in an owned subprocess group.

    Captures unmodified pre-sampling logits. Teacher forcing overrides only sampled
    token IDs, so the production scheduler and caches execute every decode step.
    Lifecycle jobs compare batched requests with independent runs and a fresh rerun.
    Returns the native capture directory. Existing results cannot be overwritten.
    """
    if failure_policy not in ("raise", "record"):
        raise ValueError("failure_policy must be raise or record")
    device = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if not device.isdigit():
        raise ValueError(
            "ASCEND_RT_VISIBLE_DEVICES must explicitly select one free chip"
        )
    if not fixtures or teacher_steps < 1 or free_steps < 1:
        raise ValueError("fixtures and positive step counts are required")
    if any(len(f.continuation_ids) < teacher_steps for f in fixtures):
        raise ValueError("saved continuation is shorter than requested teacher forcing")
    directory.mkdir(parents=True, exist_ok=False)
    jobs = []
    for fixture in fixtures:
        jobs.append(
            dict(
                name=f"teacher_{fixture.name}",
                fixtures=[dataclasses.asdict(fixture)],
                teacher=True,
                steps=teacher_steps,
            )
        )
        jobs.append(
            dict(
                name=f"free_{fixture.name}",
                fixtures=[dataclasses.asdict(fixture)],
                teacher=False,
                steps=free_steps,
            )
        )
    if reuse_attempts < 1:
        raise ValueError("reuse_attempts must be positive")
    if lifecycle:
        if len(fixtures) < 2 or fixtures[0].input_ids == fixtures[1].input_ids:
            raise ValueError("lifecycle requires two distinct saved inputs")
        jobs.extend(
            [
                dict(
                    name="interleaved",
                    fixtures=[dataclasses.asdict(f) for f in fixtures[:2]],
                    teacher=True,
                    steps=teacher_steps,
                ),
                dict(
                    name="fresh_repeat",
                    fixtures=[dataclasses.asdict(fixtures[0])],
                    teacher=True,
                    steps=teacher_steps,
                ),
            ]
        )
    if lifecycle:
        for attempt in range(1, reuse_attempts):
            jobs.append(
                dict(
                    name=f"fresh_repeat_{attempt}",
                    fixtures=[dataclasses.asdict(fixtures[0])],
                    teacher=True,
                    steps=teacher_steps,
                )
            )
    longest = max(fixtures, key=lambda fixture: len(fixture.input_ids)).name
    longest_jobs = [
        job
        for job in jobs
        if len(job["fixtures"]) == 1
        and job["fixtures"][0]["name"] == longest
        and not job["name"].startswith("fresh_repeat")
    ]
    jobs = [job for job in jobs if job not in longest_jobs] + longest_jobs
    (directory / "jobs.json").write_text(json.dumps(jobs, ensure_ascii=False, indent=2))
    (directory / "active.json").write_text(
        json.dumps(dict(jobs[0], capture_enabled=False))
    )
    bootstrap = directory / "bootstrap"
    bootstrap.mkdir()
    (bootstrap / "sitecustomize.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "from experiments.homework1.evaluation.capture import install\n"
        "install(Path(os.environ['HOMEWORK_EVAL_CAPTURE']))\n"
    )
    env = os.environ.copy()
    env["HOMEWORK_EVAL_CAPTURE"] = str(directory.resolve())
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
    env.update(
        OMP_NUM_THREADS="8",
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
        GRPC_ENABLE_HTTP_PROXY="0",
    )
    command = [
        sys.executable,
        "-m",
        "experiments.homework1.evaluation.worker",
        "--directory",
        str(directory.resolve()),
        "--checkpoint",
        str(checkpoint.resolve()),
        "--capacity",
        str(capacity),
        "--context-length",
        str(context_length),
        "--seed",
        str(seed),
        "--eos-token-ids",
        json.dumps(eos_token_ids),
    ]
    (directory / "command.json").write_text(json.dumps(command))
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
            code = process.wait(timeout=timeout_seconds)
            active = json.loads((directory / "active.json").read_text())
            (directory / "status.json").write_text(
                json.dumps(
                    dict(
                        exit_code=code,
                        last_job=active["name"],
                        completed=code == 0,
                        failure_policy=failure_policy,
                    )
                )
            )
            if code and failure_policy == "record":
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            if code and failure_policy == "raise":
                raise RuntimeError(
                    f'Native worker exited {code}; see {directory / "worker.log"}'
                )
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    return directory


def load_steps(directory: Path, job: str, fixture: Fixture) -> list[dict]:
    """Load native captures in chronological order; no captures is an error."""
    files = sorted((directory / job).glob(f"{fixture.name}_*.pt"))
    if not files:
        raise RuntimeError(f"No captures for {job}/{fixture.name}")
    rows = [torch.load(path, weights_only=True) for path in files]
    if [row["step"] for row in rows] != list(range(len(rows))):
        raise RuntimeError("Missing or duplicated native decode step")
    return rows


class Reference:
    """An independent eager HF CPU FP32 model, with a fresh cache for each case."""

    def __init__(self, checkpoint: Path, *, threads: int, dtype: torch.dtype):
        from transformers import Qwen3_5ForConditionalGeneration

        torch.set_num_threads(threads)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(checkpoint),
            local_files_only=True,
            dtype=dtype,
            attn_implementation="eager",
        ).eval()
        self.config = self.model.config.text_config

    @torch.inference_mode()
    def teacher(
        self,
        fixture: Fixture,
        *,
        steps: int,
        position_overrides: list[torch.Tensor] | None,
    ) -> list[dict]:
        cache = None
        ids = fixture.input_ids
        result = []
        offset = 0
        for step in range(steps):
            positions = (
                torch.arange(offset, offset + len(ids))
                .reshape(1, 1, -1)
                .expand(3, 1, -1)
                if position_overrides is None
                else position_overrides[step].unsqueeze(1)
            )
            output = self.model(
                input_ids=torch.tensor([ids]),
                position_ids=positions,
                attention_mask=torch.ones(1, offset + len(ids), dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = output.past_key_values
            logits = output.logits[0, -1].detach().clone()
            result.append(
                dict(
                    logits=logits,
                    input_ids=list(ids),
                    positions=positions[:, 0].clone(),
                )
            )
            offset += len(ids)
            ids = [fixture.continuation_ids[step]]
        return result

    @torch.inference_mode()
    def greedy(
        self, fixture: Fixture, *, steps: int, eos_token_ids: list[int]
    ) -> list[dict]:
        cache = None
        ids = fixture.input_ids
        result = []
        eos = set(eos_token_ids)
        for step in range(steps):
            output = self.model(
                input_ids=torch.tensor([ids]),
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = output.past_key_values
            logits = output.logits[0, -1].detach().clone()
            token = int(logits.argmax())
            top = logits.topk(2).values
            result.append(
                dict(token=token, margin=float(top[0] - top[1]), logits=logits)
            )
            if token in eos:
                break
            ids = [token]
        return result


def compare_teacher(
    run: RunArtifacts,
    native: Path,
    reference: Reference,
    fixtures: list[Fixture],
    *,
    steps: int,
    limits: Limits,
):
    """Compare raw/centered logits with identical IDs and captured positions.

    Independent exact checks also validate positions against canonical text
    positions, preventing aligned numerical references from hiding a bad offset.
    """
    for fixture in fixtures:
        try:
            actual = load_steps(native, f"teacher_{fixture.name}", fixture)
        except RuntimeError as error:
            run.record_failure(
                f"teacher_{fixture.name}",
                str(error),
                details=dict(stage="teacher_unavailable"),
            )
            continue
        expected = reference.teacher(
            fixture,
            steps=len(actual),
            position_overrides=[row["positions"] for row in actual],
        )
        if len(actual) != steps:
            run.record_failure(
                f"teacher_steps_{fixture.name}",
                f"Expected {steps} native steps, got {len(actual)}",
                details=dict(stage="teacher_unavailable"),
            )
        for step, (a, b) in enumerate(
            zip(actual, expected[: len(actual)], strict=True)
        ):
            if step > 0 and not a["cached_decode"]:
                raise RuntimeError(
                    "Teacher forcing did not execute native cached decode"
                )
            if a["input_ids"] != b["input_ids"]:
                raise RuntimeError(
                    f"Teacher-forcing inputs differ at {fixture.name}/{step}"
                )
            run.record(
                f"positions_{fixture.name}_{step}",
                a["positions"],
                torch.arange(
                    0 if step == 0 else len(fixture.input_ids) + step - 1,
                    (
                        len(fixture.input_ids)
                        if step == 0
                        else len(fixture.input_ids) + step
                    ),
                )
                .reshape(1, -1)
                .expand(3, -1),
                limits=Limits(0, 0, 0, 0),
                details=dict(stage="input_position_alignment", step=step),
            )
            x = a["logits"].float()
            y = b["logits"].float()
            run.record(
                f"raw_logits_{fixture.name}_{step}",
                x,
                y,
                limits=limits,
                details=dict(stage="raw_model_logits", step=step, fixture=fixture.name),
            )
            run.record(
                f"logits_{fixture.name}_{step}",
                x - x.mean(),
                y - y.mean(),
                limits=limits,
                details=dict(
                    stage="prefill" if step == 0 else "cached_teacher",
                    step=step,
                    input_ids=a["input_ids"],
                    native_mode=a["mode"],
                    centering="subtract vocabulary mean",
                    native_top1=int(x.argmax()),
                    reference_top1=int(y.argmax()),
                    reference_margin=float(y.topk(2).values.diff().abs()[0]),
                ),
            )


def compare_generation(
    run: RunArtifacts,
    native: Path,
    reference: Reference,
    fixtures: list[Fixture],
    *,
    steps: int,
    tokenizer,
    eos_token_ids: list[int],
):
    """Report first greedy divergence and aligned-prefix margins, separately from numerics."""
    results = []
    for fixture in fixtures:
        try:
            actual = load_steps(native, f"free_{fixture.name}", fixture)
        except RuntimeError as error:
            run.record_failure(
                f"free_{fixture.name}",
                str(error),
                details=dict(stage="generation_unavailable"),
            )
            actual = []
        expected = reference.greedy(fixture, steps=steps, eos_token_ids=eos_token_ids)
        a = [row["sampled_token"] for row in actual]
        b = [row["token"] for row in expected]
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if first is None and len(a) != len(b):
            first = min(len(a), len(b))
        margin = (
            expected[first]["margin"]
            if first is not None and first < len(expected)
            else None
        )
        results.append(
            dict(
                fixture=fixture.name,
                status="evaluated" if actual else "unavailable",
                native_ids=a,
                reference_ids=b,
                native_text=tokenizer.decode(a),
                reference_text=tokenizer.decode(b),
                first_difference=first,
                reference_margin_at_difference=margin,
                exact_match=a == b,
                ignore_eos=False,
                eos_token_ids=eos_token_ids,
            )
        )
        torch.save(
            dict(native=actual, reference=expected),
            run.path / "tensors" / f"generation_{fixture.name}.pt",
        )
    run.write_json("generation.json", results)
    return results


def compare_lifecycle(
    run: RunArtifacts, native: Path, fixtures: list[Fixture], *, limits: Limits
):
    """Compare real batched requests and rerun against independent serving jobs.

    Saves physical block IDs and explicitly reports whether reuse was observed;
    repeat execution alone is not evidence of physical state-page reuse.
    """
    observations = []
    jobs = json.loads((native / "jobs.json").read_text())
    repeats = [job["name"] for job in jobs if job["name"].startswith("fresh_repeat")]
    for job, subset in [
        ("interleaved", fixtures[:2]),
        *[(name, fixtures[:1]) for name in repeats],
    ]:
        for fixture in subset:
            try:
                independent = load_steps(native, f"teacher_{fixture.name}", fixture)
                batched = load_steps(native, job, fixture)
            except RuntimeError as error:
                run.record_failure(
                    f"{job}_{fixture.name}_unavailable",
                    str(error),
                    details=dict(stage="lifecycle_unavailable"),
                )
                continue
            if len(batched) != len(independent):
                raise RuntimeError("Lifecycle step counts differ")
            for step, (a, b) in enumerate(zip(batched, independent, strict=True)):
                if a["input_ids"] != b["input_ids"]:
                    raise RuntimeError("Lifecycle inputs differ")
                for field in ("logits", "layer_output", "conv", "ssm"):
                    run.record(
                        f"{job}_{fixture.name}_{field}_{step}",
                        a[field],
                        b[field],
                        limits=limits,
                        details=dict(
                            stage="state_lifecycle",
                            job=job,
                            step=step,
                            field=field,
                            independent_block=b["block_out"],
                            observed_block=a["block_out"],
                        ),
                    )
                observations.append(
                    dict(
                        job=job,
                        fixture=fixture.name,
                        step=step,
                        batch_number=a["batch_number"],
                        batch_size=a["batch_size"],
                        slot=a["slot"],
                        block_in=a["block_in"],
                        block_out=a["block_out"],
                        baseline_block=b["block_out"],
                    )
                )
    first_repeat = next(
        i for i, job in enumerate(jobs) if job["name"].startswith("fresh_repeat")
    )
    baseline_blocks = {
        torch.load(path, weights_only=True)["block_out"]
        for job in jobs[:first_repeat]
        if not job["name"].startswith("fresh_repeat")
        for path in (native / job["name"]).glob("*.pt")
    }
    repeat = [
        torch.load(path, weights_only=True)
        for job in repeats
        for path in (native / job).glob("*.pt")
    ]
    reused = any(row["block_out"] in baseline_blocks for row in repeat)
    result = dict(
        observations=observations,
        batched_execution_observed=any(
            row["batch_size"] > 1 for row in observations if row["job"] == "interleaved"
        ),
        physical_reuse_observed=reused,
        physical_reuse_status="observed" if reused else "unverified",
        prefix_caching=False,
    )
    run.write_json("lifecycle.json", result)
    return result


def rounding_baseline(
    run: RunArtifacts,
    fp32: Reference,
    bf16: Reference,
    fixtures: list[Fixture],
    *,
    steps: int,
    limits: Limits,
):
    """Measure independent HF BF16-vs-FP32 cached logits before evaluating serving.

    Acceptance limits are caller supplied in advance; this function never changes
    them. Full BF16 computation is compared, not just rounding the final logits.
    """
    for fixture in fixtures:
        a = bf16.teacher(fixture, steps=steps, position_overrides=None)
        b = fp32.teacher(fixture, steps=steps, position_overrides=None)
        for step, (x, y) in enumerate(zip(a, b, strict=True)):
            actual = x["logits"].float()
            expected = y["logits"].float()
            run.record(
                f"rounding_{fixture.name}_{step}",
                actual - actual.mean(),
                expected - expected.mean(),
                limits=limits,
                details=dict(
                    stage="reference_rounding_baseline",
                    reference="HF eager CPU BF16 versus FP32",
                    step=step,
                ),
            )
