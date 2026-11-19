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


"""Evaluate saved TP trajectories, sharded state, and real request isolation."""

import dataclasses
import json
from pathlib import Path

import torch

from experiments.homework1.evaluation.metrics import Limits, compare_tensors
from experiments.homework2.validation import (
    compare_greedy,
    merge_conv_states,
    validate_rank_agreement,
    validate_steps,
)


def load_rows(directory: Path, job: str, fixture: str) -> list[dict]:
    return [
        torch.load(p, map_location="cpu", weights_only=True)
        for p in sorted((directory / job).glob(f"{fixture}_*.pt"))
    ]


def analyze(
    directory: Path,
    config: dict,
    fixtures: list[dict],
    model_config: dict,
    *,
    reference: str,
) -> dict:
    """Require actual evidence from TP1 and both TP2 ranks; absent captures fail."""
    # These are CPU FP64 evidence reductions, independent of model execution.
    torch.set_num_threads(8)
    checks = []

    def record(name, passed, **details):
        checks.append(dict(name=name, passed=bool(passed), **details))

    def tensor(name, actual, expected, limits):
        record(
            name,
            limits=dataclasses.asdict(limits),
            **compare_tensors(actual, expected, limits=limits),
        )

    limits = Limits(atol=0.6, rtol=0.03, relative_l2=0.06, rmse=0.18)
    state_limits = Limits(atol=0.02, rtol=0.05, relative_l2=0.08, rmse=0.02)
    lifecycle_limits = Limits(atol=0.03, rtol=0.01, relative_l2=0.02, rmse=0.01)
    # KV projections follow three TP-reduced decoder layers. Different TP sizes
    # therefore use the existing decoder-activation envelope, not the tighter
    # same-topology fresh/batched isolation envelope below.
    activation_limits = Limits(atol=0.08, rtol=0.05, relative_l2=0.06, rmse=0.03)
    ranks = [directory / "tp1/rank0", directory / "tp2/rank0", directory / "tp2/rank1"]
    for index, rd in enumerate(ranks):
        metadata = json.loads((rd / "model.json").read_text())
        record(
            f"rank_metadata_{index}",
            metadata["rank"] == (0 if index < 2 else 1)
            and metadata["world_size"] == (1 if index == 0 else 2),
        )
        record(
            f"npu_parameters_{index}",
            bool(metadata["parameters"])
            and all(
                p["device"].startswith("npu") for p in metadata["parameters"].values()
            ),
        )
    one = json.loads((ranks[0] / "model.json").read_text())["parameters"]
    for ri, rd in enumerate(ranks[1:]):
        two = json.loads((rd / "model.json").read_text())["parameters"]
        for name, p in one.items():
            axis = None
            if name.endswith(
                (
                    "in_proj_qkvzba.weight",
                    "in_proj_qkvz.weight",
                    "in_proj_ba.weight",
                    "conv1d.weight",
                    "qkv_proj.weight",
                    "gate_up_proj.weight",
                    "lm_head.weight",
                    "embed_tokens.weight",
                )
            ):
                axis = 0
            elif name.endswith(
                ("out_proj.weight", "o_proj.weight", "down_proj.weight")
            ):
                axis = 1
            if axis is not None:
                expected = list(p["shape"])
                expected[axis] //= 2
                record(
                    f"weight_shard_rank{ri}_{name}",
                    two[name]["shape"] == expected,
                    actual=two[name]["shape"],
                    expected=expected,
                )
    for fixture in fixtures:
        name = fixture["name"]
        teacher = [load_rows(rd, f"teacher_{name}", name) for rd in ranks]
        free = [load_rows(rd, f"free_{name}", name) for rd in ranks]
        record(
            f"free_peer_agreement_{name}", **validate_rank_agreement(free[1], free[2])
        )
        for ri, rows in enumerate(free):
            record(
                f"free_positions_{name}_{ri}",
                **validate_steps(
                    rows, input_ids=fixture["input_ids"], steps=config["steps"]
                ),
            )
            record(
                f"free_continuation_{name}_{ri}",
                all(
                    rows[i]["input_ids"] == [rows[i - 1]["sampled_token"]]
                    for i in range(1, len(rows))
                ),
            )
        for ri, rows in enumerate(teacher):
            record(
                f"positions_{name}_{ri}",
                **validate_steps(
                    rows, input_ids=fixture["input_ids"], steps=config["steps"]
                ),
            )
        for ri, rows in enumerate(teacher):
            record(
                f"teacher_inputs_{name}_{ri}",
                all(
                    rows[i]["input_ids"] == [fixture["continuation_ids"][i - 1]]
                    for i in range(1, len(rows))
                ),
            )
        for ri in (1, 2):
            outcome = compare_greedy(
                free[ri], free[0], allow_bf16_ties=config["allow_bf16_ties"]
            )
            record(
                f"free_tp{ri}_{name}",
                outcome["status"] in ("exact", "bf16_tie"),
                **outcome,
            )
        for step in range(config["steps"]):
            if any(len(rows) <= step for rows in teacher):
                record(f"missing_state_{name}_{step}", False)
                continue
            baseline, a, b = [rows[step] for rows in teacher]
            record(
                f"rank_tokens_{name}_{step}", a["sampled_token"] == b["sampled_token"]
            )
            tensor(
                f"rank_logits_{name}_{step}",
                a["logits"],
                b["logits"],
                Limits(atol=0, rtol=0, relative_l2=0, rmse=0),
            )
            tensor(f"tp_logits_{name}_{step}", a["logits"], baseline["logits"], limits)
            record(
                f"state_heads_{name}_{step}",
                a["ssm"].shape[0] * 2 == model_config["linear_num_value_heads"]
                and b["ssm"].shape == a["ssm"].shape,
            )
            for field in ("kv_key", "kv_value"):
                if field not in a or field not in b or field not in baseline:
                    record(f"missing_{field}_{name}_{step}", False)
                    continue
                record(
                    f"kv_heads_{field}_{name}_{step}",
                    a[field].shape[1] * 2 == model_config["num_key_value_heads"],
                )
                tensor(
                    f"tp_{field}_{name}_{step}",
                    torch.cat([a[field], b[field]], dim=1),
                    baseline[field],
                    activation_limits,
                )
            tensor(
                f"tp_ssm_{name}_{step}",
                torch.cat([a["ssm"], b["ssm"]], dim=0),
                baseline["ssm"],
                state_limits,
            )
            conv = merge_conv_states(
                [a["conv"], b["conv"]],
                key_channels=model_config["linear_num_key_heads"]
                * model_config["linear_key_head_dim"],
                value_channels=model_config["linear_num_value_heads"]
                * model_config["linear_value_head_dim"],
            )
            tensor(f"tp_conv_{name}_{step}", conv, baseline["conv"], lifecycle_limits)
        if reference == "hf":
            expected = torch.load(
                directory / "reference" / f"{name}.pt",
                map_location="cpu",
                weights_only=True,
            )
            outcome = compare_greedy(
                free[1], expected, allow_bf16_ties=config["allow_bf16_ties"]
            )
            record(f"hf_{name}", outcome["status"] in ("exact", "bf16_tie"), **outcome)
    for ri, rd in enumerate(ranks):
        batched = []
        used = set()
        reused = False
        for fixture in fixtures[:2]:
            baseline = load_rows(rd, f"teacher_{fixture['name']}", fixture["name"])
            batch = load_rows(rd, "batch", fixture["name"])
            batched.extend(batch)
            record(f"batch_steps_{ri}_{fixture['name']}", len(batch) == config["steps"])
            for step, (a, b) in enumerate(zip(batch, baseline, strict=False)):
                for field in ("logits", "ssm", "conv", "kv_key", "kv_value"):
                    tensor(
                        f"batch_{ri}_{fixture['name']}_{step}_{field}",
                        a[field],
                        b[field],
                        lifecycle_limits,
                    )
            used.update(r["block_out"] for r in baseline)
        record(f"batch_observed_rank{ri}", any(r["batch_size"] >= 2 for r in batched))
        for attempt in range(config["reuse_attempts"]):
            name = fixtures[0]["name"]
            actual = load_rows(rd, f"repeat_{attempt}", name)
            baseline = load_rows(rd, f"teacher_{name}", name)
            record(f"repeat_steps_{ri}_{attempt}", len(actual) == config["steps"])
            for step, (a, b) in enumerate(zip(actual, baseline, strict=False)):
                for field in ("logits", "ssm", "conv", "kv_key", "kv_value"):
                    tensor(
                        f"repeat_{ri}_{attempt}_{step}_{field}",
                        a[field],
                        b[field],
                        lifecycle_limits,
                    )
                reused |= a["block_out"] in used
            used.update(r["block_out"] for r in actual)
        record(f"physical_reuse_rank{ri}", reused)
    result = dict(
        passed=bool(checks) and all(r["passed"] for r in checks),
        total=len(checks),
        failed=[r for r in checks if not r["passed"]],
        checks=checks,
        strict_generation=all(r.get("status", "exact") == "exact" for r in checks),
        reference=reference,
    )
    (directory / "validation.json").write_text(json.dumps(result, indent=2))
    return result
