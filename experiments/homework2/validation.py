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


"""Explicit TP numerical acceptance and reconstruction of rank-local state."""

import math

import torch


def compare_greedy(
    actual: list[dict], reference: list[dict], *, allow_bf16_ties: bool
) -> dict:
    """Check equal token trajectories or classify their first aligned divergence.

    A BF16 tie is distinct from exact agreement. After the first different token,
    the remaining logits have different prefixes and are not compared.
    """
    if not actual or len(actual) != len(reference):
        return dict(status="failed", first_difference=None, reason="missing steps")
    if any(not bool(torch.isfinite(row["logits"]).all()) for row in actual + reference):
        return dict(status="failed", first_difference=None, reason="nonfinite logits")
    for step, (a, b) in enumerate(zip(actual, reference, strict=True)):
        candidate, chosen = a["sampled_token"], b["sampled_token"]
        if candidate == chosen:
            continue
        logits = b["logits"].float()
        peak = float(logits.max())
        gap = peak - float(logits[candidate])
        ulp = (
            math.ldexp(1.0, max(math.floor(math.log2(abs(peak))), -126) - 7)
            if peak
            else 2.0**-133
        )
        top = logits.topk(min(5, logits.numel())).indices.tolist()
        tied = allow_bf16_ties and candidate in top and 0 <= gap <= ulp
        return dict(
            status="bf16_tie" if tied else "failed",
            first_difference=step,
            actual=candidate,
            reference=chosen,
            reference_gap=gap,
            bf16_ulp=ulp,
            reference_top_five=top,
        )
    return dict(status="exact", first_difference=None)


def validate_steps(rows: list[dict], *, input_ids: list[int], steps: int) -> dict:
    """Require one real prefill followed by cached single-token decode at canonical positions."""
    errors = []
    if len(rows) != steps:
        errors.append(f"expected {steps} steps, observed {len(rows)}")
    for index, row in enumerate(rows):
        expected = (
            torch.arange(len(input_ids))
            if index == 0
            else torch.tensor([len(input_ids) + index - 1])
        )
        positions = row["positions"]
        canonical = (
            expected.expand_as(positions)
            if positions.shape[-1] == expected.numel()
            else expected
        )
        if positions.shape != canonical.shape or not torch.equal(positions, canonical):
            errors.append(f"incorrect positions at step {index}")
        if row["step"] != index or row["cached_decode"] != (index > 0):
            errors.append(f"incorrect forward mode/step at {index}")
        if (index == 0 and row["input_ids"] != input_ids) or (
            index > 0 and len(row["input_ids"]) != 1
        ):
            errors.append(f"incorrect input extent at step {index}")
    return dict(passed=not errors, errors=errors)


def merge_conv_states(
    shards: list[torch.Tensor], *, key_channels: int, value_channels: int
) -> torch.Tensor:
    """Undo rank-local [Q|K|V] packing, preserving history dimensions and global head order."""
    world_size = len(shards)
    if not world_size or key_channels % world_size or value_channels % world_size:
        raise ValueError("convolution channels must divide evenly across TP ranks")
    widths = [
        key_channels // world_size,
        key_channels // world_size,
        value_channels // world_size,
    ]
    parts = [shard.split(widths, dim=0) for shard in shards]
    return torch.cat(
        [
            torch.cat([rank[component] for rank in parts], dim=0)
            for component in range(3)
        ],
        dim=0,
    )


def validate_rank_agreement(left: list[dict], right: list[dict]) -> dict:
    """Require exact agreement between peer ranks in one native free-running deployment."""
    errors = []
    if not left or len(left) != len(right):
        errors.append("missing peer-rank steps")
    for step, (a, b) in enumerate(zip(left, right, strict=False)):
        if a["sampled_token"] != b["sampled_token"]:
            errors.append(f"different peer tokens at {step}")
        if a["input_ids"] != b["input_ids"] or not torch.equal(
            a["positions"], b["positions"]
        ):
            errors.append(f"different peer inputs/positions at {step}")
    return dict(passed=not errors, errors=errors)


def validate_performance_resources(config: dict, override: dict):
    """Refuse unequal TP resource budgets rather than call them comparable timings."""
    for key in ("context_length", "capacity", "chunked_prefill_size", "max_num_seqs"):
        if key in override and override[key] != config[key]:
            raise ValueError(
                f"TP performance requires equal {key}; use an equal-budget config"
            )


def validate_performance(records: list[dict], config: dict) -> dict:
    """Reject missing matrix entries, samples or required client timings."""
    errors = []
    expected = {
        (w["input_tokens"], w["output_tokens"], parallel)
        for w in config["workloads"]
        for parallel in (False, True)
    }
    seen = []
    for record in records:
        key = (
            record["case"]["input_tokens"],
            record["case"]["output_tokens"],
            record["parallel"],
        )
        seen.append(key)
        trials = record["trials"]
        if sorted(t["repetition"] for t in trials) != list(
            range(config["repetitions"])
        ):
            errors.append(f"incomplete repetitions: {key}")
        for trial in trials:
            requests = trial["requests"]
            if len(requests) != config["concurrency"]:
                errors.append(f"incomplete request group: {key}")
            if trial["wall_seconds"] <= 0 or not math.isfinite(
                trial["throughput_tokens_per_second"]
            ):
                errors.append(f"invalid throughput: {key}")
            for request in requests:
                timing = [
                    request.get(k)
                    for k in ("client_ttft_ms", "client_tpot_ms", "e2e_ms")
                ]
                if (
                    not request["valid"]
                    or request["actual_output_tokens"] != key[1]
                    or any(v is None or not math.isfinite(v) or v < 0 for v in timing)
                ):
                    errors.append(f"invalid request: {key}")
    if set(seen) != expected or len(seen) != len(expected):
        errors.append("missing, duplicated or unexpected workload/mode")
    return dict(passed=not errors, errors=errors)
