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


"""Fail-closed acceptance checks and measured summaries of native artifacts."""

import argparse
import json
import statistics
from pathlib import Path

import torch


def read_jsonl(path: Path):
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def compare_capture(reference: Path, actual: Path, *, exact_logits: bool):
    left, right = sorted(reference.glob("*.pt")), sorted(actual.glob("*.pt"))
    if not left or len(left) != len(right):
        raise AssertionError(f"incomplete model evidence: {reference}, {actual}")
    maximum = 0.0
    for source, destination in zip(left, right):
        a = torch.load(source, weights_only=True, map_location="cpu")
        b = torch.load(destination, weights_only=True, map_location="cpu")
        if not torch.equal(a["positions"], b["positions"]):
            raise AssertionError("restored computed context boundary differs")
        for field in ("conv", "ssm"):
            if not torch.equal(a["state"][field], b["state"][field]):
                raise AssertionError(f"{destination}: {field} state differs")
        if exact_logits and not torch.equal(a["logits"], b["logits"]):
            raise AssertionError(f"{destination}: matched-path logits differ")
        maximum = max(
            maximum, float((a["logits"].float() - b["logits"].float()).abs().max())
        )
    return dict(
        steps=len(left),
        maximum_logit_difference=maximum,
        logits_bitwise=exact_logits,
        states_bitwise=True,
    )


def analyze(directory: Path, *, task: str, require_eviction: bool):
    if not (directory / "complete.json").exists():
        raise AssertionError(f"worker did not complete: {directory}")
    launch = json.loads((directory / "launch.json").read_text())
    jobs = json.loads((directory / "jobs.json").read_text())
    records = json.loads((directory / "results.json").read_text())
    if len(jobs) != len(records):
        raise AssertionError("missing workload results")
    by_name = {r["name"]: r for r in records}
    for record in records:
        if record["action"] == "generate":
            response = record["response"]
            if (
                len(response["output_ids"]) != record["output_tokens"]
                or response["meta_info"]["completion_tokens"] != record["output_tokens"]
            ):
                raise AssertionError("incomplete native generation")
    ranks = []
    checks = []
    for rank in range(launch["world_size"]):
        rd = directory / f"rank{rank}"
        transfers = read_jsonl(rd / "transfers.jsonl")
        storage = read_jsonl(rd / "storage.jsonl")
        if not launch["disable_kvstore"]:
            if len(storage) != 1:
                raise AssertionError(
                    f"rank{rank}: missing or duplicate Host storage evidence"
                )
            allocation = storage[0]
            if (
                allocation["parents"] <= 0
                or allocation["parent_bytes"] <= 0
                or allocation["allocated_bytes"]
                != allocation["parents"] * allocation["parent_bytes"]
            ):
                raise AssertionError("invalid Host storage allocation")
        for item in records:
            gauge = item["ranks"][rank]["after"]
            if gauge["device_active"] != 0 or gauge["host_pinned"] != 0:
                raise AssertionError(f"{item['name']}: leaked active blocks/pins")
            if storage and not 0 <= gauge["host_free"] <= storage[0]["parents"]:
                raise AssertionError("invalid Host capacity")
            if not 0 <= gauge["device_free"] <= gauge["device_total"]:
                raise AssertionError("invalid Device capacity")
            if item["action"] in ("clear_device", "clear_all"):
                if gauge["device_free"] != gauge["device_total"]:
                    raise AssertionError("Device blocks were not reclaimed")
        eviction = max(x["ranks"][rank]["after"]["host_evictions"] for x in records)
        if require_eviction:
            if eviction <= 0:
                raise AssertionError("the small Host budget did not cause eviction")
            saw_eviction = False
            saw_miss = False
            for record in records:
                if (
                    saw_eviction
                    and record["action"] == "generate"
                    and record["response"]["meta_info"]["cached_tokens"] == 0
                ):
                    saw_miss = True
                saw_eviction |= record["ranks"][rank]["after"]["host_evictions"] > 0
            if not saw_miss:
                raise AssertionError("no cache miss/recompute after Host eviction")
        if not launch["disable_kvstore"]:
            if not any(t["direction"] == "d2h" and t["bytes"] > 0 for t in transfers):
                raise AssertionError(f"rank{rank}: no actual offload")
            if (
                task == "functional" or launch["debug_force_retraction_interval"]
            ) and not any(
                t["direction"] == "h2d" and t["bytes"] > 0 for t in transfers
            ):
                raise AssertionError(f"rank{rank}: no actual restore")
        if task == "functional":
            for index in range(2):
                cold, device, resume = (
                    by_name[f"{name}{index}"] for name in ("cold", "device", "resume")
                )
                if cold["response"]["output_ids"] != resume["response"]["output_ids"]:
                    raise AssertionError(
                        "restored continuation differs from cold reference"
                    )
                hit = resume["response"]["meta_info"]["cached_tokens"]
                if not require_eviction and hit != 128 + index:
                    raise AssertionError(
                        "natural checkpoint was not restored at the expected boundary"
                    )
                checks.append(
                    compare_capture(
                        rd / (f"device{index}" if hit else f"cold{index}"),
                        rd / f"resume{index}",
                        exact_logits=True,
                    )
                )
            for cycle in range(3):
                for index in range(3):
                    actual = by_name[f"cycle{cycle}_{index}"]
                    reference = by_name[f"reference{index}"]
                    if (
                        actual["response"]["output_ids"]
                        != reference["response"]["output_ids"]
                    ):
                        raise AssertionError("cross-session continuation corruption")
                    hit = actual["response"]["meta_info"]["cached_tokens"]
                    reference_name = (
                        f"device_reference{index}" if hit else f"reference{index}"
                    )
                    checks.append(
                        compare_capture(
                            rd / reference_name,
                            rd / f"cycle{cycle}_{index}",
                            exact_logits=True,
                        )
                    )
            # Actual source/destination block IDs recur across independent transfers.
            copied = [(g, d) for t in transfers for g, d, h in t["rows"]]
            if len(set(copied)) == len(copied):
                raise AssertionError("no evidence of Device block reuse")
        if task != "performance":
            for t in transfers:
                cache_bytes = t.get("cache_bytes", t["bytes"])
                if t.get("verified_bytes", -1) != cache_bytes:
                    raise AssertionError(
                        "cache fields did not pass byte-exact verification"
                    )
        if task == "retraction" and launch["debug_force_retraction_interval"]:
            for kind in ("_start_snapshot_store", "_start_restore"):
                if not any(t["kind"] == kind for t in transfers):
                    raise AssertionError(f"retraction did not exercise {kind}")
            final = records[-1]["ranks"][rank]["after"]
            initial = records[0]["ranks"][rank]["before"]
            expected_free = storage[0].get("snapshot_parents", initial["snapshot_free"])
            if final["snapshot_free"] != expected_free:
                raise AssertionError("request-private image blocks leaked")
        gauges = [r["ranks"][rank]["after"] for r in records]
        parent_bytes = storage[0]["parent_bytes"] if storage else 0
        parents = storage[0]["parents"] if storage else 0
        ranks.append(
            dict(
                rank=rank,
                host_allocated_bytes=storage[0]["allocated_bytes"] if storage else 0,
                host_peak_occupied_parent_bytes=max(
                    (parents - g["host_free"]) * parent_bytes for g in gauges
                ),
                host_evictions=eviction,
                d2h_bytes=sum(
                    t.get("cache_bytes", t["bytes"])
                    for t in transfers
                    if t["direction"] == "d2h"
                ),
                h2d_bytes=sum(
                    t.get("cache_bytes", t["bytes"])
                    for t in transfers
                    if t["direction"] == "h2d"
                ),
                slot_blob_serialized_bytes=sum(
                    t.get("slot_blob_bytes", 0) for t in transfers
                ),
                transfer_milliseconds=sum(t["milliseconds"] for t in transfers),
                restore_milliseconds=[
                    t["milliseconds"] for t in transfers if t["direction"] == "h2d"
                ],
                device_total_blocks=gauges[0]["device_total"],
                minimum_free_device_blocks=min(g["device_free"] for g in gauges),
            )
        )
    # Mirrored schedulers must agree on exact capacity/recovery progress.
    for record in records:
        values = [r["after"] for r in record["ranks"]]
        # Client responses precede final publication ACKs. Per-rank stats
        # sampled then need not have the same published-child count yet.
        keys = (
            values[0].keys()
            if record["action"] != "generate"
            else ("device_total", "device_active", "host_evictions", "snapshot_free")
        )
        if any(any(v[k] != values[0][k] for k in keys) for v in values[1:]):
            raise AssertionError("TP scheduler capacity/frontier disagreement")
        if record["action"] == "generate":
            positions = []
            for rank in range(launch["world_size"]):
                files = sorted(
                    (directory / f"rank{rank}" / record["name"]).glob("*.pt")
                )
                positions.append(
                    [
                        torch.load(f, weights_only=True)["positions"].tolist()
                        for f in files
                    ]
                )
            if any(p != positions[0] for p in positions[1:]):
                raise AssertionError("TP restored context positions differ")
    # Check the actual suffix against the scheduler's restored token frontier.
    boundaries = []
    job_specs = {job["name"]: job for job in jobs}
    for record in records:
        if record["action"] != "generate":
            continue
        cached = record["response"]["meta_info"]["cached_tokens"]
        if not 0 <= cached < record["input_tokens"]:
            raise AssertionError("invalid recovered context frontier")
        for rank in range(launch["world_size"]):
            files = sorted((directory / f"rank{rank}" / record["name"]).glob("*.pt"))
            if job_specs[record["name"]]["capture"]:
                if len(files) != record["output_tokens"]:
                    raise AssertionError("incomplete captured continuation")
                first = torch.load(files[0], weights_only=True)["positions"]
                expected = torch.arange(
                    cached, record["input_tokens"], dtype=first.dtype
                )
                if first.shape[-1] != len(expected) or not torch.all(first == expected):
                    raise AssertionError(
                        "captured suffix does not match restored context frontier"
                    )
        boundaries.append(
            dict(
                name=record["name"],
                recovered_tokens=cached,
                processed_prompt_range=[cached, record["input_tokens"]],
            )
        )
    summaries = {}
    for mode in ("high", "low"):
        rows = [r for r in records if r["name"].startswith(f"{mode}_")]
        if rows:
            seconds = sum(r["seconds"] for r in rows)
            cached = [r["response"]["meta_info"]["cached_tokens"] for r in rows]
            seen = set()
            recomputed = 0
            for row, frontier in zip(rows, cached):
                key = tuple(job_specs[row["name"]]["ids"])
                if key in seen:
                    recomputed += row["input_tokens"] - frontier
                seen.add(key)
            summaries[mode] = dict(
                requests=len(rows),
                hits=sum(c > 0 for c in cached),
                misses=sum(c == 0 for c in cached),
                cached_tokens=sum(cached),
                prefill_input_tokens=sum(
                    r["input_tokens"] - c for r, c in zip(rows, cached)
                ),
                recomputed_prompt_tokens=recomputed,
                latency_median_ms=statistics.median(r["seconds"] * 1000 for r in rows),
                output_tokens_per_second=sum(r["output_tokens"] for r in rows)
                / seconds,
            )
    result = dict(
        passed=True,
        ranks=ranks,
        correctness=checks,
        boundaries=boundaries,
        workloads=summaries,
    )
    (directory / "analysis.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--task", choices=["functional", "performance", "retraction"], required=True
    )
    parser.add_argument("--require-eviction", choices=["yes", "no"], required=True)
    options = parser.parse_args()
    try:
        result = analyze(
            options.directory,
            task=options.task,
            require_eviction=options.require_eviction == "yes",
        )
    except (AssertionError, OSError, KeyError, ValueError) as error:
        (options.directory / "analysis.json").write_text(
            json.dumps(dict(passed=False, error=str(error)), indent=2)
        )
        raise
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
