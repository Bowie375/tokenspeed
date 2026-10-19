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


"""Join real forward evidence to client samples and persist honest summaries."""

import csv
import json
from pathlib import Path

from experiments.homework1.performance.config import PerformanceConfig
from experiments.homework1.performance.measurements import summarize_samples
from experiments.homework1.performance.profiling import analyze_profiles


def collect_samples(directory: Path) -> list[dict]:
    """Require matching native forward records, exact token counts and measured scopes."""
    requests = directory / "requests.jsonl"
    samples = (
        [json.loads(line) for line in requests.read_text().splitlines() if line.strip()]
        if requests.exists()
        else []
    )
    for row in samples:
        path = directory / "instrumentation" / f"{row['id']}.json"
        if not path.exists():
            row.update(
                valid=False, instrumentation_error="missing real forward capture"
            )
            continue
        capture = json.loads(path.read_text())
        row["instrumentation"] = capture
        events = capture["events"]
        prefills = [e for e in events if e["kind"] == "prefill"]
        decodes = [e for e in events if e["kind"] == "decode"]
        expected_decode = max(0, row["actual_output_tokens"] - 1)
        errors = []
        if len(prefills) != 1 or prefills[0]["input_tokens"] != row["input_tokens"]:
            errors.append("expected one complete, uncached prefill")
        if len(decodes) != expected_decode or any(
            e["input_tokens"] != 1 for e in decodes
        ):
            errors.append(
                f"expected {expected_decode} one-token decode forwards, got {len(decodes)}"
            )
        if row["phase"] == "forward":
            if any(e["duration_ms"] is None or e["duration_ms"] <= 0 for e in events):
                errors.append("missing synchronized forward duration")
            else:
                row["prefill_forward_ms"] = sum(e["duration_ms"] for e in prefills)
                row["decode_forward_step_ms"] = [e["duration_ms"] for e in decodes]
                row["decode_forward_ms_per_step"] = (
                    sum(row["decode_forward_step_ms"]) / len(decodes)
                    if decodes
                    else None
                )
        memory = capture["memory_after_bytes"]
        for name, source in (
            ("peak_allocated_mib", "max_memory_allocated"),
            ("peak_reserved_mib", "max_memory_reserved"),
        ):
            value = memory.get(source)
            row[name] = (
                value / 2**20
                if value is not None and capture["peak_reset_supported"]
                else None
            )
        if errors:
            row.update(valid=False, instrumentation_error="; ".join(errors))
    return samples


def write_csv(path: Path, rows: list[dict]):
    """Save every row; encode structured columns as JSON without losing detail."""
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                key: (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                )
                for key, value in row.items()
            }
            for row in rows
        )


def finish(directory: Path, config: PerformanceConfig, *, worker_status: dict) -> dict:
    """Persist raw samples, medians/min/max, trace evidence and limitations."""
    samples = collect_samples(directory)
    summaries = summarize_samples(samples)
    profiles = analyze_profiles(directory / "profiles")
    for name, value in (
        ("samples", samples),
        ("statistics", summaries),
        ("profile_summary", profiles),
    ):
        (directory / f"{name}.json").write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
    write_csv(directory / "samples.csv", samples)
    flattened = []
    for row in summaries:
        flat = {}
        for key, value in row.items():
            if isinstance(value, dict):
                flat.update({f"{key}_{stat}": number for stat, number in value.items()})
            else:
                flat[key] = value
        flattened.append(flat)
    write_csv(directory / "statistics.csv", flattened)
    complete = len(summaries) == 8 and all(
        s["measured_repetitions"] == config.repetitions and s["failed_repetitions"] == 0
        for s in summaries
    )
    profiles_observed = {
        str(p["path"]).split("/profiles/")[-1].split("/")[0]
        for p in profiles
        if p["status"] == "observed"
    }
    expected_profiles = {
        f"in{n}_out{min(config.output_lengths)}" for n in config.input_lengths
    }
    evidence = json.loads((directory / "correctness_status.json").read_text())
    result = dict(
        worker=worker_status,
        all_cases_measured=bool(complete and worker_status["exit_code"] == 0),
        ten_repetition_baseline=bool(
            complete and config.repetitions >= 10 and worker_status["exit_code"] == 0
        ),
        configured_repetitions=config.repetitions,
        profile_requested=config.profile,
        short_and_long_device_profiles_observed=(
            expected_profiles <= profiles_observed if config.profile else False
        ),
        correctness=evidence,
        failed_requests=[s["id"] for s in samples if not s["valid"]],
    )
    result["complete_default_evidence"] = bool(
        result["ten_repetition_baseline"]
        and result["short_and_long_device_profiles_observed"]
    )
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lookup = {(s["case"], s["phase"]): s for s in summaries}

    def value(row, field):
        item = row.get(field)
        return (
            "unavailable"
            if item is None
            else f"{item['median']:.3f} [{item['min']:.3f}, {item['max']:.3f}]"
        )

    lines = [
        "# Step 12 performance evidence",
        "",
        f"Run: `{directory.name}`.",
        "",
        f"Complete measured matrix: **{result['all_cases_measured']}**. At least ten valid repetitions per case/pass: **{result['ten_repetition_baseline']}**.",
        f"EOS policy: `{config.eos_policy}`; execution: `{config.execution_mode}`, BF16 weights/activations, FP32 recurrent state, one request at a time.",
        "",
        "Values are median [minimum, maximum], in milliseconds except HBM (MiB). Forward times and client latency come from separate passes.",
        "",
        "| Case | Actual output counts | Client TTFT ms | Client TPOT ms | E2E ms | Prefill forward ms | Decode forward ms/step | Peak allocated MiB | Valid reps latency/forward |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for n in config.input_lengths:
        for out in config.output_lengths:
            name = f"in{n}_out{out}"
            latency = lookup.get((name, "latency"), {})
            forward = lookup.get((name, "forward"), {})
            lines.append(
                f"| {name} | {latency.get('actual_output_tokens',[])} | {value(latency,'client_ttft_ms')} | {value(latency,'client_tpot_ms')} | {value(latency,'e2e_ms')} | {value(forward,'prefill_forward_ms')} | {value(forward,'decode_forward_ms_per_step')} | {value(forward,'peak_allocated_mib')} | {latency.get('measured_repetitions',0)}/{forward.get('measured_repetitions',0)} |"
            )
    lines += [
        "",
        "## Measurement scopes",
        "",
        "Client latency uses the in-process streaming Engine API, including scheduler IPC and detokenization; it does not include HTTP/network transport. TTFT ends at receipt of the first nonempty generated-text event. TPOT uses receipt of the first and last token-count increases, divided by N-1; it is unavailable if the first token event contains multiple tokens. Every stream event and cumulative count is saved.",
        "",
        "Primary latency has no added per-step device synchronization. A separate forward pass synchronizes immediately before and after ModelExecutor._forward_step. That wall-time scope includes model computation, logits, sampling and the cache work inside that method, with Python/launch overhead. It excludes scheduler queueing, input-buffer preparation, attention metadata built earlier, and response transport. It is not a pure device-kernel duration. All decoder-layer state updates remain inside the real execution path.",
        "",
        "Each Engine call creates a fresh request, with prefix caching, speculation and scheduling overlap disabled. All four workload shapes are warmed before measurement in each pass. Process-local hooks query allocator counters and retain scalar records in RAM; control files are read at prompt boundaries and records are flushed on the next prompt. Each request is followed by an unmeasured one-token control request that flushes its buffered records and profile before the next client timer. These hooks still add some host overhead even to primary latency. No numerical capture or teacher-forcing hooks are installed.",
        "",
        "## HBM and profiling",
        "",
        "startup_memory.json records framework allocated/reserved and startup peaks before the first peak reset. Instrumentation records request-boundary resets, before/after allocator bytes, and absolute warmed request peaks. Model weights, preallocated KV/recurrent pools and workspace are included, not separately attributed. Bytes are converted to MiB with 2^20. These are process allocator counters, not device-wide HBM usage. Large startup pools can hide request-length differences.",
        "",
        "npu_smi_idle.json, npu_smi_allocated.json and npu_smi_in*_out*.json retain device-wide raw snapshots and collection intervals. Workload snapshots run in separate memory requests after the first streamed event. Compare those intervals to request_start_monotonic/request_end_monotonic; a snapshot finishing after a short request is not guaranteed to represent its active HBM. A missing tool/API is recorded as unavailable.",
        "",
        "Profiles are collected after warmup, separately from both timing passes. profiles/ retains CPU/NPU timelines; profile_summary.json groups recognized device-event names and reports both summed durations and union busy time across streams. Overlapping duration sums must not be interpreted as wall time. Large observed categories identify candidates for investigation, not proof of a compute-, memory-, or host-bound hardware limit.",
        "",
        "## Correctness and completeness",
        "",
        f"Linked Step 11 evidence: `{evidence}`. Performance measurements do not clear outstanding correctness failures.",
        f"Observed short and long NPU profiles: **{result['short_and_long_device_profiles_observed']}**. Failed requests: `{result['failed_requests']}`.",
        "",
        "Reduced-repetition or altered-length runs are smoke validation, not the default ten-repetition baseline. Raw samples and errors are retained even when a request fails or the worker times out.",
    ]
    if profiles:
        lines += ["", "## Observed profile groups", ""]
        for profile in profiles:
            lines.append(
                f"- `{profile['path']}`: {profile['status']}; dominant name-based group `{profile.get('dominant_observed_group')}`; top kernels and overlap-aware statistics are in profile_summary.json."
            )
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    return result
