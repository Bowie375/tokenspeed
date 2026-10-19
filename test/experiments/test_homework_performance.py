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


import pytest

from experiments.homework1.evaluation.fixtures import Fixture
from experiments.homework1.performance.config import PerformanceConfig, workload_matrix
from experiments.homework1.performance.measurements import (
    StreamMeasurements,
    summarize_samples,
)
from experiments.homework1.performance.profiling import summarize_events


def config(**changes):
    values = dict(
        input_lengths=(128, 2048),
        output_lengths=(32, 128),
        warmups=3,
        repetitions=10,
        eos_policy="fixed_length",
        execution_mode="eager",
        context_length=4096,
        capacity=32768,
        seed=17,
        profile=True,
    )
    return PerformanceConfig(**dict(values, **changes))


def fixtures():
    return [Fixture(f"raw_{n}", [7] * n, [], "raw", "") for n in (128, 2048)]


def test_matrix_uses_exact_saved_ids_and_four_combinations():
    matrix = workload_matrix(config(), fixtures())
    assert [(len(w.input_ids), w.output_tokens) for w in matrix] == [
        (128, 32),
        (128, 128),
        (2048, 32),
        (2048, 128),
    ]
    assert all(list(w.input_ids) == [7] * len(w.input_ids) for w in matrix)


@pytest.mark.parametrize(
    "changes",
    [
        dict(input_lengths=(128, 128)),
        dict(output_lengths=(32, 32)),
        dict(repetitions=0),
        dict(warmups=0),
        dict(context_length=2100),
        dict(eos_policy="unknown"),
        dict(execution_mode="graph"),
    ],
)
def test_invalid_workload_is_rejected(changes):
    with pytest.raises(ValueError):
        config(**changes)


def test_missing_saved_fixture_is_rejected():
    with pytest.raises(ValueError, match="2048"):
        workload_matrix(config(), fixtures()[:1])


def test_stream_ignores_empty_text_and_uses_cumulative_token_counts():
    stream = StreamMeasurements(start=10.0)
    stream.observe(10.2, dict(text="", meta_info=dict(completion_tokens=1)))
    stream.observe(10.3, dict(text="a", meta_info=dict(completion_tokens=1)))
    stream.observe(10.5, dict(text="ab", meta_info=dict(completion_tokens=2)))
    stream.observe(
        10.9,
        dict(
            text="abc",
            meta_info=dict(completion_tokens=3, finish_reason=dict(type="length")),
        ),
    )
    result = stream.finish(end=11.0, target_tokens=3, eos_policy="fixed_length")
    assert result["client_ttft_ms"] == pytest.approx(300)
    assert result["client_tpot_ms"] == pytest.approx(350)
    assert result["e2e_ms"] == pytest.approx(1000)
    assert result["actual_output_tokens"] == 3
    assert result["valid"]


def test_coalesced_first_stream_event_has_no_per_token_tpot():
    stream = StreamMeasurements(start=0)
    stream.observe(0.2, dict(text="ab", meta_info=dict(completion_tokens=2)))
    stream.observe(
        0.3,
        dict(
            text="abcd",
            meta_info=dict(completion_tokens=4, finish_reason=dict(type="length")),
        ),
    )
    result = stream.finish(end=0.4, target_tokens=4, eos_policy="fixed_length")
    assert result["client_tpot_ms"] is None
    assert result["client_tpot_reason"] == "first_event_contains_multiple_tokens"


def test_natural_eos_keeps_actual_count_and_fixed_length_rejects_short_output():
    for policy, valid in [("natural", True), ("fixed_length", False)]:
        stream = StreamMeasurements(start=0)
        stream.observe(
            0.1,
            dict(
                text="x",
                meta_info=dict(completion_tokens=1, finish_reason=dict(type="stop")),
            ),
        )
        result = stream.finish(end=0.2, target_tokens=32, eos_policy=policy)
        assert result["valid"] == valid
        assert result["actual_output_tokens"] == 1
        assert result["client_tpot_ms"] is None


def test_unfinished_or_decreasing_stream_is_rejected():
    stream = StreamMeasurements(start=0)
    stream.observe(0.1, dict(text="ab", meta_info=dict(completion_tokens=2)))
    with pytest.raises(ValueError):
        stream.observe(0.2, dict(text="a", meta_info=dict(completion_tokens=1)))
    assert not stream.finish(end=0.3, target_tokens=2, eos_policy="fixed_length")[
        "valid"
    ]


def test_summary_excludes_failed_and_warmup_samples_and_reports_variability():
    rows = [
        dict(
            case="a",
            phase="latency",
            warmup=False,
            valid=True,
            client_ttft_ms=v,
            actual_output_tokens=32,
        )
        for v in (1, 3, 8)
    ]
    rows += [
        dict(
            case="a",
            phase="latency",
            warmup=True,
            valid=True,
            client_ttft_ms=999,
            actual_output_tokens=32,
        ),
        dict(
            case="a",
            phase="latency",
            warmup=False,
            valid=False,
            client_ttft_ms=999,
            actual_output_tokens=2,
        ),
    ]
    result = summarize_samples(rows)[0]
    assert result["measured_repetitions"] == 3
    assert result["failed_repetitions"] == 1
    assert result["client_ttft_ms"] == dict(median=3, min=1, max=8)
    assert result["actual_output_tokens"] == [32]


def test_profile_keeps_cpu_and_device_separate_and_does_not_sum_overlaps_as_wall_time():
    events = [
        dict(name="matmul", cat="kernel", ph="X", ts=0, dur=10),
        dict(name="_gdn_recurrence_kernel", cat="kernel", ph="X", ts=5, dur=10),
        dict(name="aten::matmul", cat="cpu_op", ph="X", ts=0, dur=30),
    ]
    result = summarize_events(events)
    assert result["device_busy_union_us"] == 15
    assert result["device_duration_sum_us"] == 20
    assert result["cpu_duration_sum_us"] == 30
    assert result["device_groups_us"]["projections_mlp"] == 10
    assert result["device_groups_us"]["gdn"] == 10
    assert result["device_events"] == 2


def test_profile_without_device_events_is_not_bottleneck_evidence():
    result = summarize_events(
        [dict(name="aten::matmul", cat="cpu_op", ph="X", ts=0, dur=1)]
    )
    assert result["status"] == "no_device_events"
    assert result["dominant_observed_group"] is None


def test_forward_samples_require_real_prefill_and_one_token_decode_counts(tmp_path):
    import json

    from experiments.homework1.performance.results import collect_samples

    (tmp_path / "instrumentation").mkdir()
    request = dict(
        id="r1",
        case="a",
        phase="forward",
        warmup=False,
        repetition=0,
        input_tokens=128,
        target_output_tokens=3,
        actual_output_tokens=3,
        valid=True,
        client_ttft_ms=1,
        e2e_ms=3,
    )
    (tmp_path / "requests.jsonl").write_text(json.dumps(request) + "\n")
    capture = dict(
        events=[
            dict(kind="prefill", duration_ms=4, input_tokens=128),
            dict(kind="decode", duration_ms=2, input_tokens=1),
            dict(kind="decode", duration_ms=6, input_tokens=1),
        ],
        memory_after_bytes=dict(max_memory_allocated=2**20, max_memory_reserved=2**21),
        peak_reset_supported=True,
    )
    path = tmp_path / "instrumentation/r1.json"
    path.write_text(json.dumps(capture))
    result = collect_samples(tmp_path)[0]
    assert result["valid"]
    assert result["prefill_forward_ms"] == 4
    assert result["decode_forward_ms_per_step"] == 4
    assert result["peak_allocated_mib"] == 1
    capture["events"] = capture["events"][:2]
    path.write_text(json.dumps(capture))
    result = collect_samples(tmp_path)[0]
    assert not result["valid"]
    assert "decode" in result["instrumentation_error"]


def test_missing_forward_capture_is_not_a_passing_sample(tmp_path):
    import json

    from experiments.homework1.performance.results import collect_samples

    (tmp_path / "requests.jsonl").write_text(
        json.dumps(dict(id="r", phase="forward", valid=True)) + "\n"
    )
    row = collect_samples(tmp_path)[0]
    assert not row["valid"]
    assert row["instrumentation_error"]


def test_failed_stream_retains_partial_counts_and_events():
    stream = StreamMeasurements(start=0)
    stream.observe(0.1, dict(text="a", meta_info=dict(completion_tokens=1)))
    stream.observe(0.2, dict(text="ab", meta_info=dict(completion_tokens=2)))
    result = stream.fail(
        end=0.3, target_tokens=32, eos_policy="fixed_length", error="connection lost"
    )
    assert not result["valid"]
    assert result["actual_output_tokens"] == 2
    assert len(result["stream_events"]) == 2
    assert result["error"] == "connection lost"


def test_ascend_trace_uses_hardware_process_metadata_and_ignores_overlap_track():
    events = [
        dict(ph="X", name="_gdn_recurrence_kernel", pid=7, tid=1, ts="100.5", dur=3),
        dict(ph="X", name="Computing", pid=8, tid=2, ts="100.5", dur=3),
        dict(ph="M", name="process_name", pid=7, args=dict(name="Ascend Hardware")),
        dict(ph="M", name="process_name", pid=8, args=dict(name="Overlap Analysis")),
        dict(ph="s", name="flow", cat="async_npu", pid=7, ts="100.5"),
    ]
    result = summarize_events(events)
    assert result["device_events"] == 1
    assert result["device_busy_union_us"] == 3
    assert result["device_groups_us"]["gdn"] == 3
