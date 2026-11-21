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

"""Check numerical acceptance against real saved tensor evidence."""

import json

import pytest
import torch

from experiments.homework3.analyze import analyze, compare_capture
from experiments.homework3.summary import check_resources, summarize_matrix


def capture(path, *, value, position, dtype):
    path.mkdir()
    torch.save(
        dict(
            positions=torch.tensor([position]),
            logits=torch.tensor([[value]]),
            state=dict(
                conv=torch.tensor([value], dtype=dtype), ssm=torch.tensor([1.0])
            ),
        ),
        path / "000.pt",
    )


def test_matched_path_requires_bit_exact_state_and_logits(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    capture(left, value=1.0, position=129, dtype=torch.bfloat16)
    capture(right, value=1.0, position=129, dtype=torch.bfloat16)
    assert compare_capture(left, right, exact_logits=True)["steps"] == 1
    record = torch.load(right / "000.pt", weights_only=True)
    record["logits"] += 0.001
    torch.save(record, right / "000.pt")
    with pytest.raises(AssertionError, match="matched-path logits"):
        compare_capture(left, right, exact_logits=True)


def test_state_corruption_is_rejected_even_when_logits_are_not_compared(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    capture(left, value=1.0, position=129, dtype=torch.bfloat16)
    capture(right, value=1.0078125, position=129, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="conv state differs"):
        compare_capture(left, right, exact_logits=False)


def test_missing_evidence_or_changed_context_is_rejected(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    capture(left, value=1.0, position=129, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="incomplete"):
        compare_capture(left, right, exact_logits=True)
    capture(right, value=1.0, position=128, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="boundary"):
        compare_capture(left, right, exact_logits=True)


@pytest.mark.parametrize("key", ["world_size", "max_total_tokens", "attention_backend"])
def test_performance_comparison_rejects_unequal_resources(key):
    launch = dict(
        model="checkpoint",
        world_size=1,
        dtype="bfloat16",
        max_model_len=1024,
        max_num_seqs=2,
        max_total_tokens=8192,
        max_prefill_tokens=512,
        chunked_prefill_size=512,
        attention_backend="mha",
        disable_overlap_schedule=True,
        enforce_eager=True,
    )
    check_resources(launch, dict(launch))
    other = dict(launch)
    other[key] = "different"
    with pytest.raises(ValueError, match=key):
        check_resources(launch, other)


def test_different_replay_start_with_same_end_is_rejected(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    capture(left, value=1.0, position=129, dtype=torch.bfloat16)
    capture(right, value=1.0, position=129, dtype=torch.bfloat16)
    for path, positions in [(left, [128, 129]), (right, [0, 129])]:
        record = torch.load(path / "000.pt", weights_only=True)
        record["positions"] = torch.tensor(positions)
        torch.save(record, path / "000.pt")
    with pytest.raises(AssertionError, match="context"):
        compare_capture(left, right, exact_logits=True)


def performance_artifacts(path, *, ratio):
    """Construct parser inputs, not a substitute runtime or a native result."""
    path.mkdir()
    rank = path / "rank0"
    rank.mkdir()
    launch = dict(
        model="checkpoint",
        world_size=1,
        dtype="bfloat16",
        max_model_len=1024,
        max_num_seqs=2,
        max_total_tokens=8192,
        max_prefill_tokens=512,
        chunked_prefill_size=512,
        attention_backend="mha",
        disable_overlap_schedule=True,
        enforce_eager=True,
        disable_kvstore=ratio == 0,
        kvstore_ratio=ratio,
        debug_force_retraction_interval=0,
    )
    jobs = [
        dict(name=name, action="generate", ids=ids, steps=1, capture=False)
        for name, ids in [
            ("high_0", [42, 42]),
            ("high_1", [42, 42]),
            ("low_0", [43, 43]),
        ]
    ]
    gauge = dict(
        device_total=8,
        device_active=0,
        device_free=4,
        host_pinned=0,
        host_free=0 if ratio == 0 else 2,
        host_cached=0 if ratio == 0 else 2,
        host_evictions=0,
        snapshot_free=0,
    )
    records = [
        dict(
            name=job["name"],
            action="generate",
            input_tokens=2,
            output_tokens=1,
            seconds=1.0,
            response=dict(
                output_ids=[1], meta_info=dict(completion_tokens=1, cached_tokens=0)
            ),
            ranks=[dict(before=gauge, after=gauge)],
        )
        for job in jobs
    ]
    for name, value in [
        ("launch", launch),
        ("jobs", jobs),
        ("results", records),
        ("complete", dict(completed=True)),
    ]:
        (path / f"{name}.json").write_text(json.dumps(value))
    if ratio:
        (rank / "storage.jsonl").write_text(
            json.dumps(dict(allocated_bytes=256, parent_bytes=64, parents=4)) + "\n"
        )
        (rank / "transfers.jsonl").write_text(
            json.dumps(
                dict(
                    kind="_start_writing",
                    direction="d2h",
                    bytes=8,
                    rows=[[0, 1, 1]],
                    milliseconds=0.1,
                )
            )
            + "\n"
        )
    return path


def test_missing_host_storage_evidence_is_rejected(tmp_path):
    directory = performance_artifacts(tmp_path / "run", ratio=4)
    (directory / "rank0/storage.jsonl").unlink()
    with pytest.raises(AssertionError, match="storage"):
        analyze(directory, task="performance", require_eviction=False)


def test_fair_matrix_rejects_different_request_sequences(tmp_path):
    directories = [
        performance_artifacts(tmp_path / f"run{index}", ratio=ratio)
        for index, ratio in enumerate((0, 0.5, 4))
    ]
    jobs_path = directories[-1] / "jobs.json"
    jobs = json.loads(jobs_path.read_text())
    jobs[0]["ids"] = [99, 99]
    jobs_path.write_text(json.dumps(jobs))
    with pytest.raises(ValueError, match="workload"):
        summarize_matrix(directories)


def test_eviction_coverage_requires_a_subsequent_miss(tmp_path):
    directory = performance_artifacts(tmp_path / "run", ratio=4)
    path = directory / "results.json"
    records = json.loads(path.read_text())
    for record in records:
        record["response"]["meta_info"]["cached_tokens"] = 1
        record["ranks"][0]["after"]["host_evictions"] = 1
    path.write_text(json.dumps(records))
    with pytest.raises(AssertionError, match="miss"):
        analyze(directory, task="performance", require_eviction=True)


def test_host_capacity_bounds_are_checked(tmp_path):
    directory = performance_artifacts(tmp_path / "run", ratio=4)
    path = directory / "results.json"
    records = json.loads(path.read_text())
    records[0]["ranks"][0]["after"]["host_free"] = 5
    path.write_text(json.dumps(records))
    with pytest.raises(AssertionError, match="Host capacity"):
        analyze(directory, task="performance", require_eviction=False)
