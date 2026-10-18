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


import csv
import json

import pytest
import torch

from experiments.homework1.evaluation.fixtures import exact_length_ids
from experiments.homework1.evaluation.metrics import (
    Limits,
    RunArtifacts,
    compare_tensors,
)


def test_metrics_report_actual_errors_and_accept_boundary():
    result = compare_tensors(
        torch.tensor([1.0, 3.0]),
        torch.tensor([1.0, 2.0]),
        limits=Limits(1.0, 0.0, 1.0, 1.0),
    )
    assert result["max_abs"] == 1.0
    assert result["rmse"] == pytest.approx(2**-0.5)
    assert result["relative_l2"] == pytest.approx(5**-0.5)
    assert result["passed"]
    assert not compare_tensors(
        torch.tensor([1.0, 3.0]),
        torch.tensor([1.0, 2.0]),
        limits=Limits(0.9, 0.0, 1.0, 1.0),
    )["passed"]


def test_metrics_reject_shape_broadcast_and_nonfinite_values():
    with pytest.raises(ValueError, match="shape"):
        compare_tensors(
            torch.zeros(2, 1), torch.zeros(2), limits=Limits(0.0, 0.0, 0.0, 0.0)
        )
    for value in (float("nan"), float("inf")):
        assert not compare_tensors(
            torch.tensor([value]), torch.zeros(1), limits=Limits(1.0, 0.0, 1.0, 1.0)
        )["passed"]


def test_metrics_zero_reference_and_empty_inputs():
    assert (
        compare_tensors(
            torch.zeros(3), torch.zeros(3), limits=Limits(0.0, 0.0, 0.0, 0.0)
        )["relative_l2"]
        == 0
    )
    assert not compare_tensors(
        torch.ones(3), torch.zeros(3), limits=Limits(0.1, 0.0, 0.1, 0.1)
    )["passed"]
    assert compare_tensors(
        torch.empty(0, 3), torch.empty(0, 3), limits=Limits(0.0, 0.0, 0.0, 0.0)
    )["passed"]


def test_exact_length_ids_repeats_without_global_rng():
    assert exact_length_ids([2, 7, 9], length=5) == [2, 7, 9, 2, 7]
    assert exact_length_ids([2], length=1) == [2]
    with pytest.raises(ValueError):
        exact_length_ids([], length=4)
    with pytest.raises(ValueError):
        exact_length_ids([1], length=0)


def test_artifacts_preserve_failed_comparisons_and_reject_overwrite(tmp_path):
    run = RunArtifacts(tmp_path, run_id="case", manifest={"seed": 17})
    run.record(
        "bad",
        torch.ones(2),
        torch.zeros(2),
        limits=Limits(0.0, 0.0, 0.0, 0.0),
        details={"step": 3},
    )
    rows = list(csv.DictReader((run.path / "metrics.csv").open()))
    assert rows[0]["passed"] == "False"
    tensors = torch.load(run.path / "tensors/bad.pt", weights_only=True)
    torch.testing.assert_close(tensors["actual"], torch.ones(2))
    assert json.loads((run.path / "manifest.json").read_text())["seed"] == 17
    with pytest.raises(FileExistsError):
        RunArtifacts(tmp_path, run_id="case", manifest={})


def test_saved_fixtures_roundtrip_without_retokenizing(tmp_path):
    from experiments.homework1.evaluation.fixtures import (
        Fixture,
        load_fixtures,
        save_fixtures,
    )

    fixtures = [
        Fixture("zh", [19, 8, 2], [3, 4], "chat", "中文"),
        Fixture("length", [4] * 129, [7], "raw", ""),
    ]
    target = tmp_path / "fixtures.json"
    save_fixtures(target, fixtures)
    assert load_fixtures(target) == fixtures
    with pytest.raises(ValueError, match="unique"):
        save_fixtures(target, fixtures + fixtures)


def test_native_step_evidence_requires_complete_sequence(tmp_path):
    from experiments.homework1.evaluation.fixtures import Fixture
    from experiments.homework1.evaluation.model import load_steps

    fixture = Fixture("a", [1, 3], [5], "raw", "")
    (tmp_path / "job").mkdir()
    with pytest.raises(RuntimeError, match="No captures"):
        load_steps(tmp_path, "job", fixture)
    torch.save(
        dict(step=0, input_ids=[1, 3], logits=torch.tensor([1.0, 3.0])),
        tmp_path / "job/a_000.pt",
    )
    assert load_steps(tmp_path, "job", fixture)[0]["input_ids"] == [1, 3]
    torch.save(
        dict(step=2, input_ids=[5], logits=torch.tensor([3.0, 1.0])),
        tmp_path / "job/a_002.pt",
    )
    with pytest.raises(RuntimeError, match="Missing"):
        load_steps(tmp_path, "job", fixture)


def test_capture_preserves_logits_and_records_true_teacher_sequence(tmp_path):
    from experiments.homework1.evaluation.capture import Capture
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo

    capture = Capture(tmp_path)
    capture.job = dict(
        name="teacher",
        steps=2,
        teacher=True,
        fixtures=[dict(name="a", input_ids=[2, 7], continuation_ids=[9, 3])],
    )
    info = SamplingBatchInfo(req_pool_indices=torch.tensor([5]), device="cpu")
    logits = torch.tensor([[1.0, 4.0, 2.0]])
    original = logits.clone()
    for step, ids in enumerate(([2, 7], [9])):
        capture.pending = dict(
            model_logits=logits.clone(),
            ids=torch.tensor(ids),
            positions=torch.arange(len(ids)),
            bounds=[0, len(ids)],
            batch_number=step,
            batch_size=1,
            mode="extend" if step == 0 else "decode",
            num_extends=1 if step == 0 else 0,
            cached_decode=step > 0,
        )
        token = logits.argmax(-1)
        capture.sample(logits, info, token)
        assert token.tolist() == [[9, 3][step]]
        torch.testing.assert_close(logits, original, rtol=0, atol=0)
        saved = torch.load(tmp_path / f"teacher/a_{step:03d}.pt", weights_only=True)
        assert saved["step"] == step
        assert saved["input_ids"] == ids
        assert saved["cached_decode"] == (step > 0)
        torch.testing.assert_close(saved["logits"], original[0])
    assert len(list((tmp_path / "teacher").glob("*.pt"))) == 2


def test_empty_runs_do_not_count_as_passing_evidence(tmp_path):
    run = RunArtifacts(tmp_path, run_id="empty", manifest={})
    assert run.summary() == dict(total=0, passed=0, failed=0, all_passed=False)


def test_fixture_rejects_path_traversal_and_invalid_tokens():
    from experiments.homework1.evaluation.fixtures import Fixture

    with pytest.raises(ValueError, match="name"):
        Fixture("../escape", [1], [2], "raw", "")
    with pytest.raises(ValueError, match="token"):
        Fixture("negative", [1, -2], [3], "raw", "")


def test_unavailable_check_prevents_passing_summary_and_keeps_reason(tmp_path):
    run = RunArtifacts(tmp_path, run_id="failed_worker", manifest={})
    run.record(
        "available", torch.ones(1), torch.ones(1), limits=Limits(0, 0, 0, 0), details={}
    )
    run.record_failure(
        "long_decode", "worker died", details={"log": "native/worker.log"}
    )
    assert run.summary() == dict(total=2, passed=1, failed=1, all_passed=False)
    failure = json.loads((run.path / "failures.json").read_text())[0]
    assert failure["reason"] == "worker died"
    assert failure["details"]["log"] == "native/worker.log"
