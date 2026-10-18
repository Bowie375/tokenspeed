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
