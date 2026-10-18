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


"""Numerical comparisons and append-only experiment evidence, independent of IPython."""

import csv
import dataclasses
import json
import math
import re
from pathlib import Path

import torch


@dataclasses.dataclass(frozen=True)
class Limits:
    """Explicit elementwise, relative-L2 and RMSE acceptance criteria."""

    atol: float
    rtol: float
    relative_l2: float
    rmse: float

    def __post_init__(self):
        if any(
            not math.isfinite(value) or value < 0 for value in dataclasses.astuple(self)
        ):
            raise ValueError("all numerical limits must be finite and nonnegative")


def compare_tensors(actual, expected, *, limits: Limits) -> dict:
    """Compare equal-shaped tensors in CPU FP64; nonfinite inputs always fail.

    Returns maximum absolute error, RMSE, relative L2, finiteness, and pass/fail.
    Relative L2 divides by max(norm(reference), 1e-12), including zero references.
    Empty, equal-shaped tensors have zero error. Broadcasting is prohibited.
    """
    if actual.shape != expected.shape:
        raise ValueError(
            f"shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return dict(
            max_abs=None, rmse=None, relative_l2=None, finite=False, passed=False
        )
    delta = a - b
    max_abs = float(delta.abs().max()) if delta.numel() else 0.0
    rmse = float(delta.square().mean().sqrt()) if delta.numel() else 0.0
    relative_l2 = float(delta.norm() / b.norm().clamp_min(1e-12))
    elementwise = bool((delta.abs() <= limits.atol + limits.rtol * b.abs()).all())
    return dict(
        max_abs=max_abs,
        rmse=rmse,
        relative_l2=relative_l2,
        finite=True,
        passed=elementwise
        and relative_l2 <= limits.relative_l2
        and rmse <= limits.rmse,
    )


class RunArtifacts:
    """Own one unique run directory; persist failed results before caller asserts."""

    def __init__(self, root: Path, *, run_id: str, manifest: dict):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError("run_id must be a simple name without path separators")
        self.path = Path(root) / run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.rows: list[dict] = []
        (self.path / "tensors").mkdir()
        self.write_json("manifest.json", manifest)

    def write_json(self, name: str, value):
        """Write JSON inside this run, rejecting nonfinite JSON numbers."""
        if Path(name).name != name:
            raise ValueError("artifact name must not contain directories")
        (self.path / name).write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )

    def record(self, name, actual, expected, *, limits: Limits, details: dict):
        """Save tensor evidence and numerical metrics; return the comparison row."""
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("comparison name must be a simple unique name")
        target = self.path / "tensors" / f"{name}.pt"
        if target.exists():
            raise FileExistsError(target)
        result = compare_tensors(actual, expected, limits=limits)
        row = dict(
            name=name,
            **result,
            **{f"limit_{k}": v for k, v in dataclasses.asdict(limits).items()},
            details=json.dumps(details, ensure_ascii=False, allow_nan=False),
        )
        torch.save(
            dict(
                actual=actual.detach().cpu().clone(),
                expected=expected.detach().cpu().clone(),
                details=details,
            ),
            target,
        )
        self.rows.append(row)
        with (self.path / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(self.rows)
        self.write_json("metrics.json", self.rows)
        return row

    def summary(self):
        """Return aggregate comparison counts; an empty run is not a passing run."""
        passed = sum(row["passed"] for row in self.rows)
        return dict(
            total=len(self.rows),
            passed=passed,
            failed=len(self.rows) - passed,
            all_passed=bool(self.rows) and passed == len(self.rows),
        )
