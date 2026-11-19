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


"""Independent HF CPU reference, reusable without rerunning saved native jobs."""

import argparse
import json
import os
from pathlib import Path


def generate_reference(directory: Path, *, dtype: str):
    """Save full-vocabulary logits and greedy tokens with an explicit CPU dtype."""
    # Arm Compute's OpenMP scheduler may otherwise create hundreds of threads,
    # independently of torch.set_num_threads. Limit teams before importing torch.
    os.environ.update(
        OMP_NUM_THREADS="8",
        OMP_THREAD_LIMIT="8",
        OPENBLAS_NUM_THREADS="8",
        OMP_WAIT_POLICY="PASSIVE",
        GOMP_SPINCOUNT="0",
    )
    import torch

    from experiments.homework1.evaluation.fixtures import Fixture
    from experiments.homework1.evaluation.model import Reference

    if dtype not in ("float32", "bfloat16"):
        raise ValueError("reference dtype must be float32 or bfloat16")
    config = json.loads((directory / "config.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    fixtures = json.loads((directory / "fixtures.json").read_text())
    destination = directory / "reference"
    destination.mkdir(exist_ok=True)
    settings = dict(
        device="cpu",
        dtype=dtype,
        attention="eager",
        eos_policy="ignore",
        steps=config["steps"],
        openmp_thread_limit=8,
    )
    (directory / "reference_settings.json").write_text(json.dumps(settings, indent=2))
    model = Reference(
        Path(manifest["checkpoint"]),
        threads=8,
        dtype={"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype],
    )
    for f in fixtures:
        fixture = Fixture(f["name"], f["input_ids"], f["continuation_ids"], "raw", "")
        rows = model.greedy(fixture, steps=config["steps"], eos_token_ids=[])
        torch.save(
            [dict(sampled_token=r["token"], logits=r["logits"]) for r in rows],
            destination / f"{f['name']}.pt",
        )
        print(f"HF {dtype} {f['name']}: {len(rows)} cached steps", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], required=True)
    options = parser.parse_args()
    generate_reference(options.directory, dtype=options.dtype)
    from experiments.homework2.analyze import analyze

    directory = options.directory
    manifest = json.loads((directory / "manifest.json").read_text())
    result = analyze(
        directory,
        json.loads((directory / "config.json").read_text()),
        json.loads((directory / "fixtures.json").read_text()),
        manifest["model_config"],
        reference="hf",
    )
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
