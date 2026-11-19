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


"""Download pinned Homework 2 checkpoints without changing the Python environment."""

import argparse
import json
from pathlib import Path

REVISIONS = {
    "Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
    "Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "Qwen3.5-27B": "fc05daec18b0a78c049392ed2e771dde82bdf654",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=sorted(REVISIONS), required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    for name in args.models:
        destination = args.destination / name
        snapshot_download(
            f"Qwen/{name}",
            revision=REVISIONS[name],
            local_dir=destination,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.jinja",
                "merges.txt",
                "LICENSE",
            ],
        )
        (destination / "checkpoint_provenance.json").write_text(
            json.dumps(
                dict(repository=f"Qwen/{name}", revision=REVISIONS[name]), indent=2
            )
        )
        print(destination, flush=True)


if __name__ == "__main__":
    main()
