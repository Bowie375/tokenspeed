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


"""Disposable native Engine worker. Never import this to start a server."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--eos-token-ids", type=json.loads, required=True)
    options = parser.parse_args()
    directory = options.directory
    from tokenspeed.runtime.entrypoints.engine import Engine

    engine = Engine(
        model=options.checkpoint,
        device="npu",
        dtype="bfloat16",
        language_model_only=True,
        attention_backend="mha",
        sampling_backend="greedy",
        enforce_eager=True,
        disable_prefill_graph=True,
        disable_pdl=True,
        disable_autotune=True,
        enable_prefix_caching=False,
        disable_kvstore=True,
        disable_overlap_schedule=True,
        retraction_snapshot_ratio=0,
        max_model_len=options.context_length,
        max_num_seqs=2,
        max_total_tokens=options.capacity,
        max_prefill_tokens=options.context_length,
        chunked_prefill_size=options.context_length,
        seed=options.seed,
        mamba_ssm_dtype="float32",
        log_level="warning",
    )
    try:
        jobs = json.loads((directory / "jobs.json").read_text())
        for job in jobs:
            job["capture_enabled"] = True
            temporary = directory / "active.next.json"
            temporary.write_text(json.dumps(job))
            temporary.replace(directory / "active.json")
            results = engine.generate(
                input_ids=[f["input_ids"] for f in job["fixtures"]],
                sampling_params=dict(
                    temperature=0,
                    max_new_tokens=job["steps"],
                    ignore_eos=job["teacher"],
                    stop_token_ids=[] if job["teacher"] else options.eos_token_ids,
                ),
            )
            (directory / f"{job['name']}_response.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2)
            )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
