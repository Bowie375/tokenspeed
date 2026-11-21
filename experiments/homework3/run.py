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


"""Run native two-level cache acceptance and fair fixed-capacity comparisons."""

import argparse
import json
import os
import sys
from pathlib import Path

from experiments.homework2.run import execute


def jobs_for(task: str):
    def generate(name, ids, steps, capture):
        return dict(name=name, action="generate", ids=ids, steps=steps, capture=capture)

    def control(name, action):
        return dict(name=name, action=action)

    if task == "functional":
        jobs = []
        for index, length in enumerate((128, 129)):
            prompt = [42 + index] * length
            target = prompt + [80 + index]
            jobs.extend(
                [
                    control(f"cold_clear{index}", "clear_all"),
                    generate(f"cold{index}", target, 8, True),
                    control(f"seed_clear{index}", "clear_all"),
                    generate(f"seed{index}", prompt, 1, False),
                    generate(f"device{index}", target, 8, True),
                    control(f"reseed_clear{index}", "clear_all"),
                    generate(f"reseed{index}", prompt, 1, False),
                    control(f"drop{index}", "clear_device"),
                    generate(f"resume{index}", target, 8, True),
                ]
            )
        # Alternating distinct sessions; every Device clear preserves Host L2.
        for index in range(3):
            jobs.append(generate(f"session_seed{index}", [44 + index] * 129, 1, False))
        for cycle in range(3):
            for index in range(3):
                jobs.extend(
                    [
                        control(f"cycle_drop{cycle}_{index}", "clear_device"),
                        generate(
                            f"cycle{cycle}_{index}",
                            [44 + index] * 129 + [84 + index],
                            8,
                            True,
                        ),
                    ]
                )
        for index in range(3):
            jobs.extend(
                [
                    control(f"reference_clear{index}", "clear_all"),
                    generate(
                        f"reference{index}", [44 + index] * 129 + [84 + index], 8, True
                    ),
                    control(f"device_reference_clear{index}", "clear_all"),
                    generate(
                        f"device_reference_seed{index}", [44 + index] * 129, 1, False
                    ),
                    generate(
                        f"device_reference{index}",
                        [44 + index] * 129 + [84 + index],
                        8,
                        True,
                    ),
                ]
            )
        jobs.append(control("final_clear", "clear_all"))
        return jobs
    if task == "retraction":
        return [
            control("initial_stats", "stats"),
            generate("uninterrupted", [42] * 129 + [80], 20, True),
            control("final_clear", "clear_all"),
        ]
    jobs = [
        generate("warmup", [40] * 129, 8, False),
        control("initial_clear", "clear_all"),
    ]
    # Cold low-reuse requests and a three-session working set, same engine budget.
    for mode in ("high", "low"):
        jobs.append(control(f"clear_{mode}", "clear_all"))
        for round_index in range(4):
            for index in range(3):
                token = 42 + index if mode == "high" else 50 + 3 * round_index + index
                jobs.append(
                    generate(
                        f"{mode}_{round_index}_{index}",
                        [token] * 257 + [80 + index],
                        8,
                        False,
                    )
                )
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--tp", type=int, choices=[1, 2], required=True)
    parser.add_argument(
        "--task", choices=["functional", "performance", "retraction"], required=True
    )
    parser.add_argument(
        "--host-ratio", type=float, required=True, help="0 disables Host L2"
    )
    parser.add_argument("--device-tokens", type=int, required=True)
    parser.add_argument("--force-retraction", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    directory = options.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    bootstrap = directory / "bootstrap"
    bootstrap.mkdir()
    (bootstrap / "sitecustomize.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "from experiments.homework3.hooks import install\n"
        "install(Path(os.environ['HOMEWORK_HOST_CAPTURE']))\n"
    )
    devices = options.devices.split(",")
    if len(set(devices)) != options.tp:
        raise ValueError("Supply one distinct physical chip per TP rank")
    env = os.environ.copy()
    env.update(
        HOMEWORK_HOST_CAPTURE=str(directory),
        HOMEWORK_HOST_VERIFY="0" if options.task == "performance" else "1",
        ASCEND_RT_VISIBLE_DEVICES=",".join(sorted(devices, key=int)),
        OMP_NUM_THREADS="8",
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
        GRPC_ENABLE_HTTP_PROXY="0",
    )
    env["PYTHONPATH"] = (
        os.pathsep.join(
            str(p)
            for p in (
                bootstrap,
                root,
                root / "python",
                root / "tokenspeed-kernel/python",
                root / "tokenspeed-kernel-npu/python",
            )
        )
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    launch = dict(
        model=str(options.model.resolve()),
        world_size=options.tp,
        device="npu",
        dtype="bfloat16",
        language_model_only=True,
        attention_backend="mha",
        sampling_backend="greedy",
        enforce_eager=True,
        disable_prefill_graph=True,
        disable_pdl=True,
        disable_autotune=True,
        enable_prefix_caching=True,
        disable_kvstore=options.host_ratio == 0,
        kvstore_ratio=options.host_ratio,
        kvstore_io_backend="direct",
        disable_overlap_schedule=True,
        retraction_snapshot_ratio=2.0 if options.force_retraction else 0,
        retraction_snapshot_max_requests=2 if options.force_retraction else 0,
        debug_force_retraction_interval=options.force_retraction,
        max_model_len=1024,
        max_num_seqs=2,
        max_total_tokens=options.device_tokens,
        max_prefill_tokens=512,
        chunked_prefill_size=512,
        mamba_ssm_dtype="float32",
        log_level="info",
    )
    (directory / "launch.json").write_text(json.dumps(launch, indent=2))
    (directory / "jobs.json").write_text(json.dumps(jobs_for(options.task), indent=2))
    execute(
        [
            sys.executable,
            "-m",
            "experiments.homework3.worker",
            "--directory",
            str(directory),
        ],
        directory=directory,
        env=env,
        timeout=1800,
        root=root,
    )
    print(directory)


if __name__ == "__main__":
    main()
