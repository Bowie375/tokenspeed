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


"""Explicit Step 12 workloads and one comparable eager Ascend configuration."""

from dataclasses import dataclass

from experiments.homework1.evaluation.fixtures import Fixture


@dataclass(frozen=True)
class PerformanceConfig:
    """Benchmark policy; execution and EOS choices are always explicit."""

    input_lengths: tuple[int, int]
    output_lengths: tuple[int, int]
    warmups: int
    repetitions: int
    eos_policy: str
    execution_mode: str
    context_length: int
    capacity: int
    seed: int
    profile: bool

    def __post_init__(self):
        for name, lengths in (
            ("input", self.input_lengths),
            ("output", self.output_lengths),
        ):
            if (
                len(lengths) != 2
                or len(set(lengths)) != 2
                or any(type(n) is not int or n < 1 for n in lengths)
            ):
                raise ValueError(
                    f"{name} lengths must contain two distinct positive integers"
                )
        if self.warmups < 1 or self.repetitions < 1:
            raise ValueError("warmups and repetitions must be positive")
        if max(self.input_lengths) + max(self.output_lengths) > self.context_length:
            raise ValueError("prompt plus output exceeds context length")
        if self.capacity < self.context_length:
            raise ValueError("cache capacity must cover the context length")
        if self.eos_policy not in ("fixed_length", "natural"):
            raise ValueError("EOS policy must be fixed_length or natural")
        if self.execution_mode != "eager":
            raise ValueError("this baseline implements eager execution explicitly")


@dataclass(frozen=True)
class Workload:
    """One saved-token prompt and an explicit generated-token target."""

    name: str
    input_ids: tuple[int, ...]
    output_tokens: int


def workload_matrix(
    config: PerformanceConfig, fixtures: list[Fixture]
) -> list[Workload]:
    """Build the Cartesian matrix without retokenizing or truncating saved IDs."""
    result = []
    for length in config.input_lengths:
        matches = [
            f for f in fixtures if len(f.input_ids) == length and f.kind == "raw"
        ]
        if len(matches) != 1:
            raise ValueError(f"need exactly one saved raw fixture of length {length}")
        for output in config.output_lengths:
            result.append(
                Workload(f"in{length}_out{output}", tuple(matches[0].input_ids), output)
            )
    return result


def engine_options(config: PerformanceConfig, checkpoint: str) -> dict:
    """Return the exact recorded Engine launch options; no correctness hooks."""
    return dict(
        model=checkpoint,
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
        max_model_len=config.context_length,
        max_num_seqs=2,
        max_total_tokens=config.capacity,
        max_prefill_tokens=config.context_length,
        chunked_prefill_size=config.context_length,
        seed=config.seed,
        mamba_ssm_dtype="float32",
        stream_interval=1,
        log_level="warning",
    )
