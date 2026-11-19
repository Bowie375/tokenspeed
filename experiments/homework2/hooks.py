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


"""Disposable correctness hooks; each TP rank writes its own real model/state evidence."""

import json
from pathlib import Path

import torch

from experiments.homework1.evaluation.capture import Capture


class TPCapture(Capture):
    """Extend first-GDN-layer evidence with the first full-attention layer's stored KV rows."""

    def __init__(self, directory: Path):
        super().__init__(directory)
        self.kv_values: tuple[torch.Tensor, torch.Tensor] | None = None
        self.collective_events: list[dict] = []

    def begin(self, ctx, input_ids, positions):
        self.kv_values = None
        self.collective_events = []
        return super().begin(ctx, input_ids, positions)

    def sample(self, logits, sampling_info, tokens):
        pending = self.pending
        super().sample(logits, sampling_info, tokens)
        if pending is None or self.kv_values is None:
            return
        slots = sampling_info.req_pool_indices.cpu().tolist()
        first = self.slots[slots[0]]
        step = self.steps[slots[0]] - 1
        destination = (
            self.directory
            / self.job["name"]
            / f"communication_{first['name']}_{step:03d}.json"
        )
        destination.write_text(json.dumps(self.collective_events, indent=2))
        for row, slot in enumerate(slots):
            fixture = self.slots[slot]
            step = self.steps[slot] - 1
            filename = (
                self.directory / self.job["name"] / f"{fixture['name']}_{step:03d}.pt"
            )
            saved = torch.load(filename, map_location="cpu", weights_only=True)
            left, right = pending["bounds"][row : row + 2]
            saved.update(
                kv_key=self.kv_values[0][left:right].clone(),
                kv_value=self.kv_values[1][left:right].clone(),
            )
            torch.save(saved, filename)


def install(directory: Path):
    """Observe model loading and reuse the teacher-forcing/state capture on every rank.

    No collectives, cache allocation, kernels or numerical outputs are replaced.
    Teacher forcing is enabled only by individual correctness jobs.
    """
    from tokenspeed_kernel.platform import current_platform

    from tokenspeed.runtime.execution.model_runner import ModelRunner

    original = ModelRunner.load_model

    def load_model(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        rank_directory = directory / f"rank{self.global_rank}"
        rank_directory.mkdir(parents=True, exist_ok=True)
        (rank_directory / "active.json").symlink_to(directory / "active.json")
        parameters = {
            name: dict(shape=list(p.shape), dtype=str(p.dtype), device=str(p.device))
            for name, p in self.model.named_parameters()
        }
        payload = dict(
            rank=self.global_rank,
            world_size=self.mapping.world_size,
            device=self.gpu_id,
            platform=repr(current_platform()),
            parameters=parameters,
        )
        (rank_directory / "model.json").write_text(json.dumps(payload, indent=2))
        import experiments.homework1.evaluation.capture as capture_module
        from tokenspeed.runtime.layers.paged_attention import PagedAttention

        config = json.loads((directory / "config.json").read_text())
        if config["task"] == "correctness":
            capture_module.Capture = TPCapture
            capture = capture_module.install(rank_directory)
            from experiments.homework2.communication import install_collective_trace

            install_collective_trace(capture)
            prologue_original = PagedAttention.prologue
            full_layer = next(
                layer.layer_id
                for layer in self.model.modules()
                if isinstance(layer, PagedAttention)
            )

            def prologue(layer, q, k, v, positions, ctx):
                result = prologue_original(layer, q, k, v, positions, ctx)
                if capture.pending is not None and layer.layer_id == full_layer:
                    target = layer._write_target(q, ctx)
                    indices = target.slots.long()
                    capture.kv_values = (
                        target.k_cache.index_select(0, indices).detach().cpu().clone(),
                        target.v_cache.index_select(0, indices).detach().cpu().clone(),
                    )
                return result

            PagedAttention.prologue = prologue
        # Record allocator values in the model's device process, rather than the client.
        memory = {
            name: int(getter())
            for name, getter in (
                ("allocated", torch.npu.memory_allocated),
                ("reserved", torch.npu.memory_reserved),
                ("peak_allocated", torch.npu.max_memory_allocated),
            )
        }
        (rank_directory / "load_memory.json").write_text(json.dumps(memory, indent=2))
        return result

    forward_original = ModelRunner.forward

    def forward(self, *args, **kwargs):
        result = forward_original(self, *args, **kwargs)
        active = json.loads((directory / "active.json").read_text())
        if active.get("phase") == "memory":
            memory = dict(
                allocated=int(torch.npu.memory_allocated()),
                reserved=int(torch.npu.memory_reserved()),
                peak_allocated=int(torch.npu.max_memory_allocated()),
                peak_reserved=int(torch.npu.max_memory_reserved()),
            )
            (directory / f"rank{self.global_rank}" / "memory.json").write_text(
                json.dumps(memory)
            )
        return result

    ModelRunner.load_model = load_model
    ModelRunner.forward = forward
