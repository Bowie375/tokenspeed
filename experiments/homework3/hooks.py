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


"""Process-local observers of real transfers, scheduler capacity and model states.

The only control operation is an explicitly requested, quiescent Device-index
clear. Host entries, transfers, sampling and numerical kernels are unchanged.
"""

import inspect
import json
import os
import threading
from pathlib import Path

import torch


def install(directory: Path):
    from tokenspeed.runtime.cache.l2.executor import HostCacheExecutor
    from tokenspeed.runtime.engine.event_loop import EventLoop
    from tokenspeed.runtime.execution.model_runner import ModelRunner
    from tokenspeed.runtime.models.qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
        Qwen3_5LinearDecoderLayer,
    )
    from tokenspeed.runtime.utils import get_device_module

    device = get_device_module()
    lock = threading.Lock()
    pending = []
    executor = [None]
    captures = {}
    counter = {}

    def emit(rank, name, data):
        path = directory / f"rank{rank}"
        path.mkdir(exist_ok=True)
        with lock, (path / f"{name}.jsonl").open("a") as stream:
            stream.write(json.dumps(data) + "\n")

    init = HostCacheExecutor.__init__

    def initialize(self, *args, **kwargs):
        init(self, *args, **kwargs)
        executor[0] = self
        storage = self.host_storage
        emit(
            self.attn_tp_rank,
            "storage",
            dict(
                allocated_bytes=0 if storage is None else storage.host_buffer_bytes,
                parent_bytes=0 if storage is None else storage.host_lcm_block_bytes,
                parents=0 if storage is None else storage.num_host_lcm_blocks,
                snapshot_parents=self.num_snapshot_pages - 1,
                slot_blob_bytes=self.blob_bytes,
                device_pool_bytes=sum(
                    b.numel() * b.element_size() for b in self.layout.buffers
                ),
                allocated_device_bytes=int(device.memory_allocated()),
                reserved_device_bytes=int(device.memory_reserved()),
                groups=[
                    dict(
                        name=g.group_id,
                        fields=[
                            dict(name=f.field_id, bytes=f.payload_bytes)
                            for f in g.fields
                        ],
                    )
                    for g in self.layout.groups
                ],
            ),
        )

    HostCacheExecutor.__init__ = initialize

    def instrument(method, direction, stream_name):
        original = getattr(HostCacheExecutor, method)
        signature = inspect.signature(original)

        def transfer(self, *args, **kwargs):
            bound = signature.bind(self, *args, **kwargs).arguments
            if method == "_start_restore":
                rows = [row for tier in bound["rows_by_tier"].values() for row in tier]
            else:
                rows = list(bound["transfers"])
            if not rows and not bound.get("ops"):
                return original(self, *args, **kwargs)
            stream = getattr(self, stream_name)
            start, end = device.Event(enable_timing=True), device.Event(
                enable_timing=True
            )
            prerequisite = bound["prerequisite_stream"]
            if prerequisite is not None:
                stream.wait_stream(prerequisite)
            start.record(stream)
            result = original(self, *args, **kwargs)
            end.record(stream)
            size = sum(
                sum(f.payload_bytes for f in self.layout.groups[g].fields)
                for g, _, _ in rows
            )
            verified = 0
            if os.environ["HOMEWORK_HOST_VERIFY"] == "1":
                end.synchronize()
                from tokenspeed.runtime.cache.transfer.ops import HostTier

                tiers = bound.get("rows_by_tier")
                selections = (
                    [(self.host_storage, rows)]
                    if tiers is None
                    else [
                        (self.host_storage, tiers[HostTier.L2]),
                        (self.snapshot_storage, tiers[HostTier.SNAPSHOT_POOL]),
                    ]
                )
                if method == "_start_snapshot_store":
                    selections = [(self.snapshot_storage, rows)]
                for storage, selection in selections:
                    for group_index, block, host_block in selection:
                        for field_index, field in enumerate(
                            self.layout.groups[group_index].fields
                        ):
                            offset = (
                                field.device_block_zero_offset_bytes
                                + block * field.block_stride_bytes
                            )
                            actual = (
                                self.layout.buffers[field.device_buffer_index]
                                .view(torch.uint8)
                                .flatten()[offset : offset + field.payload_bytes]
                                .cpu()
                            )
                            host_offset = storage.host_field_offset(
                                group_index, host_block, field_index
                            )
                            expected = storage.host_buffer[
                                host_offset : host_offset + field.payload_bytes
                            ]
                            if not torch.equal(actual, expected):
                                raise AssertionError(
                                    f"{direction} corrupt {field.field_id} block {block}"
                                )
                            verified += field.payload_bytes
            blob_bytes = (
                self._slot_state.nbytes * len(bound["ops"]) if bound.get("ops") else 0
            )
            active = json.loads((directory / "active.json").read_text())
            record = dict(
                kind=method,
                direction=direction,
                bytes=size,
                cache_bytes=size,
                slot_blob_bytes=blob_bytes,
                verified_bytes=verified,
                job=active["name"],
                rows=[list(row) for row in rows],
            )
            with lock:
                pending.append((self.attn_tp_rank, start, end, record))
            return result

        setattr(HostCacheExecutor, method, transfer)

    for method, direction, stream in [
        ("_start_writing", "d2h", "write_stream"),
        ("_start_loading", "h2d", "load_stream"),
        ("_start_snapshot_store", "d2h", "write_stream"),
        ("_start_restore", "h2d", "load_stream"),
    ]:
        instrument(method, direction, stream)

    process = EventLoop._process_new_requests

    def receive(self):
        # Query events on the control plane; never synchronize device work here.
        ready = []
        with lock:
            for item in list(pending):
                if item[2].query():
                    pending.remove(item)
                    ready.append(item)
        for rank, start, end, record in ready:
            record["milliseconds"] = float(start.elapsed_time(end))
            emit(rank, "transfers", record)
        command_path = directory / "command.json"
        if (
            command_path.exists()
            and self._num_running() == 0
            and self.scheduler.waiting_size() == 0
            and self._cache_hooks._num_open == 0
        ):
            command = json.loads(command_path.read_text())
            ack = directory / f"rank{self.attn_tp_rank}" / f"ack_{command['id']}.json"
            if not ack.exists():
                s = self.scheduler

                def gauges():
                    return dict(
                        device_free=s.empty_lcm_blocks(),
                        device_active=s.active_lcm_blocks(),
                        device_total=self._scheduler_cache_geometry.num_usable_pages,
                        host_free=s.host_pool_free_blocks(),
                        host_cached=s.host_pool_cached_blocks(),
                        host_pinned=s.host_pool_pinned_blocks(),
                        host_evictions=s.host_cache_evictions(),
                        snapshot_free=s.snapshot_pool_free_blocks(),
                    )

                before = gauges()
                if command["action"] == "clear_device":
                    if not s.clear_l1_cache():
                        raise RuntimeError("Device cache clear rejected at quiescence")
                elif command["action"] == "clear_all":
                    if not s.clear_cache():
                        raise RuntimeError("Cache clear rejected at quiescence")
                elif command["action"] != "stats":
                    raise ValueError(command)
                ack.parent.mkdir(exist_ok=True)
                temporary = ack.with_suffix(".tmp")
                temporary.write_text(json.dumps(dict(before=before, after=gauges())))
                temporary.replace(ack)
        return process(self)

    EventLoop._process_new_requests = receive

    load = ModelRunner.load_model

    def load_model(self, *args, **kwargs):
        result = load(self, *args, **kwargs)
        # Each rank has its own process and capture dictionary.
        captures["rank"] = self.global_rank
        return result

    ModelRunner.load_model = load_model

    forward = Qwen3_5ForConditionalGeneration.forward
    signature = inspect.signature(forward)

    def model(self, *args, **kwargs):
        active = json.loads((directory / "active.json").read_text())
        if not active["capture"]:
            captures["ctx"] = None
            return forward(self, *args, **kwargs)
        bound = signature.bind(self, *args, **kwargs).arguments
        ctx = bound["ctx"]
        captures["ctx"] = ctx
        captures["state"] = None
        result = forward(self, *args, **kwargs)
        name = active["name"]
        step = counter.get(name, 0)
        counter[name] = step + 1
        rank = captures["rank"]
        path = directory / f"rank{rank}" / name
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                positions=bound["positions"].detach().cpu(),
                logits=result.next_token_logits.detach().cpu(),
                state=captures["state"],
            ),
            path / f"{step:03d}.pt",
        )
        return result

    Qwen3_5ForConditionalGeneration.forward = model
    layer_forward = Qwen3_5LinearDecoderLayer.forward

    def layer(self, *args, **kwargs):
        result = layer_forward(self, *args, **kwargs)
        if self.layer_id == 0 and captures.get("ctx") is not None:
            ctx = captures["ctx"]
            backend = ctx.attn_backend.linear_attn_backend
            _, outgoing, conv, ssm = backend._layer_state(0)
            indices = outgoing[: ctx.bs].long()
            captures["state"] = dict(
                conv=conv.index_select(0, indices).detach().cpu(),
                ssm=ssm.index_select(0, indices).detach().cpu(),
            )
        return result

    Qwen3_5LinearDecoderLayer.forward = layer
