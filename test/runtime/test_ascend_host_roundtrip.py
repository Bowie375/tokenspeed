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

"""Byte-exact DMA round trips through real compact mixed-state Host storage."""

import pytest
import torch

from tokenspeed.runtime.cache.l2.storage import HostCacheStorage
from tokenspeed.runtime.cache.transfer.lanes import (
    HostTransferLane,
    build_transfer_geometry,
)
from tokenspeed.runtime.cache.transfer.layout import (
    CacheField,
    CacheGroupLayout,
    CacheTransferLayout,
)


@pytest.mark.parametrize("backend", ["dma", "auto"])
def test_ascend_mixed_fields_roundtrip_and_page_reuse(backend):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires an Ascend chip")
    # Include padding and distinct block strides so a contiguous bulk copy is wrong.
    buffers = tuple(
        torch.arange(6 * stride, dtype=torch.int32).to("npu").view(torch.uint8)
        for stride in (40, 48, 56)
    )
    layout = CacheTransferLayout(
        5,
        (
            CacheGroupLayout(
                "history", 1, (CacheField("layer.0.key", 0, 0, 160, 128),)
            ),
            CacheGroupLayout(
                "state",
                2,
                (
                    CacheField("layer.1.conv", 1, 0, 96, 64),
                    CacheField("layer.1.ssm", 2, 0, 112, 80),
                ),
            ),
        ),
        buffers,
        (("layer.0.key",), ("layer.1.conv", "layer.1.ssm")),
    )
    storage = HostCacheStorage(layout, num_host_lcm_blocks=4)
    geometry = build_transfer_geometry(layout, storage, io_backend="direct")
    write, load = torch.npu.Stream(), torch.npu.Stream()
    d2h, h2d = HostTransferLane(), HostTransferLane()
    saved = [buffer.cpu().clone() for buffer in buffers]
    for cycle in range(4):
        # State child 3 and history child 1 occupy different Host parents.
        rows = [(0, 1, 1), (1, 1, 3)]
        finish = d2h.start_d2h(
            rows,
            device_buffers=buffers,
            host_buffer=storage.host_buffer,
            geometry=geometry,
            stream=write,
            prerequisite_stream=torch.npu.current_stream(),
            backend=backend,
        )
        torch.npu.current_stream().wait_event(finish)
        # Device sources are now safely reclaimed and overwritten by another owner.
        for buffer in buffers:
            buffer.fill_(cycle + 199)
        untouched = [buffer.cpu().clone() for buffer in buffers]
        restored = h2d.start_h2d(
            [(0, 2, 1), (1, 2, 3)],
            device_buffers=buffers,
            host_buffer=storage.host_buffer,
            geometry=geometry,
            stream=load,
            prerequisite_stream=torch.npu.current_stream(),
            backend=backend,
        )
        torch.npu.current_stream().wait_event(restored)
        for group in layout.groups:
            for field in group.fields:
                actual = buffers[field.device_buffer_index].cpu()
                start = 2 * field.block_stride_bytes
                expected = untouched[field.device_buffer_index]
                expected[start : start + field.payload_bytes] = saved[
                    field.device_buffer_index
                ][
                    field.block_stride_bytes : field.block_stride_bytes
                    + field.payload_bytes
                ]
                assert torch.equal(actual, expected)
        # Restore the original source for the next metadata-reuse cycle.
        for buffer, original in zip(buffers, saved):
            buffer.copy_(original)
    empty = h2d.start_h2d(
        [],
        device_buffers=buffers,
        host_buffer=storage.host_buffer,
        geometry=geometry,
        stream=load,
        prerequisite_stream=torch.npu.current_stream(),
        backend=backend,
    )
    empty.synchronize()
