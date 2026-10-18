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

"""Ascend causal-convolution registrations."""

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_npu:
    from tokenspeed_kernel_npu.ops.causal_conv1d import (
        ascend_causal_conv1d_prefill,
        ascend_causal_conv1d_update,
    )

    for mode, function in (
        ("causal_conv1d_prefill", ascend_causal_conv1d_prefill),
        ("causal_conv1d_update", ascend_causal_conv1d_update),
    ):
        register_kernel(
            "attention",
            mode,
            name=function.__name__,
            solution="triton",
            capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
            signatures=format_signatures(
                "x", "dense", {torch.float16, torch.bfloat16, torch.float32}
            ),
            priority=Priority.PORTABLE,
            traits={"tree": frozenset({False})} if mode.endswith("update") else {},
        )(function)
