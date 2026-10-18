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

"""Ascend registrations for the shared GDN prefill and decode APIs."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_npu:
    from tokenspeed_kernel.ops.attention.gdn import GdnChunkPrefillResult
    from tokenspeed_kernel_npu.ops.gdn import gdn_decode, gdn_prefill

    @register_kernel(
        "attention",
        "gdn_chunk_prefill",
        name="ascend_gdn_chunk_prefill",
        solution="triton",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=format_signatures(
            ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16, torch.float32}
        ),
        priority=Priority.PORTABLE,
        traits={
            "output_h": frozenset({False}),
            "qk_l2norm": frozenset({False, True}),
            "num_v_gte_num_q": frozenset({True}),
        },
    )
    def ascend_gdn_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        *,
        scale,
        initial_state,
        cu_seqlens,
        qk_l2norm,
        output_final_state,
        output_h,
    ) -> GdnChunkPrefillResult:
        """Adapt the Ascend FP32 recurrence to the shared prefill result."""
        if output_h:
            raise NotImplementedError(
                "Ascend GDN does not support intermediate checkpoints"
            )
        out, final = gdn_prefill(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            qk_l2norm=qk_l2norm,
            output_final_state=output_final_state,
        )
        return GdnChunkPrefillResult(out=out, final_state=final)

    @register_kernel(
        "attention",
        "gdn_decode_step",
        name="ascend_gdn_decode_step",
        solution="triton",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=format_signatures(
            ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16, torch.float32}
        ),
        priority=Priority.PORTABLE,
    )
    def ascend_gdn_decode_step(
        q,
        k,
        v,
        *,
        A_log,
        a,
        dt_bias,
        b,
        initial_state,
        initial_state_indices,
        scale,
        output_state_indices,
        use_qk_l2norm,
    ) -> torch.Tensor:
        """Execute one Ascend decode step through the shared indexed-state ABI."""
        return gdn_decode(
            q,
            k,
            v,
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            b=b,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            scale=scale,
            output_state_indices=output_state_indices,
            use_qk_l2norm=use_qk_l2norm,
        )
