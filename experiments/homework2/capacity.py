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


"""Check logical cache admission with the runtime recipe before loading weights."""

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.qwen35 import QwenGDNRecipe
from tokenspeed.runtime.layers.attention.kv_cache.recipes.scheduler_bridge import (
    capacity_model,
)
from tokenspeed.runtime.layers.attention.registry import _create_attn_config
from tokenspeed.runtime.utils.server_args import ServerArgs


def check_cache_capacity(launch: dict, *, requests: int, request_tokens: int) -> dict:
    """Reject logical token limits that cannot hold the requested real batch.

    This host-only calculation uses the scheduler's actual group-page rules.
    The deliberately large planning budget isolates the configured token cap;
    it does not predict available HBM or allocate any device tensors.
    """
    if requests < 1 or request_tokens < 1:
        raise ValueError("request count and token count must be positive")
    args = ServerArgs(**launch)
    args.mapping.rank = 0
    model = ModelConfig(
        args.model,
        model_override_args="{}",
        dtype=args.dtype,
        context_length=args.max_model_len,
        server_args=args,
    )
    recipe = QwenGDNRecipe(
        server_args=args,
        model_config=model,
        attn_config=_create_attn_config(args, model),
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=2**40,
        probe_batch_rows=None,
        decode_input_tokens=1,
        overlap_schedule_depth=1,
    )
    setup = recipe.setup()
    plan = setup.spec.memory_plan
    capacity = capacity_model(
        setup.spec.cache_group_specs,
        prefix_granularity=plan.prefix_granularity,
        virtual_packing={g.group_id: g.cache_blocks_per_lcm_block for g in plan.groups},
        limits=recipe.scheduler_limits,
    )
    needed = capacity.lcm_blocks_needed_for(
        [
            requests * count
            for count in capacity.single_request_group_pages(token_limit=request_tokens)
        ]
    )
    result = dict(
        usable_parents=plan.num_lcm_blocks,
        required_parents=needed,
        parent_bytes=plan.lcm_block_bytes,
        max_single_request_tokens=capacity.max_single_request_tokens(
            usable_lcm_blocks=plan.num_lcm_blocks
        ),
        requests=requests,
        request_tokens=request_tokens,
        scope="logical token-cap admission; available HBM checked by the runtime",
    )
    if plan.num_lcm_blocks < needed:
        raise ValueError(
            f"cache cap {args.max_total_tokens} provides {plan.num_lcm_blocks} parents, "
            f"but {requests} requests need {needed}; increase capacity"
        )
    return result
