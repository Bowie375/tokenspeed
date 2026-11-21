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

"""Authoritative Host capacity through real scheduler publication and ACKs."""

import tokenspeed_scheduler as ts

from tokenspeed.runtime.engine.scheduler_utils import (
    advance_scheduler,
    make_config,
    make_extend_result_event,
    make_finish_event,
)


def make_scheduler():
    cfg = make_config(
        num_device_pages=17,
        max_scheduled_tokens=64,
        max_batch_size=2,
        prefix_granularity=16,
        num_host_pages=9,
        disable_l2_cache=False,
        enable_l3_storage=False,
        role="fused",
        num_snapshot_pages=1,
        max_retracted_requests=0,
        l3_prefetch_min_pages=0,
        debug_force_retraction_interval=0,
        cache_groups=[
            ts.CacheGroupConfig(
                group_id="history",
                block_granularity=16,
                total_pages=17,
                attn_kind=ts.AttnKind.Full,
            )
        ],
    )
    return ts.Scheduler(cfg)


def test_host_pool_gauges_follow_publication_ack_and_clear():
    scheduler = make_scheduler()
    assert scheduler.host_pool_free_blocks() == 8
    assert scheduler.host_pool_cached_blocks() == 0
    spec = ts.RequestSpec()
    spec.request_id = "a"
    spec.tokens = list(range(32))
    spec.max_new_tokens = 4
    scheduler.submit_requests([spec])
    scheduler.next_execution_plan()
    advance_scheduler(scheduler, [make_extend_result_event("a", [42])])
    advance_scheduler(scheduler, [make_finish_event("a")])
    saw_write = False
    for _ in range(12):
        plan = scheduler.next_execution_plan()
        events = []
        for op in plan.cache:
            if isinstance(op, ts.Cache.WriteBackOp):
                saw_write = True
                assert scheduler.host_pool_free_blocks() < 8
                for op_id in op.op_ids:
                    event = ts.Cache.WriteBackDoneEvent()
                    event.op_id = op_id
                    events.append(event)
        if events:
            advance_scheduler(scheduler, events)
    assert saw_write
    assert scheduler.host_pool_cached_blocks() >= 2
    assert scheduler.host_pool_free_blocks() < 8
    assert scheduler.host_pool_pinned_blocks() == 0
    assert scheduler.clear_cache()
    assert scheduler.host_pool_free_blocks() == 8
    assert scheduler.host_pool_cached_blocks() == 0


def test_host_budget_evictions_are_monotonic_and_flush_is_not_an_eviction():
    scheduler = make_scheduler()
    previous = 0
    for index in range(12):
        spec = ts.RequestSpec()
        spec.request_id = f"session{index}"
        spec.tokens = [index + 1] * 32
        spec.max_new_tokens = 4
        scheduler.submit_requests([spec])
        scheduler.next_execution_plan()
        advance_scheduler(scheduler, [make_extend_result_event(spec.request_id, [42])])
        advance_scheduler(scheduler, [make_finish_event(spec.request_id)])
        for _ in range(6):
            plan = scheduler.next_execution_plan()
            events = []
            for op in plan.cache:
                if isinstance(op, ts.Cache.WriteBackOp):
                    for op_id in op.op_ids:
                        event = ts.Cache.WriteBackDoneEvent()
                        event.op_id = op_id
                        events.append(event)
            if events:
                advance_scheduler(scheduler, events)
        current = scheduler.host_cache_evictions()
        assert current >= previous
        assert 0 <= scheduler.host_pool_free_blocks() <= 8
        assert scheduler.host_pool_pinned_blocks() == 0
        previous = current
    assert previous > 0
    assert scheduler.clear_cache()
    assert scheduler.host_cache_evictions() == previous
    assert scheduler.host_pool_free_blocks() == 8
