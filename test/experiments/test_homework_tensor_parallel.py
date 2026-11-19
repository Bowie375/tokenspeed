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


"""Behavioral tests for TP acceptance, layout reconstruction, and timing."""

from pathlib import Path

import pytest
import torch

from experiments.homework2.collectives import bandwidth_gbps
from experiments.homework2.validation import (
    compare_greedy,
    merge_conv_states,
    validate_steps,
)


def test_divergence_only_accepts_one_ulp_reference_top_five():
    ref = torch.tensor([20.5, 20.5, 18.0, 17.0, 16.0, 15.0])
    a = [dict(sampled_token=1, logits=ref)]
    b = [dict(sampled_token=0, logits=ref)]
    result = compare_greedy(a, b, allow_bf16_ties=True)
    assert result["status"] == "bf16_tie" and result["first_difference"] == 0
    assert compare_greedy(a, b, allow_bf16_ties=False)["status"] == "failed"
    a[0]["sampled_token"] = 2
    assert compare_greedy(a, b, allow_bf16_ties=True)["status"] == "failed"


def test_missing_decode_steps_and_misaligned_positions_fail():
    rows = [
        dict(
            step=0,
            input_ids=[4, 4],
            positions=torch.tensor([[0, 1]]).expand(3, -1),
            cached_decode=False,
        )
    ]
    assert not validate_steps(rows, input_ids=[4, 4], steps=2)["passed"]
    rows.append(
        dict(
            step=1,
            input_ids=[7],
            positions=torch.tensor([[2]]).expand(3, -1),
            cached_decode=True,
        )
    )
    assert validate_steps(rows, input_ids=[4, 4], steps=2)["passed"]
    rows[1]["positions"] = rows[1]["positions"] + 1
    assert not validate_steps(rows, input_ids=[4, 4], steps=2)["passed"]


def test_conv_rank_merge_preserves_qkv_order_with_unequal_value_width():
    q, k, v = (
        torch.arange(8).reshape(4, 2),
        torch.arange(8, 16).reshape(4, 2),
        torch.arange(16, 40).reshape(12, 2),
    )
    shards = [
        torch.cat([q.chunk(2)[r], k.chunk(2)[r], v.chunk(2)[r]]) for r in range(2)
    ]
    actual = merge_conv_states(shards, key_channels=4, value_channels=12)
    torch.testing.assert_close(actual, torch.cat([q, k, v]), rtol=0, atol=0)


def test_greedy_incomplete_or_nonfinite_comparison_is_not_accepted():
    row = dict(sampled_token=0, logits=torch.tensor([1.0, 0.0]))
    assert compare_greedy([row], [], allow_bf16_ties=True)["status"] == "failed"
    bad = dict(sampled_token=1, logits=torch.tensor([float("nan"), 0.0]))
    assert compare_greedy([bad], [row], allow_bf16_ties=True)["status"] == "failed"


def test_payload_bandwidth_uses_bytes_and_decimal_gigabytes():
    assert bandwidth_gbps(payload_bytes=2048, latency_us=250) == pytest.approx(0.008192)
    with pytest.raises(ValueError):
        bandwidth_gbps(payload_bytes=2048, latency_us=0)


def test_free_tp_rank_disagreement_is_never_a_rounding_tie():
    from experiments.homework2.validation import validate_rank_agreement

    row = dict(sampled_token=0, input_ids=[42], positions=torch.tensor([[0]]))
    peer = dict(row, sampled_token=1)
    assert not validate_rank_agreement([row], [peer])["passed"]
    assert validate_rank_agreement([row], [dict(row)])["passed"]
    peer = dict(row, input_ids=[43])
    assert not validate_rank_agreement([row], [peer])["passed"]


def test_token_ttft_includes_special_tokens_and_rejects_coalesced_tpot():
    from experiments.homework2.measurements import TokenMeasurements

    stream = TokenMeasurements(start=10)
    stream.observe(10.1, dict(text="", meta_info=dict(completion_tokens=1)))
    stream.observe(
        10.3,
        dict(
            text="hello",
            meta_info=dict(completion_tokens=2, finish_reason=dict(type="length")),
        ),
    )
    result = stream.finish(end=10.4, target_tokens=2, eos_policy="fixed_length")
    assert result["valid"]
    assert result["client_ttft_ms"] == pytest.approx(100)
    assert result["client_first_text_ms"] == pytest.approx(300)
    coalesced = TokenMeasurements(start=10)
    coalesced.observe(
        10.3,
        dict(
            text="hi",
            meta_info=dict(completion_tokens=2, finish_reason=dict(type="length")),
        ),
    )
    assert not coalesced.finish(end=10.4, target_tokens=2, eos_policy="fixed_length")[
        "valid"
    ]


@pytest.mark.parametrize("exit_early", [True, False])
def test_worker_cleanup_terminates_descendants_even_after_parent_exit(
    tmp_path, exit_early
):
    import os
    import signal
    import subprocess
    import sys
    import time

    from experiments.homework2.run import execute

    pidfile = tmp_path / "child.json"
    child = "import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print(os.getpid(),os.getpgrp(),flush=True); time.sleep(60)"
    source = (
        "import subprocess,sys,time; "
        f"f=open({str(pidfile)!r},'w'); "
        f"p=subprocess.Popen([sys.executable,'-c',{child!r}],stdout=f); "
        "time.sleep(0.4); " + ("sys.exit(3)" if exit_early else "time.sleep(60)")
    )
    try:
        with pytest.raises((RuntimeError, subprocess.TimeoutExpired)):
            execute(
                [sys.executable, "-c", source],
                directory=tmp_path,
                env=os.environ.copy(),
                timeout=1,
                root=tmp_path,
            )
        pid, pgid = map(int, pidfile.read_text().split())
        for _ in range(20):
            stat = Path(f"/proc/{pid}/stat")
            if not stat.exists() or stat.read_text().split()[2] == "Z":
                break
            time.sleep(0.05)
        else:
            pytest.fail("worker descendant survived cleanup")
    finally:
        if pidfile.exists() and pidfile.read_text().strip():
            _, pgid = map(int, pidfile.read_text().split())
            assert pgid != os.getpgrp()
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_comparable_performance_rejects_resource_mismatch():
    from experiments.homework2.validation import validate_performance_resources

    config = dict(
        context_length=4096, capacity=16384, chunked_prefill_size=1024, max_num_seqs=4
    )
    validate_performance_resources(config, {})
    validate_performance_resources(config, dict(config))
    with pytest.raises(ValueError, match="capacity"):
        validate_performance_resources(config, dict(capacity=8192))


def test_performance_requires_every_case_mode_repetition_and_request():
    from experiments.homework2.validation import validate_performance

    config = dict(
        workloads=[
            dict(input_tokens=128, output_tokens=32),
            dict(input_tokens=512, output_tokens=16),
        ],
        repetitions=2,
        concurrency=2,
    )
    records = []
    for case in config["workloads"]:
        for parallel in (False, True):
            samples = [
                dict(
                    valid=True,
                    actual_output_tokens=case["output_tokens"],
                    client_ttft_ms=1,
                    client_tpot_ms=1,
                    e2e_ms=10,
                )
            ] * 2
            records.append(
                dict(
                    case=case,
                    parallel=parallel,
                    trials=[
                        dict(
                            repetition=i,
                            requests=list(samples),
                            wall_seconds=1,
                            throughput_tokens_per_second=2 * case["output_tokens"],
                        )
                        for i in range(2)
                    ],
                )
            )
    assert validate_performance(records, config)["passed"]
    assert not validate_performance(records[:-1], config)["passed"]
    records[0]["trials"][0]["requests"].pop()
    assert not validate_performance(records, config)["passed"]


def test_27b_logical_cache_cap_admits_the_actual_state_groups(tmp_path):
    from experiments.homework2.capacity import check_cache_capacity
    from tokenspeed.runtime.configs.qwen3_5_config import (
        Qwen3_5Config,
        Qwen3_5TextConfig,
    )

    text = Qwen3_5TextConfig(
        hidden_size=5120,
        intermediate_size=17408,
        num_hidden_layers=64,
        full_attention_interval=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]
        * 16,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        vocab_size=248320,
    )
    config = Qwen3_5Config(text_config=text.to_dict())
    config.architectures = ["Qwen3_5ForConditionalGeneration"]
    config.save_pretrained(tmp_path)
    launch = dict(
        model=str(tmp_path),
        world_size=1,
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
        max_model_len=512,
        max_num_seqs=2,
        max_total_tokens=8192,
        max_prefill_tokens=512,
        chunked_prefill_size=512,
        mamba_ssm_dtype="float32",
    )
    with pytest.raises(ValueError, match="provides 5 parents"):
        check_cache_capacity(launch, requests=1, request_tokens=273)
    launch["max_total_tokens"] = 16384
    with pytest.raises(ValueError, match="2 requests need 13"):
        check_cache_capacity(launch, requests=2, request_tokens=273)
    launch["max_total_tokens"] = 32768
    result = check_cache_capacity(launch, requests=2, request_tokens=273)
    assert result["required_parents"] <= result["usable_parents"]
    assert result["max_single_request_tokens"] >= 273


def test_performance_symlink_cannot_bypass_unequal_resource_guard(tmp_path):
    import subprocess
    import sys

    checkpoint = tmp_path / "Qwen3.5-27B"
    checkpoint.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(checkpoint, target_is_directory=True)
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.homework2.run",
            "--model",
            str(alias),
            "--devices",
            "0,1",
            "--config",
            str(root / "experiments/homework2/configs/baseline.json"),
            "--task",
            "performance",
            "--reference",
            "none",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "equal" in result.stderr
    assert "config.json" not in result.stderr


def test_worker_orders_visible_chips_without_changing_first_tp1_chip(tmp_path):
    from experiments.homework2.run import worker_environment

    single = tmp_path / "single"
    pair = tmp_path / "pair"
    single.mkdir()
    pair.mkdir()
    root = Path(__file__).resolve().parents[2]
    assert worker_environment(root, single, ["1"])["ASCEND_RT_VISIBLE_DEVICES"] == "1"
    assert (
        worker_environment(root, pair, ["1", "0"])["ASCEND_RT_VISIBLE_DEVICES"] == "0,1"
    )


def test_reanalysis_refuses_incomplete_native_workers_without_marking_success(tmp_path):
    import json

    from experiments.homework2.reanalyze import reanalyze

    (tmp_path / "tp1").mkdir()
    (tmp_path / "tp1/worker_status.json").write_text(json.dumps(dict(completed=False)))
    original = dict(completed=False, error="native worker failed")
    (tmp_path / "status.json").write_text(json.dumps(original))
    with pytest.raises(ValueError, match="native worker did not complete"):
        reanalyze(tmp_path, reference="none", communication="unavailable")
    assert json.loads((tmp_path / "status.json").read_text()) == original


def test_missing_model_communication_evidence_is_a_failure(tmp_path):
    import json

    from experiments.homework2.communication import validate_model_communication

    (tmp_path / "config.json").write_text(json.dumps(dict(steps=1)))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            dict(
                model_config=dict(
                    num_hidden_layers=24,
                    hidden_size=1024,
                    vocab_size=248320,
                )
            )
        )
    )
    (tmp_path / "fixtures.json").write_text(
        json.dumps([dict(name="raw_1", input_ids=[42])])
    )
    result = validate_model_communication(tmp_path)
    assert not result["passed"]
    assert all(not check["passed"] for check in result["checks"])


def test_performance_requires_memory_from_every_real_rank(tmp_path):
    from experiments.homework2.performance import validate_rank_memory

    result = validate_rank_memory(tmp_path, world_size=2)
    assert not result["passed"] and len(result["errors"]) == 2


def test_invalid_stream_timings_produce_validation_artifact(tmp_path):
    import json

    from experiments.homework2.performance import analyze_performance

    config = dict(
        workloads=[dict(input_tokens=128, output_tokens=32)],
        repetitions=1,
        concurrency=1,
        context_length=1024,
        capacity=32768,
        chunked_prefill_size=512,
        max_num_seqs=4,
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    records = [
        dict(
            case=config["workloads"][0],
            parallel=parallel,
            trials=[
                dict(
                    repetition=0,
                    wall_seconds=1,
                    throughput_tokens_per_second=32,
                    requests=[
                        dict(
                            valid=False,
                            actual_output_tokens=32,
                            client_ttft_ms=1,
                            client_tpot_ms=None,
                            e2e_ms=1000,
                        )
                    ],
                )
            ],
        )
        for parallel in (False, True)
    ]
    for tp in (1, 2):
        rd = tmp_path / f"tp{tp}"
        rd.mkdir()
        (rd / "config.json").write_text(json.dumps(config))
        (rd / "performance.json").write_text(json.dumps(records))
    result = analyze_performance(tmp_path)
    assert not result["passed"]
    saved = json.loads((tmp_path / "validation.json").read_text())
    assert not saved["checks"]["tp1_requests"]["passed"]
    summary = json.loads((tmp_path / "performance_summary.json").read_text())
    assert "skipped" in summary["tp1"]
