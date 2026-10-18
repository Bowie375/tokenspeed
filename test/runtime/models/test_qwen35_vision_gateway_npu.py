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

"""Opt-in local-checkpoint image serving and first-token reference validation.

Run with QWEN35_VISION_E2E=1; artifacts include the exact prompts, logits,
responses and server timing log. One visible, otherwise unused NPU is required.
"""

import base64
import io
import json
import os
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests
import torch

pytest.importorskip("torch_npu")
from test_qwen35_vision_npu import fixed_images
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

pytestmark = pytest.mark.skipif(
    os.environ.get("QWEN35_VISION_E2E") != "1" or not torch.npu.is_available(),
    reason="set QWEN35_VISION_E2E=1 on a free Ascend chip",
)


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def gateway(tmp_path_factory):
    root = Path(__file__).resolve().parents[3]
    path = Path(
        os.environ.get("QWEN35_VISION_MODEL", root / "outputs/models/Qwen3.5-0.8B")
    )
    if not (path / "config.json").exists():
        pytest.skip("set QWEN35_VISION_MODEL to the pinned checkpoint")
    artifact = Path(
        os.environ.get("QWEN35_VISION_ARTIFACTS", root / "outputs/vision_validation")
    )
    artifact.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.update(
        TOKENSPEED_MM_ENABLE_ENCODER_CUDA_GRAPH="0",
        TOKENSPEED_LOG_MM_TIMING="1",
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
        GRPC_ENABLE_HTTP_PROXY="0",
        OMP_NUM_THREADS="8",
    )
    # The serving API returns only the sampled token score. Capture the real
    # logits once in the test subprocess without changing model computation.
    capture_dir = tmp_path_factory.mktemp("vision_logits_capture")
    capture_dir.joinpath("sitecustomize.py").write_text(
        "import os,json\nfrom pathlib import Path\n"
        "import torch\n"
        "from tokenspeed.runtime.layers.logits_processor import LogitsProcessor\n"
        "_forward = LogitsProcessor.forward\n"
        "_memory_peak = [0,0]\n"
        "def capture(self,input_ids,*args,**kwargs):\n"
        "    output = _forward(self,input_ids,*args,**kwargs)\n"
        "    target = Path(os.environ['QWEN35_LOGITS_CAPTURE'])\n"
        "    if input_ids.numel() == int(os.environ['QWEN35_LOGITS_INPUT_LENGTH']) and not target.exists():\n"
        "        torch.save({'input_ids':input_ids.cpu(),'logits':output.next_token_logits.detach().cpu()},target)\n"
        "    device = torch.get_device_module(input_ids.device)\n"
        "    peak = [device.max_memory_allocated(),device.max_memory_reserved()]\n"
        "    if peak != _memory_peak:\n"
        "        _memory_peak[:] = peak\n"
        "        target.with_name('gateway_memory.json').write_text(json.dumps({'peak_allocated_bytes':peak[0],'peak_reserved_bytes':peak[1]}))\n"
        "    return output\n"
        "LogitsProcessor.forward = capture\n"
    )
    capture_path = artifact / "ascend_first_logits.pt"
    capture_path.unlink(missing_ok=True)

    env["QWEN35_LOGITS_CAPTURE"] = str(capture_path)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    capture_prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": fixed_images()["color"]},
                {"type": "text", "text": "Describe this image in one short sentence."},
            ],
        }
    ]
    capture_inputs = processor.apply_chat_template(
        capture_prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    env["QWEN35_LOGITS_INPUT_LENGTH"] = str(capture_inputs.input_ids.numel())
    env["PYTHONPATH"] = f"{capture_dir}:{env.get('PYTHONPATH','')}"
    cmd = [
        sys.executable,
        "-m",
        "tokenspeed.cli",
        "serve",
        "--device",
        "npu",
        "--model",
        str(path),
        "--served-model-name",
        "qwen35-vision-npu",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--world-size",
        "1",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "2",
        "--max-prefill-tokens",
        "4096",
        "--chunked-prefill-size",
        "4096",
        "--gpu-memory-utilization",
        "0.2",
        "--attention-backend",
        "mha",
        "--sampling-backend",
        "greedy",
        "--mm-attention-backend",
        "ascend",
        "--policy",
        "random",
        "--disable-kvstore",
        "--disable-prefill-graph",
        "--disable-prefix-caching",
        "--disable-pdl",
        "--disable-autotune",
        "--enforce-eager",
        "--prefix-granularity",
        "128",
        "--dtype",
        "bfloat16",
        "--enable-output-logprobs",
    ]
    (artifact / "launch.json").write_text(json.dumps(cmd, indent=2) + "\n")
    with (artifact / "gateway.log").open("w") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 300
            session = requests.Session()
            session.trust_env = False
            while time.monotonic() < deadline:
                assert proc.poll() is None, (artifact / "gateway.log").read_text()[
                    -6000:
                ]
                try:
                    if session.get(url + "/readiness", timeout=2).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                pytest.fail((artifact / "gateway.log").read_text()[-6000:])
            yield url, path, artifact
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=10)


def _payload(images, prompt):
    content = []
    for image in images:
        data = io.BytesIO()
        image.save(data, format="PNG")
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,"
                    + base64.b64encode(data.getvalue()).decode("ascii")
                },
            }
        )
    content.append({"type": "text", "text": prompt})
    return {
        "model": "qwen35-vision-npu",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 16,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _request(url, payload):
    session = requests.Session()
    session.trust_env = False
    started = time.perf_counter()
    response = session.post(url + "/v1/chat/completions", json=payload, timeout=120)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["choices"][0]["message"]["content"].strip(), result
    assert result["usage"]["completion_tokens"] > 0, result
    return {"response": result, "elapsed_seconds": time.perf_counter() - started}


def test_gateway_image_variations_and_first_token_scores(gateway):
    url, path, artifact = gateway
    images = fixed_images()
    prompt = "Describe this image in one short sentence."
    payload = _payload([images["color"]], prompt)
    results = {"color": _request(url, payload)}
    results["rectangle"] = _request(url, _payload([images["rectangle"]], prompt))
    results["text_image"] = _request(
        url, _payload([images["text"]], "Read the text in this image.")
    )
    results["two_images"] = _request(
        url, _payload([images["color"], images["rectangle"]], "Describe both images.")
    )
    results["repeat"] = _request(url, payload)
    assert (
        results["repeat"]["response"]["choices"][0]["message"]
        == results["color"]["response"]["choices"][0]["message"]
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        pairs = list(
            pool.map(
                lambda p: _request(url, p), [payload, _payload([], "What is 2 + 2?")]
            )
        )
    results["mixed_image"], results["mixed_text"] = pairs
    with ThreadPoolExecutor(max_workers=2) as pool:
        pairs = list(
            pool.map(
                lambda p: _request(url, p),
                [payload, _payload([images["rectangle"]], prompt)],
            )
        )
    results["concurrent_color"], results["concurrent_rectangle"] = pairs
    assert (
        "red"
        in results["concurrent_color"]["response"]["choices"][0]["message"][
            "content"
        ].lower()
    )
    rectangle_text = results["concurrent_rectangle"]["response"]["choices"][0][
        "message"
    ]["content"].lower()
    assert "blue" in rectangle_text and "yellow" in rectangle_text, rectangle_text
    (artifact / "gateway_results.json").write_text(json.dumps(results, indent=2) + "\n")
    torch.set_num_threads(8)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    processor_counts = {}
    cases = {
        "color": ([images["color"]], prompt),
        "rectangle": ([images["rectangle"]], prompt),
        "text_image": ([images["text"]], "Read the text in this image."),
        "two_images": ([images["color"], images["rectangle"]], "Describe both images."),
    }
    for name, (case_images, case_prompt) in cases.items():
        case_messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image} for image in case_images]
                + [{"type": "text", "text": case_prompt}],
            }
        ]
        case_inputs = processor.apply_chat_template(
            case_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        assert (
            results[name]["response"]["usage"]["prompt_tokens"]
            == case_inputs.input_ids.numel()
        )
        processor_counts[name] = {
            "prompt_tokens": case_inputs.input_ids.numel(),
            "visual_tokens": int((case_inputs.image_grid_thw.prod(-1) // 4).sum()),
            "grid": case_inputs.image_grid_thw.tolist(),
        }
    (artifact / "gateway_processor_counts.json").write_text(
        json.dumps(processor_counts, indent=2) + "\n"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": images["color"]},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    assert (
        results["color"]["response"]["usage"]["prompt_tokens"]
        == inputs.input_ids.shape[1]
    )
    reference = Qwen3_5ForConditionalGeneration.from_pretrained(
        path, local_files_only=True, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    with torch.inference_mode():
        logits = (
            reference(**inputs, logits_to_keep=1, use_cache=False).logits[0, -1].cpu()
        )
    first = results["color"]["response"]["choices"][0]["logprobs"]["content"][0]
    expected_logprobs = logits.log_softmax(-1)
    comparisons = []
    for entry in [first]:
        token_ids = processor.tokenizer.encode(entry["token"], add_special_tokens=False)
        assert len(token_ids) == 1, entry
        token_id = token_ids[0]
        expected = float(expected_logprobs[token_id])
        comparisons.append(
            {
                "token_id": token_id,
                "token": entry["token"],
                "actual": entry["logprob"],
                "reference": expected,
                "absolute_error": abs(entry["logprob"] - expected),
            }
        )
    torch.save(
        {"inputs": dict(inputs), "logits": logits},
        artifact / "reference_first_logits.pt",
    )
    (artifact / "first_token_comparison.json").write_text(
        json.dumps(comparisons, indent=2) + "\n"
    )
    # Full BF16 model scores are compared after normalization; logits have an
    # arbitrary common offset. Keep top-token agreement separate from captions.
    assert max(row["absolute_error"] for row in comparisons) < 0.25, comparisons
    reference_top = set(expected_logprobs.topk(10).indices.tolist())
    assert all(row["token_id"] in reference_top for row in comparisons), comparisons
    captured = torch.load(artifact / "ascend_first_logits.pt", weights_only=True)
    actual_ids = captured["input_ids"].to(torch.int64)
    actual_ids[inputs.input_ids[0] == reference.config.image_token_id] = (
        reference.config.image_token_id
    )
    torch.testing.assert_close(actual_ids, inputs.input_ids[0], rtol=0, atol=0)
    actual_logits = captured["logits"][0].float()[: len(logits)]
    assert actual_logits.shape == logits.shape
    centered_actual = actual_logits - actual_logits.mean()
    centered_expected = logits - logits.mean()
    error = centered_actual - centered_expected
    metrics = {
        "relative_l2": float(error.norm() / centered_expected.norm()),
        "rmse": float(error.square().mean().sqrt()),
        "max_abs": float(error.abs().max()),
        "top10_overlap": len(
            set(actual_logits.topk(10).indices.tolist()) & reference_top
        ),
    }
    (artifact / "full_logits_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    assert (
        metrics["relative_l2"] < 0.05
        and metrics["rmse"] < 0.15
        and metrics["max_abs"] < 0.6
    ), metrics
    assert metrics["top10_overlap"] >= 8, metrics
