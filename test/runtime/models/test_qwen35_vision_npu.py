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

"""Pinned Qwen3.5 vision checkpoint comparisons on Ascend versus HF CPU FP32."""

import copy
import json
import os
from pathlib import Path

import pytest
import torch

pytest.importorskip("torch_npu")
pytest.importorskip("PIL")

from PIL import Image, ImageDraw, ImageFont
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

from tokenspeed.runtime.configs.qwen3_5_config import Qwen3_5Config
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.model_loader.utils import set_default_torch_dtype
from tokenspeed.runtime.models.qwen3_5 import Qwen3_5ForConditionalGeneration

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires Ascend")


def fixed_images():
    color = Image.new("RGB", (256, 256), (220, 20, 20))
    rectangle = Image.new("RGB", (384, 224), (20, 60, 200))
    ImageDraw.Draw(rectangle).rectangle((192, 0, 383, 223), fill=(240, 200, 20))
    text = Image.new("RGB", (512, 128), "white")
    ImageDraw.Draw(text).text(
        (20, 30), "HELLO 123", font=ImageFont.load_default(size=48), fill="black"
    )
    return {"color": color, "rectangle": rectangle, "text": text}


@pytest.fixture(scope="module")
def pinned_vision():
    root = Path(__file__).resolve().parents[3]
    path = Path(
        os.environ.get("QWEN35_VISION_MODEL", root / "outputs/models/Qwen3.5-0.8B")
    )
    if not (path / "config.json").exists():
        pytest.skip("set QWEN35_VISION_MODEL to the pinned checkpoint")
    torch.set_num_threads(8)
    config = Qwen3_5Config.from_pretrained(path, local_files_only=True)
    config.encoder_only = True
    assert (
        config.vision_config.depth == 12
        and config.vision_config.deepstack_visual_indexes == []
    )
    reference_config = copy.deepcopy(
        AutoConfig.from_pretrained(path, local_files_only=True).vision_config
    )
    reference_config._attn_implementation = "eager"
    with torch.device("cpu"):
        reference = Qwen3_5VisionModel(reference_config).eval().float()
    weights = {}
    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as checkpoint:
            for name in checkpoint.keys():
                if name.startswith("model.visual."):
                    weights[name] = checkpoint.get_tensor(name)
    reference.load_state_dict(
        {name.removeprefix("model.visual."): value for name, value in weights.items()},
        strict=True,
    )
    with torch.device("npu"), set_default_torch_dtype(torch.bfloat16):
        model = Qwen3_5ForConditionalGeneration(
            config,
            Mapping(rank=0, world_size=1),
            is_multimodal_active=True,
            mm_attention_backend="ascend",
        ).eval()
        loaded = model.load_weights(weights.items())
    expected_names = {
        name.replace("model.visual.", "visual.").replace("attn.qkv.", "attn.qkv_proj.")
        for name in weights
    }
    assert loaded == expected_names
    parameters = dict(model.named_parameters())
    assert set(parameters) == expected_names
    for name, tensor in weights.items():
        target = name.replace("model.visual.", "visual.").replace(
            "attn.qkv.", "attn.qkv_proj."
        )
        torch.testing.assert_close(
            parameters[target].cpu(),
            tensor.to(parameters[target].dtype),
            rtol=0,
            atol=0,
        )
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    return model, reference, processor, path


def _compare(actual, expected, *, relative_l2_limit, max_abs_limit):
    actual, expected = actual.detach().cpu().float(), expected.detach().cpu().float()
    delta = actual - expected
    relative_l2 = float(
        torch.linalg.vector_norm(delta)
        / torch.linalg.vector_norm(expected).clamp_min(1e-12)
    )
    max_abs = float(delta.abs().max())
    # Full BF16 tower drift accumulates and the final block has large outliers.
    # Individual operations below use tighter elementwise bounds.
    assert relative_l2 < relative_l2_limit, (relative_l2, max_abs)
    assert max_abs < max_abs_limit, max_abs
    cosine = float(
        torch.nn.functional.cosine_similarity(
            actual.flatten(), expected.flatten(), dim=0
        )
    )
    assert cosine > 0.98, cosine
    return {"relative_l2": relative_l2, "max_abs": max_abs, "cosine": cosine}


@pytest.mark.parametrize("image_name", ["color", "rectangle", "text"])
@torch.inference_mode()
def test_pinned_encoder_stages_and_merged_features(pinned_vision, image_name):
    model, reference, processor, path = pinned_vision
    image = fixed_images()[image_name]
    inputs = processor.image_processor(images=[image], return_tensors="pt")
    pixels, grid = inputs.pixel_values, inputs.image_grid_thw
    cpu_stages, npu_stages, bf16_stages = {}, {}, {}
    bf16_reference = copy.deepcopy(reference)
    # Keep rotary frequencies FP32, as model construction does; convert only
    # checkpoint parameters to the production dtype.
    for parameter in bf16_reference.parameters():
        parameter.data = parameter.data.bfloat16()
    handles = []

    def capture(target, name):
        def hook(module, args, output):
            target[name] = output.detach().cpu().squeeze(1)

        return hook

    for index, block in enumerate(reference.blocks):
        handles.append(
            block.register_forward_hook(capture(cpu_stages, f"block_{index}"))
        )
    for index, block in enumerate(bf16_reference.blocks):
        handles.append(
            block.register_forward_hook(capture(bf16_stages, f"block_{index}"))
        )
    for index, block in enumerate(model.visual.blocks):
        handles.append(
            block.register_forward_hook(capture(npu_stages, f"block_{index}"))
        )
    try:
        expected = reference(pixels.float(), grid)
        baseline = bf16_reference(pixels.bfloat16(), grid)
        patched = model.visual.prepare_patch_embed(pixels, grid)
        metadata = model.visual.prepare_metadata(grid)
        actual = model.visual.forward_blocks(patched, metadata)
    finally:
        for handle in handles:
            handle.remove()
    # Separate early operation drift from accumulated late-block rounding.
    limits = [
        0.012,
        0.022,
        0.028,
        0.032,
        0.04,
        0.085,
        0.08,
        0.08,
        0.08,
        0.08,
        0.095,
        0.17,
    ]
    absolute_limits = [0.08, 0.09, 0.09, 0.10, 0.25, 10, 10, 10, 10, 10, 10, 1000]
    metrics = {
        name: _compare(
            npu_stages[name],
            values,
            relative_l2_limit=limits[index],
            max_abs_limit=absolute_limits[index],
        )
        for index, (name, values) in enumerate(cpu_stages.items())
    }
    metrics["merged"] = _compare(
        actual, expected.pooler_output, relative_l2_limit=0.18, max_abs_limit=2.0
    )
    baseline_metrics = {}
    for name, expected_stage in {
        **cpu_stages,
        "merged": expected.pooler_output,
    }.items():
        bf16_stage = baseline.pooler_output if name == "merged" else bf16_stages[name]
        error = bf16_stage.float() - expected_stage.float()
        baseline_l2 = float(error.norm() / expected_stage.float().norm())
        baseline_metrics[name] = {
            "relative_l2": baseline_l2,
            "max_abs": float(error.abs().max()),
        }
        assert metrics[name]["relative_l2"] <= 2 * baseline_l2 + 0.01, (
            name,
            metrics[name],
            baseline_metrics[name],
        )
    expected_tokens = int((grid.prod(dim=-1) // 4).sum())
    assert actual.shape == (expected_tokens, 1024)
    torch.testing.assert_close(
        torch.tensor(metadata["cu_seqlens_cpu"]),
        torch.tensor([0, int(grid.prod())]),
        rtol=0,
        atol=0,
    )
    artifact = Path(
        os.environ.get("QWEN35_VISION_ARTIFACTS", path.parents[1] / "vision_validation")
    )
    artifact.mkdir(parents=True, exist_ok=True)
    image.save(artifact / f"{image_name}.png")
    torch.save(
        {
            "pixel_values": pixels,
            "image_grid_thw": grid,
            "merged_reference": expected.pooler_output,
            "merged_ascend": actual.cpu(),
            "reference_stages": cpu_stages,
            "bf16_reference_stages": bf16_stages,
            "merged_bf16_reference": baseline.pooler_output,
            "ascend_stages": npu_stages,
        },
        artifact / f"{image_name}_stages.pt",
    )
    (artifact / f"{image_name}_bf16_baseline.json").write_text(
        json.dumps(baseline_metrics, indent=2) + "\n"
    )
    (artifact / f"{image_name}_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )


@torch.inference_mode()
def test_position_interpolation_accumulates_before_bfloat16_rounding(pinned_vision):
    model, reference, _, _ = pinned_vision
    from transformers.vision_utils import get_vision_bilinear_indices_and_weights

    grid = torch.tensor([[1, 16, 16], [1, 14, 24]])
    indices, weights = get_vision_bilinear_indices_and_weights(
        grid, num_grid_per_side=48, spatial_merge_size=2
    )
    embeddings = reference.pos_embed(indices).bfloat16().float()
    expected = (embeddings * weights[:, :, None]).sum(0).bfloat16()
    actual = model.visual.fast_pos_embed_interpolate_from_list(grid.tolist()).cpu()
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.0005)
    assert (actual.float() - expected.float()).norm() / expected.float().norm() < 0.0001


@torch.inference_mode()
def test_prefilled_visual_features_release_on_their_device():
    from tokenspeed.runtime.multimodal.embedder import _drop_prefilled_encodings
    from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem

    item = MultimodalDataItem(
        modality=Modality.IMAGE, encoded=torch.ones((2, 4), device="npu")
    )
    _drop_prefilled_encodings([item])
    assert item.encoded is None and item.encoded_deepstack is None


@torch.inference_mode()
def test_patch_rotary_norm_mlp_and_one_block(pinned_vision):
    from transformers.vision_utils import get_vision_position_ids

    model, reference, processor, _ = pinned_vision
    batch = processor.image_processor(
        images=[fixed_images()["rectangle"]], return_tensors="pt"
    )
    pixels = batch.pixel_values.bfloat16()
    grid = batch.image_grid_thw
    actual_patch = model.visual.patch_embed(pixels.npu()).cpu().float()
    expected_patch = reference.patch_embed(pixels.float())
    torch.testing.assert_close(actual_patch, expected_patch, rtol=0.01, atol=0.02)
    metadata = model.visual.prepare_metadata(grid)
    angles = reference.rotary_pos_emb(get_vision_position_ids(grid, 2)).reshape(
        len(pixels), -1
    )
    torch.testing.assert_close(
        metadata["rotary_pos_emb_cos"].cpu(), angles.cos(), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        metadata["rotary_pos_emb_sin"].cpu(), angles.sin(), rtol=1e-6, atol=1e-6
    )
    x = model.visual.prepare_patch_embed(pixels, grid).cpu().float().squeeze(1)
    block = model.visual.blocks[0]
    for name in ("norm1", "norm2", "mlp"):
        actual = getattr(block, name)(x.npu().bfloat16()).cpu().float()
        expected = getattr(reference.blocks[0], name)(x.bfloat16().float())
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    expected = reference.blocks[0](
        x.bfloat16().float(),
        cu_seqlens=metadata["cu_seqlens"].cpu(),
        position_embeddings=(
            torch.cat([angles, angles], -1).cos(),
            torch.cat([angles, angles], -1).sin(),
        ),
    )
    actual = block(x.npu().bfloat16().unsqueeze(1), **metadata).cpu().float().squeeze(1)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.04)


@torch.inference_mode()
def test_processor_placeholders_positions_and_embedding_order(pinned_vision):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    from tokenspeed.runtime.multimodal.inputs import (
        Modality,
        MultimodalDataItem,
        MultimodalForwardContext,
        MultimodalInputs,
    )
    from tokenspeed.runtime.multimodal.mrope import compute_mrope_positions

    model, _, processor, path = pinned_vision
    images = [fixed_images()["color"], fixed_images()["rectangle"]]
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image} for image in images]
            + [{"type": "text", "text": "Describe both images."}],
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
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    positions = torch.where(inputs.input_ids[0] == config.image_token_id)[0]
    counts = (inputs.image_grid_thw.prod(-1) // 4).tolist()
    assert len(positions) == sum(counts)
    items = []
    pixel_start = 0
    token_start = 0
    for grid, count in zip(inputs.image_grid_thw, counts):
        patch_count = int(grid.prod())
        offset = int(positions[token_start])
        assert positions[token_start : token_start + count].tolist() == list(
            range(offset, offset + count)
        )
        items.append(
            MultimodalDataItem(
                modality=Modality.IMAGE,
                offsets=[(offset, offset + count - 1)],
                feature=inputs.pixel_values[pixel_start : pixel_start + patch_count],
                model_specific_data={"image_grid_thw": grid[None]},
            )
        )
        pixel_start += patch_count
        token_start += count
    runtime_positions, delta = compute_mrope_positions(
        config, inputs.input_ids[0].tolist(), items
    )
    with torch.device("meta"):
        reference = Qwen3_5Model(config)
    expected_positions, expected_delta = reference.get_rope_index(
        inputs.input_ids, inputs.mm_token_type_ids, image_grid_thw=inputs.image_grid_thw
    )
    torch.testing.assert_close(
        runtime_positions, expected_positions.squeeze(1), rtol=0, atol=0
    )
    torch.testing.assert_close(delta, expected_delta, rtol=0, atol=0)
    assert int(runtime_positions.max() + 1) == inputs.input_ids.shape[1] + int(
        delta.flatten()[0]
    )
    expected_features = model.get_image_feature(items)
    mm = MultimodalInputs(mm_items=items, im_token_id=config.image_token_id)
    ids = model.pad_input_ids(inputs.input_ids[0].tolist(), mm)
    embedding = torch.nn.Embedding(
        config.text_config.vocab_size, 1024, device="npu", dtype=torch.bfloat16
    )
    embedding.weight.zero_()
    actual, _ = model.vision_embedder.apply(
        torch.tensor(ids, device="npu"),
        embedding,
        MultimodalForwardContext(
            mm_inputs=[mm],
            extend_prefix_lens=[0],
            extend_seq_lens=[len(ids)],
            max_encoder_tokens=4096,
        ),
        model.get_multimodal_encoder_specs(),
        model,
    )
    torch.testing.assert_close(
        actual[positions.npu()], expected_features, rtol=0, atol=0
    )
    text_mask = inputs.input_ids[0] != config.image_token_id
    assert torch.count_nonzero(actual[text_mask.npu()]) == 0
    assert all(item.encoded is None for item in items)
    artifact = Path(
        os.environ.get("QWEN35_VISION_ARTIFACTS", path.parents[1] / "vision_validation")
    )
    torch.save(
        {"processor": dict(inputs), "positions": runtime_positions, "delta": delta},
        artifact / "multimodal_inputs.pt",
    )


@torch.inference_mode()
def test_record_encoder_stage_timings(pinned_vision):
    import importlib.metadata
    import time

    model, _, processor, path = pinned_vision
    measurements = {}
    for name, image in fixed_images().items():
        torch.npu.synchronize()
        started = time.perf_counter()
        batch = processor.image_processor(images=[image], return_tensors="pt")
        preprocess_ms = (time.perf_counter() - started) * 1000
        torch.npu.reset_peak_memory_stats()

        def timed(fn):
            torch.npu.synchronize()
            start = time.perf_counter()
            value = fn()
            torch.npu.synchronize()
            return value, (time.perf_counter() - start) * 1000

        pixels, transfer_ms = timed(
            lambda: batch.pixel_values.to(device="npu", dtype=torch.bfloat16)
        )
        patched, patch_ms = timed(
            lambda: model.visual.prepare_patch_embed(pixels, batch.image_grid_thw)
        )
        metadata, metadata_ms = timed(
            lambda: model.visual.prepare_metadata(batch.image_grid_thw)
        )
        output, encode_ms = timed(
            lambda: model.visual.forward_blocks(patched, metadata)
        )
        assert output.device.type == "npu"
        measurements[name] = {
            "preprocess_ms": preprocess_ms,
            "transfer_ms": transfer_ms,
            "patch_and_positions_ms": patch_ms,
            "metadata_ms": metadata_ms,
            "blocks_and_merger_ms": encode_ms,
            "visual_tokens": len(output),
            "grid": batch.image_grid_thw.tolist(),
            "peak_allocated_bytes": torch.npu.max_memory_allocated(),
            "peak_reserved_bytes": torch.npu.max_memory_reserved(),
        }
    artifact = Path(
        os.environ.get("QWEN35_VISION_ARTIFACTS", path.parents[1] / "vision_validation")
    )
    environment = {
        name: importlib.metadata.version(name)
        for name in ("torch", "torch_npu", "transformers", "pillow", "torchvision")
    }
    environment.update(
        device=torch.npu.get_device_name(0),
        cann="9.0.0",
        reference="Hugging Face CPU FP32 and CPU BF16 eager",
        checkpoint_revision="2fc06364715b967f1860aea9cf38778875588b17",
    )
    (artifact / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    (artifact / "encoder_timings.json").write_text(
        json.dumps(measurements, indent=2) + "\n"
    )
