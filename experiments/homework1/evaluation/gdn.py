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


"""Independent full decoder-layer and recurrent-state reference comparisons."""

import copy
from pathlib import Path

import torch

from experiments.homework1.evaluation.metrics import Limits, RunArtifacts
from experiments.homework1.evaluation.model import Reference, load_steps


@torch.inference_mode()
def compare_layer(
    run: RunArtifacts,
    native: Path,
    reference: Reference,
    fixtures,
    *,
    activation_limits: Limits,
    state_limits: Limits,
    conv_limits: Limits,
):
    """Feed real first-layer inputs through the HF decoder and its own fresh cache.

    This isolates complete layer behavior from embedding/other-layer differences.
    HF recurrent state is [H,K,V], serving state is [H,V,K]. HF convolution
    stores four samples; serving stores the last three needed by the next token.
    Every captured prefill/decode output and intermediate state is compared.
    """
    from transformers.cache_utils import DynamicCache

    layer = reference.model.model.language_model.layers[0]
    rounded_layer = copy.deepcopy(layer).to(dtype=torch.bfloat16)
    for fixture in fixtures:
        cache = DynamicCache(config=reference.config)
        rounded_cache = DynamicCache(config=reference.config)
        try:
            rows = load_steps(native, f"teacher_{fixture.name}", fixture)
        except RuntimeError as error:
            run.record_failure(
                f"layer_{fixture.name}_unavailable",
                str(error),
                details=dict(stage="layer_unavailable"),
            )
            continue
        for row in rows:
            inputs = row["layer_input"].float().unsqueeze(0)
            output = layer(
                hidden_states=inputs, position_embeddings=None, past_key_values=cache
            )
            state = cache.layers[0]
            rounded_output = rounded_layer(
                hidden_states=inputs.to(torch.bfloat16),
                position_embeddings=None,
                past_key_values=rounded_cache,
            )
            rounded_state = rounded_cache.layers[0]
            for field, actual, expected, limits in [
                (
                    "layer",
                    rounded_output.squeeze(0),
                    output.squeeze(0),
                    activation_limits,
                ),
                (
                    "ssm",
                    rounded_state.recurrent_states[0],
                    state.recurrent_states[0],
                    state_limits,
                ),
                (
                    "conv",
                    rounded_state.conv_states[0],
                    state.conv_states[0],
                    conv_limits,
                ),
            ]:
                run.record(
                    f"rounding_{field}_{fixture.name}_{row['step']}",
                    actual,
                    expected,
                    limits=limits,
                    details=dict(
                        stage="reference_layer_rounding_baseline",
                        field=field,
                        step=row["step"],
                        reference="HF CPU BF16 versus FP32",
                    ),
                )
            common = dict(
                stage="complete_decoder_layer",
                fixture=fixture.name,
                step=row["step"],
                mode=row["mode"],
                reference="HF CPU FP32 first decoder layer; identical captured inputs",
            )
            run.record(
                f"layer_{fixture.name}_{row['step']}",
                row["layer_output"],
                output.squeeze(0),
                limits=activation_limits,
                details=common,
            )
            run.record(
                f"ssm_{fixture.name}_{row['step']}",
                row["ssm"],
                state.recurrent_states[0].transpose(-1, -2).contiguous(),
                limits=state_limits,
                details=dict(**common, layout="serving V,K versus HF K,V transpose"),
            )
            run.record(
                f"conv_{fixture.name}_{row['step']}",
                row["conv"],
                state.conv_states[0, :, -row["conv"].shape[-1] :],
                limits=conv_limits,
                details=dict(**common, history="last kernel_width-1 samples"),
            )
