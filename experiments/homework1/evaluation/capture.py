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


"""Process-local evaluation hooks: capture real cache state and force sampled IDs.

Installed only by the evaluation worker's private sitecustomize directory. These
hooks never replace attention, projections, recurrence, or cache allocation.
"""

import inspect
import json
from pathlib import Path

import torch


class Capture:
    def __init__(self, directory: Path):
        self.directory = directory
        self.job = None
        self.slots = {}
        self.steps = {}
        self.pending = None
        self.layer = None
        self.batch_number = 0

    def begin(self, ctx, input_ids, positions):
        active = json.loads((self.directory / "active.json").read_text())
        if not active["capture_enabled"]:
            self.pending = None
            self.layer = None
            return
        if self.job is None or active["name"] != self.job["name"]:
            self.job = active
            self.slots = {}
            self.steps = {}
        self.batch_number += 1
        backend = ctx.attn_backend.linear_attn_backend
        bounds = backend.forward_metadata.query_start_loc.cpu().tolist()
        self.pending = dict(
            ids=input_ids.detach().cpu().clone(),
            positions=positions.detach().cpu().clone(),
            bounds=bounds,
            batch_number=self.batch_number,
            batch_size=ctx.bs,
            mode=str(ctx.forward_mode),
            num_extends=ctx.num_extends,
            cached_decode=ctx.forward_mode.is_decode(),
            backend=backend,
        )
        self.layer = None

    def sample(self, logits, sampling_info, tokens):
        if self.pending is None:
            return
        slots = sampling_info.req_pool_indices.cpu().tolist()
        p = self.pending
        for row, slot in enumerate(slots):
            left, right = p["bounds"][row : row + 2]
            ids = p["ids"][left:right].tolist()
            if slot not in self.slots or row < p["num_extends"]:
                matches = [f for f in self.job["fixtures"] if f["input_ids"] == ids]
                if len(matches) != 1:
                    raise RuntimeError(
                        f"Cannot identify request {slot} from exact input IDs: {ids}"
                    )
                self.slots[slot] = matches[0]
                self.steps[slot] = 0
            fixture = self.slots[slot]
            step = self.steps[slot]
            if step >= self.job["steps"]:
                continue
            saved = dict(
                input_ids=ids,
                positions=p["positions"][..., left:right].clone(),
                logits=p["model_logits"][row].clone(),
                sampled_token=int(tokens[row]),
                fed_next_token=(
                    fixture["continuation_ids"][step]
                    if self.job["teacher"]
                    else int(tokens[row])
                ),
                slot=slot,
                batch_number=p["batch_number"],
                batch_size=p["batch_size"],
                mode=p["mode"],
                cached_decode=p["cached_decode"],
                step=step,
                fixture=fixture["name"],
            )
            if self.layer is not None:
                saved.update(
                    layer_input=self.layer["input"][left:right],
                    layer_output=self.layer["output"][left:right],
                    conv=self.layer["conv"][row],
                    ssm=self.layer["ssm"][row],
                    block_in=self.layer["block_in"][row],
                    block_out=self.layer["block_out"][row],
                )
            target = self.directory / self.job["name"]
            target.mkdir(exist_ok=True)
            filename = target / f"{fixture['name']}_{step:03d}.pt"
            if filename.exists():
                raise FileExistsError(filename)
            torch.save(saved, filename)
            if self.job["teacher"]:
                tokens[row] = fixture["continuation_ids"][step]
            self.steps[slot] += 1
        self.pending = None


def install(directory: Path):
    """Install hooks in this disposable worker process; returns its controller."""
    from tokenspeed.runtime.models.qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
        Qwen3_5LinearDecoderLayer,
    )
    from tokenspeed.runtime.sampling.backends.greedy import GreedySamplingBackend

    capture = Capture(directory)
    model_forward = Qwen3_5ForConditionalGeneration.forward
    layer_forward = Qwen3_5LinearDecoderLayer.forward
    sample = GreedySamplingBackend.sample
    verify = GreedySamplingBackend.verify
    model_signature = inspect.signature(model_forward)
    layer_signature = inspect.signature(layer_forward)

    def forward(self, *args, **kwargs):
        bound = model_signature.bind(self, *args, **kwargs).arguments
        capture.begin(bound["ctx"], bound["input_ids"], bound["positions"])
        result = model_forward(self, *args, **kwargs)
        if capture.pending is not None:
            capture.pending["model_logits"] = (
                result.next_token_logits.detach().cpu().clone()
            )
        return result

    def layer(self, *args, **kwargs):
        if self.layer_id != 0 or capture.pending is None:
            return layer_forward(self, *args, **kwargs)
        bound = layer_signature.bind(self, *args, **kwargs).arguments
        hidden = bound["hidden_states"]
        residual = bound["residual"]
        inputs = (
            (hidden if residual is None else hidden + residual).detach().cpu().clone()
        )
        result = layer_forward(self, *args, **kwargs)
        output, residual = result
        output = output if residual is None else output + residual
        backend = capture.pending["backend"]
        incoming, outgoing, conv, ssm = backend._layer_state(0)
        count = bound["ctx"].bs
        indices = outgoing[:count].long()
        capture.layer = dict(
            input=inputs,
            output=output.detach().cpu().clone(),
            conv=conv.index_select(0, indices).detach().cpu().clone(),
            ssm=ssm.index_select(0, indices).detach().cpu().clone(),
            block_in=incoming[:count].cpu().tolist(),
            block_out=outgoing[:count].cpu().tolist(),
        )
        return result

    def sampling(self, logits_output, sampling_info):
        tokens, auxiliary = sample(self, logits_output, sampling_info)
        capture.sample(logits_output.next_token_logits, sampling_info, tokens)
        return tokens, auxiliary

    def verification(self, logits_output, sampling_info, candidates, *, tree):
        if capture.pending is not None and (
            candidates.shape[1] != 1 or tree is not None
        ):
            raise RuntimeError(
                "Evaluation requires non-speculative single-token decode"
            )
        tokens, lengths = verify(
            self, logits_output, sampling_info, candidates, tree=tree
        )
        capture.sample(
            logits_output.next_token_logits, sampling_info, tokens.reshape(-1)
        )
        return tokens, lengths

    Qwen3_5ForConditionalGeneration.forward = forward
    Qwen3_5LinearDecoderLayer.forward = layer
    GreedySamplingBackend.sample = sampling
    GreedySamplingBackend.verify = verification
    return capture
