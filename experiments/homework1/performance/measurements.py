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


"""Client stream measurements and reproducible statistics without device imports."""

import math
from collections import defaultdict
from statistics import median


class StreamMeasurements:
    """Observe cumulative Engine stream events using caller-supplied monotonic times."""

    def __init__(self, *, start: float):
        self.start = start
        self.previous_time = start
        self.count = 0
        self.first_text_time: float | None = None
        self.first_token_time: float | None = None
        self.first_token_count: int | None = None
        self.last_token_time: float | None = None
        self.finish_reason: dict | None = None
        self.events: list[dict] = []

    def observe(self, timestamp: float, event: dict):
        """Record one event; token counts, not chunk counts, define output length."""
        if not math.isfinite(timestamp) or timestamp < self.previous_time:
            raise ValueError("stream timestamps must be finite and monotonic")
        metadata = event["meta_info"]
        count = metadata["completion_tokens"]
        if type(count) is not int or count < self.count:
            raise ValueError("completion_tokens must be a nondecreasing integer")
        if event.get("text", "") and self.first_text_time is None:
            self.first_text_time = timestamp
        if count > self.count:
            if self.first_token_time is None:
                self.first_token_time, self.first_token_count = timestamp, count
            self.last_token_time = timestamp
        self.count = count
        self.previous_time = timestamp
        reason = metadata.get("finish_reason")
        if reason is not None:
            self.finish_reason = reason
        self.events.append(
            dict(
                elapsed_ms=(timestamp - self.start) * 1000,
                completion_tokens=count,
                nonempty_text=bool(event.get("text", "")),
                finish_reason=reason,
            )
        )

    def finish(self, *, end: float, target_tokens: int, eos_policy: str) -> dict:
        """Return honest TTFT/TPOT, actual counts, termination and validity."""
        if eos_policy not in ("natural", "fixed_length"):
            raise ValueError("unknown EOS policy")
        if end < self.previous_time or not math.isfinite(end):
            raise ValueError("invalid end timestamp")
        tpot, reason = None, None
        if self.count <= 1:
            reason = "fewer_than_two_tokens"
        elif self.first_token_count != 1:
            reason = "first_event_contains_multiple_tokens"
        elif self.last_token_time is not None and self.first_token_time is not None:
            tpot = (
                (self.last_token_time - self.first_token_time) * 1000 / (self.count - 1)
            )
        finish_type = (
            self.finish_reason.get("type")
            if isinstance(self.finish_reason, dict)
            else None
        )
        valid = finish_type in ("length", "stop") and 0 < self.count <= target_tokens
        if eos_policy == "fixed_length":
            valid = valid and self.count == target_tokens
        return dict(
            client_ttft_ms=(
                None
                if self.first_text_time is None
                else (self.first_text_time - self.start) * 1000
            ),
            client_tpot_ms=tpot,
            client_tpot_reason=reason,
            e2e_ms=(end - self.start) * 1000,
            target_output_tokens=target_tokens,
            actual_output_tokens=self.count,
            finish_reason=self.finish_reason,
            valid=bool(valid),
            stream_events=self.events,
        )

    def fail(
        self, *, end: float, target_tokens: int, eos_policy: str, error: str
    ) -> dict:
        """Retain available partial stream evidence while marking an error invalid."""
        result = self.finish(
            end=end, target_tokens=target_tokens, eos_policy=eos_policy
        )
        result.update(valid=False, error=error)
        return result


METRICS = (
    "client_ttft_ms",
    "client_tpot_ms",
    "e2e_ms",
    "prefill_forward_ms",
    "decode_forward_ms_per_step",
    "peak_allocated_mib",
    "peak_reserved_mib",
)


def summarize_samples(samples: list[dict]) -> list[dict]:
    """Use only valid non-warmup samples; retain failure counts and actual lengths."""
    groups = defaultdict(list)
    for sample in samples:
        if not sample["warmup"] and sample["phase"] in ("latency", "forward"):
            groups[(sample["case"], sample["phase"])].append(sample)
    result = []
    for (case, phase), rows in groups.items():
        valid = [r for r in rows if r["valid"]]
        summary = dict(
            case=case,
            phase=phase,
            measured_repetitions=len(valid),
            failed_repetitions=len(rows) - len(valid),
            actual_output_tokens=sorted({r["actual_output_tokens"] for r in valid}),
        )
        for field in METRICS:
            values = [r[field] for r in valid if r.get(field) is not None]
            summary[field] = (
                dict(median=median(values), min=min(values), max=max(values))
                if values
                else None
            )
        result.append(summary)
    return result
