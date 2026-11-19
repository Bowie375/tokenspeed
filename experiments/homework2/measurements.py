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


"""Token-based client latency; text arrival is retained as a separate observation."""

import math

from experiments.homework1.performance.measurements import StreamMeasurements


class TokenMeasurements(StreamMeasurements):
    """Require usable TTFT/TPOT rather than accepting missing stream timings."""

    def finish(self, *, end: float, target_tokens: int, eos_policy: str) -> dict:
        result = super().finish(
            end=end, target_tokens=target_tokens, eos_policy=eos_policy
        )
        result["client_first_text_ms"] = result["client_ttft_ms"]
        result["client_ttft_ms"] = (
            None
            if self.first_token_time is None
            else (self.first_token_time - self.start) * 1000
        )
        fields = ["client_ttft_ms", "e2e_ms"]
        if target_tokens > 1:
            fields.append("client_tpot_ms")
        result["valid"] = result["valid"] and all(
            result[k] is not None and math.isfinite(result[k]) and result[k] >= 0
            for k in fields
        )
        return result
