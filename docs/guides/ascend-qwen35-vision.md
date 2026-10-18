# Qwen3.5-0.8B images on Ascend

The existing Qwen vision tower uses the `ascend` multimodal attention backend
on Ascend by default. It runs native bidirectional attention through the shared
kernel package. Text attention continues to use the MHA/hybrid GDN backend.
This bring-up targets one chip, BF16, eager execution and images. Video,
encoder parallelism, quantization and encoder graph capture are not validated.

## Environment and checkpoint

The validation checkpoint is `Qwen/Qwen3.5-0.8B`, revision
`2fc06364715b967f1860aea9cf38778875588b17`, including its processor and chat
template. The tested stack is Python 3.11, PyTorch 2.9.0, torch_npu 2.9.0,
CANN 9.0.0 and Transformers 5.12.0. In the environment used for serving,
install Pillow and the torchvision version matching PyTorch; the validation
used Pillow 12.1.0 and torchvision 0.24.0. Do not replace the installed
Ascend PyTorch build while installing image dependencies.

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PYTHONPATH="${PWD}/python:${PWD}/tokenspeed-kernel/python:${PWD}/tokenspeed-kernel-npu/python:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0  # choose a free chip
export OMP_NUM_THREADS=8
export TOKENSPEED_MM_ENABLE_ENCODER_CUDA_GRAPH=0
# Local gRPC probes must bypass any outbound proxy.
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="${no_proxy}"
export GRPC_ENABLE_HTTP_PROXY=0
```

## Numerical validation

```bash
export QWEN35_VISION_MODEL="${PWD}/outputs/models/Qwen3.5-0.8B"
export QWEN35_VISION_ARTIFACTS="${PWD}/outputs/vision_validation"
python -m pytest -q tokenspeed-kernel-npu/test/test_vision_attention.py
python -m pytest -q test/runtime/models/test_qwen35_vision_npu.py
```

Checkpoint tests skip when the local checkpoint is absent. They compare with
installed Hugging Face CPU FP32 execution, verify all 153 visual parameters
exactly after loading, and retain synthetic images, exact processor tensors,
per-block features, merged features and error metrics under the artifact path.
The fixtures include a solid color, non-square two-color rectangle and text.
Metadata and embedding insertion order are compared exactly. Operation tests
check patch projection, LayerNorm, tanh GELU MLP and one block with elementwise
tolerances; rotary metadata uses FP32 tolerances. Learned positions accumulate
in FP32 before conversion to model dtype, matching the reference.

Full tower checks record maximum absolute error, relative L2 error and cosine
similarity. BF16 rounding grows through this checkpoint's later blocks, whose
activations contain large outliers. Stage-specific relative L2 limits begin at 0.012 for block 0 and grow to
0.17 for block 11; the merger limit is 0.18. Every stage also has an explicit
maximum absolute error budget and cosine similarity above 0.98. An independent
CPU BF16 reference keeps rotary buffers in FP32; each NPU stage must stay
within twice that baseline drift plus 0.01. These limits apply to accumulated
model drift, not individual attention or layout operations.
Inspect the saved metrics when changing arithmetic. Passing a caption request
alone does not establish numerical agreement.

Visual encodings remain owned by their requests. Release records the stream
of the tensor's own device. Optional multimodal timing synchronizes NPU work
before reporting elapsed encode time; enable `TOKENSPEED_LOG_MM_TIMING=1` only
when measuring, since synchronization changes serving overlap.

## Gateway validation and serving

The gateway fixture starts and shuts down its own local server on a free port:

```bash
QWEN35_VISION_E2E=1 python -m pytest -q test/runtime/models/test_qwen35_vision_gateway_npu.py
```

It exercises three image fixtures, two images in one request, repeated images,
concurrent text/image requests and two concurrent image requests. It verifies
processor-derived prompt/visual token counts and records responses and timings.
Use image-first message content when comparing with this gateway: its vision
preprocessing groups images before text. Matching token counts alone does not
establish matching prompts. The test verifies exact token IDs after restoring
internal image pad markers to the checkpoint's image token.

The serving API currently exposes sampled-token logprobs, but rejects output
top-k logprobs. A test-only subprocess hook retains the real first-token logits
before sampling; it forwards all arguments and leaves computation unchanged.
The test compares the entire centered vocabulary vector against CPU FP32
(relative L2 < 0.05, RMSE < 0.15, maximum absolute error < 0.6), requires at
least eight shared top-ten tokens, and checks the sampled-token logprob within
0.25. Captured logits and reference inputs are saved for inspection.

`encoder_timings.json` separates CPU preprocessing, host-to-device transfer,
patch/position computation, metadata and the twelve blocks plus merger.
`gateway.log` retains synchronized encoder timing and the existing execution
prefill/decode timing. `gateway_memory.json` records the worker's peak allocated
and reserved memory, including model weights and its configured cache. The
first captured request includes diagnostic device-to-host copying and file I/O;
use repeated requests when assessing serving latency. BF16 batch-dependent
rounding can change caption wording; concurrent tests check image-specific
content instead of requiring bitwise identical scores.

A matching manual launch is:

```bash
python -m tokenspeed.cli serve \
  --device npu --model "${QWEN35_VISION_MODEL}" \
  --served-model-name qwen35-vision-npu --host 127.0.0.1 --port 31891 \
  --world-size 1 --dtype bfloat16 --max-model-len 4096 --max-num-seqs 2 \
  --max-prefill-tokens 4096 --chunked-prefill-size 4096 \
  --gpu-memory-utilization 0.2 --prefix-granularity 128 \
  --attention-backend mha --mm-attention-backend ascend \
  --sampling-backend greedy --policy random \
  --disable-pdl --disable-autotune --disable-kvstore \
  --disable-prefix-caching --disable-prefill-graph --enforce-eager
```

Use the ordinary OpenAI image chat-completion request with a PNG data URL.
Set `chat_template_kwargs` to `{"enable_thinking": false}` for the short
non-thinking prompts used by validation. Add `--enable-output-logprobs` only
when requesting sampled-token scores. Video and encoder ACL graph capture
remain separate, unverified milestones.
