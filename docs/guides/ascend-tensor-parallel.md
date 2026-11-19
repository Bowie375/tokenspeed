# Qwen3.5 tensor parallelism on Ascend

The eager dense-Qwen3.5 path supports one or two Ascend chips through the
existing TP execution path. Runtime code uses `tokenspeed-kernel`; the NPU
kernels stay behind that boundary. HCCL collectives use the standard PyTorch
`all_gather_into_tensor` and `reduce_scatter_tensor` APIs, including ragged
rank-major token gathers and scatters. CUDA stream-capture queries in the
vocabulary gather are gated by the logits tensor device.

Q/K/V and MLP gate/up projections are column sharded; attention/GDN output
and MLP down projections produce partial sums reduced across ranks. KV heads,
GDN convolution channels, recurrent heads, and their state buffers follow the
same TP partition. GDN state uses FP32 `[head,value,key]`; convolution history
keeps raw projected samples. Head-local state remains in scheduler-owned
cache groups. TP changes neither cache ownership nor the unified decode
position/frontier contract.

Use `experiments/homework2` for the complete validation and measurement entry
points. It checks the actual loaded parameter shapes, full-vocabulary logits,
canonical positions, exact free-running rank token/input agreement, KV head
partitioning, concatenated GDN state and Q/K/V convolution history, and real
batch/fresh-request state isolation and physical block reuse. Numerical
comparisons between different TP sizes are tolerance based; BF16 ties are
reported separately from exact generation. Missing evidence is a failure.

Activate the project Python environment, source CANN, and choose two available
physical chips without stopping other services. The example below assumes
`ASCEND_RT_VISIBLE_DEVICES` already names the chosen pair:

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PYTHONPATH="${PWD}:${PWD}/python:${PWD}/tokenspeed-kernel/python:${PWD}/tokenspeed-kernel-npu/python:${PYTHONPATH:-}"
python -m experiments.homework2.run \
  --model outputs/models/Qwen3.5-0.8B \
  --devices "$ASCEND_RT_VISIBLE_DEVICES" \
  --config experiments/homework2/configs/baseline.json \
  --task correctness --reference hf
```

The same entry point runs 4B, 9B and 27B with `--reference none` to compare
TP1 with TP2. Independent HF CPU reference precision is explicit in the JSON
config. Correctness hooks are confined to private disposable workers; they
never replace model computation, communication or cache allocation. Teacher
forcing supplies identical continuation IDs for state/logit comparisons;
free-generation jobs retain native sampling.

This establishes the local-shard and accepted-position semantics required by
Host offloading. It does not implement Host budgets, transfer, eviction or
restore; those belong to the cache subsystem and must carry all TP ranks to
the same accepted context position before continuation.

## Dense-layer communication and message sizes

For the supplied dense TP2 launch, attention TP and dense TP both equal two.
`CommManager.use_all_reduce(False)` therefore selects replicated token rows.
`Qwen3_5AttentionDecoderLayer.self_attention` applies local Q/K/V attention and
`o_proj`; the output projection sets `reduce_results=False`. The subsequent
`post_attn_reduce_norm` calls `post_attn_comm` to sum the local `[T,H]` results.
The GDN layer uses the same reduction boundary after its local recurrence and
output projection. Gate/up activation columns and down-projection input columns
are local to each rank. `post_dense_comm` sums the down-projection partial output
(the reduction can be deferred to the next input/final norm by the communication
policy). These are two hidden-state AllReduces per decoder layer, not a gather
of each projection's output. KV, convolution and SSM state do not communicate.

For BF16, each hidden-state collective has a per-rank input payload of `2*T*H`
bytes. With 0.8B (`H=1024`, 24 layers), this is 2,048 bytes for a single decode
row, 262,144 bytes at 128 prefill rows, and 1,048,576 bytes at 512 rows. With
27B (`H=5120`, 64 layers), the corresponding sizes are 10,240, 1,310,720 and
5,242,880 bytes. These match representative small/large shapes in the HCCL
microbenchmark. A batch changes `T`; generated-token count is not its dimension.
The layer-only estimate is `2 * layer_count * collective_latency` per forward;
it excludes kernels, host work, embedding, sampling and possible overlap.
It is an order-of-magnitude estimate rather than a measured model timeline.

Vocab-parallel embedding additionally sums its masked local lookup contribution.
The LM head computes a local vocabulary slice and `LogitsProcessor._get_logits`
gathers slices before the common greedy sampler; with the 248,320-token vocabulary,
two ranks have 124,160 local columns. RMSNorm/RoPE and canonical positions are
replicated. The baseline is eager, with overlap scheduling disabled. PyTorch
collectives are stream ordered: later consumers wait for their outputs, although
the Python call returning is not an NPU synchronization timestamp.

Correctness workers also save `communication_<fixture>_<step>.json` under each
rank/job. The observations wrap the real HCCL backend and preserve all supplied
arguments and return values; each record contains the operation, runtime caller,
input shape, dtype, ranks and payload bytes. They add no tensor synchronization.
This observation is omitted from performance workers. Actual HCCL call durations
are measured separately by `collectives.py`; payload bytes are not link traffic.

For 0.8B, each TP2 rank has four full-attention Q heads and one KV head, eight
GDN key/value heads, 1,792 MLP intermediate columns, and the full 1,024-wide
hidden/residual stream. For 27B the corresponding values are 12 Q heads, two KV
heads, eight GDN key heads, 24 GDN value heads, 8,704 MLP columns and hidden width
5,120. The saved `rank*/model.json` describes all loaded projection shapes; state
captures verify local heads and reconstruct global tensors rather than assuming
these dimensions solely from the checkpoint configuration.
