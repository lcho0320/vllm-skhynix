# KV Cache Offloading — server setup

Practical configs for extending the prefix cache beyond GPU memory. Verified by
actually starting a server with each config on this box.

> **The authoritative reference is in-tree:** [`docs/features/kv_offloading_usage.md`](../docs/features/kv_offloading_usage.md).
> It is ~200 lines, first-party, and covers every key. This document is the
> operator's condensed version: what to deploy, what to watch, and what bites you.
> When they disagree, believe the upstream doc.

> ⚠️ **vLLM marks this API experimental.** Starting a server logs:
> `Initializing KVConnectorBase_V1. This API is experimental and subject to change.`
> Pin your vLLM version if you deploy this.

---

## 1. What it is, and when it helps

Prefix caching (APC) reuses KV blocks that are **still in GPU memory**. When GPU KV
cache fills, blocks are evicted LRU and that history is gone — recomputed from
scratch next time.

KV offloading adds a **second tier**: completed blocks are copied to pinned host
memory (and optionally disk / object store / peer nodes) as they are produced. A
later request that matches those blocks gets them promoted back to GPU instead of
recomputing. Transfers use DMA (`cudaMemcpyAsync`) and run asynchronously alongside
compute, so the overhead is mostly bandwidth, not GPU cores.

**It helps when:**
- Long multi-turn conversations that outlive GPU cache residency
- Many users sharing large system prompts / RAG documents
- Prefill-heavy traffic where recompute is expensive
- Your GPU KV cache is the binding constraint on hit rate

**It does not help when:**
- Your working set already fits in GPU KV cache (you'd just mirror it — see §5)
- Traffic has no shared prefixes (each request unique)
- You are decode-bound rather than prefill-bound

### ⚠️ On GB10 (this box) it is close to pointless

Unified memory means "CPU host memory" and "GPU memory" are **the same 119.7 GiB
pool**. Offloading moves blocks from the pool to the pool — you spend a DMA copy to
free nothing. This is a **discrete-GPU optimization**.

Configure and test the mechanics here; expect the payoff only on real
discrete-GPU servers where host RAM is separate and much larger than VRAM.

---

## 2. Minimum viable config — single CPU tier

```bash
vllm serve $MODEL \
  --gpu-memory-utilization 0.85 \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "block_size": 64,
      "cpu_bytes_to_use": 100000000000
    }
  }'
```

**Verified working on this box** (at `cpu_bytes_to_use: 2e9`). Startup logs confirm it:

```
Creating v1 connector with name: OffloadingConnector
Creating offloading spec with name: CPUOffloadingSpec
```

| Key | Meaning |
|---|---|
| `cpu_bytes_to_use` | **Total** host bytes for the CPU tier — *not per worker*. Required. |
| `block_size` | Offloaded chunk size in **tokens**. Must be a multiple of the GPU block size (16 by default → 64 = 4 GPU blocks). |
| `blocks_per_chunk` | Alternative to `block_size`, in GPU blocks. **Mutually exclusive** — setting both is an error. |
| `eviction_policy` | `lru` (default) or `arc` |
| `offload_prompt_only` | `true` by default — only prefill blocks offload, decode blocks are skipped |

### Sizing `cpu_bytes_to_use`

From the upstream tuning notes, and it's the single most important knob:

> **Set it larger than the aggregate GPU KV cache.** Offloading is immediate, so a
> CPU tier smaller than GPU KV cache just mirrors what the GPU already holds and
> adds no hit rate at all.

Work it out from your own numbers (example 02 prints the GPU side):
```
GPU KV cache   = num_gpu_blocks × block_size × KV_bytes_per_token
cpu_bytes_to_use ≥ 2-4× that, leaving headroom for the rest of the host
```
A 4× ratio on an 80 GB H100 with ~40 GB KV cache means ~160 GB of host RAM. Budget
it against everything else on the node.

---

## 3. Adding a disk tier

For working sets larger than host RAM. CPU stays the primary tier; disk is
secondary. **Secondary tiers have no direct GPU access** — every GPU↔disk transfer
is staged through the CPU tier.

```bash
vllm serve $MODEL \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "spec_name": "TieringOffloadingSpec",
      "cpu_bytes_to_use": 100000000000,
      "block_size": 64,
      "eviction_policy": "lru",
      "secondary_tiers": [
        {"type": "fs", "root_dir": "/mnt/kv_cache",
         "n_read_threads": 32, "n_write_threads": 16}
      ]
    }
  }'
```

Tier types: `fs` (filesystem), `obj` (S3-compatible via NIXL), `p2p` (RDMA between
vLLM instances). Reads sit on the prefill critical path, so favour read threads when
hit rates are high.

**Disk sizing:** blocks land under `<root_dir>/<model>_<digest>/`, where `<digest>`
covers model + block size + parallelism + dtype. Changing any of those orphans the
old directory — harmless, but it will silently accumulate. Add a cleanup job.

---

## 4. Sharing cache across pods

Two instances can share a `root_dir` (e.g. a shared PVC) — **but only if you pin the
hash seed**:

```bash
PYTHONHASHSEED=0 vllm serve ...
```

Without it, each process seeds `NONE_HASH` with random bytes, so identical token
content produces **different block filenames** and you get a 0% cross-instance hit
rate that looks exactly like a broken cache. This is enforced for the `p2p` tier
(startup fails); for `fs` and `obj` it fails silently.

In Kubernetes:
```yaml
env:
  - name: PYTHONHASHSEED
    value: "0"
volumeMounts:
  - name: kv-cache
    mountPath: /mnt/kv_cache
```

---

## 5. Monitoring — the metrics that exist

Verified present on a running offload server:

| Metric | What it tells you |
|---|---|
| `vllm:external_prefix_cache_hits_total` | hits served **from the offload tier** |
| `vllm:external_prefix_cache_queries_total` | denominator for external hit rate |
| `vllm:prefix_cache_hits_total` | hits served **from GPU** (local) |
| `vllm:prefix_cache_queries_total` | denominator for local hit rate |
| `vllm:kv_offload_total_bytes_total{transfer_type="GPU_to_CPU"}` | bytes written out |
| `vllm:kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"}` | bytes promoted back |
| `vllm:kv_offload_total_time_total{transfer_type=...}` | time spent transferring |
| `vllm:kv_offload_size_bucket{...}` | transfer size histogram |

The `local` vs `external` split here is the same split visible per-request as
`num_local_cached_tokens` / `num_external_cached_tokens` (API_GUIDE §5.2).

### The three numbers that decide if it is earning its keep

1. **External hit rate** = `external_prefix_cache_hits / external_prefix_cache_queries`.
   Near zero → offloading is pure overhead. Turn it off or grow `cpu_bytes_to_use`.
2. **Promotion/eviction ratio** = `CPU_to_GPU bytes / GPU_to_CPU bytes`. Writing far
   more than you read back means you are paying DMA for blocks nobody wants.
3. **TTFT**, before and after. That is the metric offloading is meant to improve. If
   p99 TTFT did not fall, it is not working for your traffic — regardless of hit rate.

---

## 6. Deployment checklist

- [ ] Confirm GPU KV cache is actually the constraint (example 02, plus
      `vllm:prefix_cache_hits_total` low while `gpu_cache_usage_perc` is high)
- [ ] `cpu_bytes_to_use` ≥ 2× aggregate GPU KV cache
- [ ] Host RAM headroom left for the OS and everything else on the node
- [ ] `block_size` a multiple of GPU block size
- [ ] `PYTHONHASHSEED=0` if sharing storage across instances
- [ ] Baseline TTFT/throughput recorded **before** enabling, for comparison
- [ ] Alerts on external hit rate and offload transfer time
- [ ] vLLM version pinned (experimental API)
- [ ] Disk cleanup job if using an `fs` tier

## 7. Common failure modes

| Symptom | Cause |
|---|---|
| 0% external hit rate | `cpu_bytes_to_use` too small (just mirroring GPU), or no shared prefixes in traffic |
| 0% hit rate **across pods** | `PYTHONHASHSEED` not pinned — different filenames for identical content |
| TTFT got *worse* | transfer cost exceeds recompute cost; your prefill was cheap. Disable. |
| `block_size not divisible` error | `block_size` not a multiple of GPU block size; hybrid models need `--enable-prefix-caching` |
| Host OOM | `cpu_bytes_to_use` is **total**, not per-worker — easy to over-commit on multi-GPU nodes |
| Startup error on both keys | `block_size` and `blocks_per_chunk` are mutually exclusive |

## 8. Related but different

Do not confuse these (API_GUIDE §5):

- **Prefix caching** — reuse in GPU, no movement. On by default. Try this first.
- **KV offload** — this document. CPU/disk tiers on one node.
- **KV connectors** (NIXL, LMCache, Mooncake) — cache across *nodes*.
- **Weight offload** (`cpu_offload_gb`) — moves *model weights*, not KV. Different
  problem: fitting a model that does not fit.
