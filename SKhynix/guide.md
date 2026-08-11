# vLLM for Data-Center Management — What to Dive Into

Study roadmap for `spark-fc72` (GB10). Companion to [README.md](README.md) and [CODE_ANALYSIS.md](CODE_ANALYSIS.md).

> Markdown version of [guide.txt](guide.txt). Same content, same structure — keep both in sync if you edit one.

---

## The short answer

Three things decide whether an LLM serving fleet behaves, and none of them is "the model":

| # | Area | Why it dominates |
|---|---|---|
| 1 | **Memory** | KV cache size determines concurrency. Everything else follows. |
| 2 | **Scheduling** | Continuous batching, admission, preemption. The throughput/latency tradeoff is made here, *per step*. |
| 3 | **Failure** | What happens when memory runs out, an engine dies, or a rollout cold-starts. This is most of the actual job. |

Learn those three deeply. Quantization, speculative decoding, structured outputs, and multimodal are features you can pick up in an afternoon each once the core is solid. **Do not start with them.**

---

## Phase 1 — Memory and capacity (week 1)

The single highest-value thing you can understand. You already met it the hard way:

```
Free memory on device cuda:0 (56.65/119.7 GiB) on startup is less than
desired GPU memory utilization (0.92, 110.12 GiB)
```

**Run**
- [00_env_check.py](examples/00_env_check.py)
- [02_memory_and_kv_cache.py](examples/02_memory_and_kv_cache.py)

**Read**
- [gpu_worker.py](../vllm/v1/worker/gpu_worker.py) — `init_device()`
- [utils.py](../vllm/v1/worker/utils.py) — `request_memory()` ← *your error*
- [kv_cache_manager.py](../vllm/v1/core/kv_cache_manager.py)
- [block_pool.py](../vllm/v1/core/block_pool.py)
- [docs/design/paged_attention.md](../docs/design/paged_attention.md)

**Master these**
- The budget equation: `utilization × pool − weights − activations = KV cache`
- `KV bytes/token = 2 × layers × kv_heads × head_dim × dtype_bytes`
- `concurrency = KV tokens / avg context length`
- Why GB10 unified memory makes this different — **host RAM *is* GPU memory**

> **You know it when:** given a model and a memory budget, you can predict `num_gpu_blocks` and max concurrency on paper *before* starting the engine.

---

## Phase 2 — Scheduling and throughput (week 2)

**Run**
- [03_batching_throughput.py](examples/03_batching_throughput.py)
- [08_online_serving.sh](examples/08_online_serving.sh) — the request-rate sweep in section 4d

**Read**
- [scheduler.py](../vllm/v1/core/sched/scheduler.py) ← **read `schedule()` line by line**
- [request_queue.py](../vllm/v1/core/sched/request_queue.py)
- [config/scheduler.py](../vllm/config/scheduler.py)

**Master these**
- Continuous batching: why a batch is re-formed every step
- `max_num_seqs` vs `max_num_batched_tokens`, and which one you hit first
- Prefill vs decode: compute-bound vs bandwidth-bound, why they conflict
- **Preemption** — what triggers it, what it costs, why it looks like a cliff
- TTFT vs ITL vs throughput — you cannot optimize all three

> **You know it when:** you can look at a p99 TTFT regression and say whether it is queueing, batch size, or cache pressure — without guessing.

---

## Phase 3 — Failure and operations (week 3)

**Run**
- [10_chaos_and_limits.sh](examples/10_chaos_and_limits.sh) — all 7 scenarios
- [11_metrics.py](examples/11_metrics.py) `--watch`

**Read**
- [core.py](../vllm/v1/engine/core.py) — the busy loop + process boundary
- [v1/metrics/](../vllm/v1/metrics/)
- [docs/deployment/k8s.md](../docs/deployment/k8s.md)

**Master these**
- The process split: API server vs `EngineCore`, and what a dead engine looks like from outside (*alive HTTP, useless service*)
- Which metrics are leading indicators: `num_requests_waiting`, `num_preemptions_total`, `kv_cache_usage_perc`
- Cold-start time as an autoscaling parameter
- **The 37-day problem** — health checks passing while doing zero work. `generation_tokens_total` is the only honest liveness signal.

> **You know it when:** you can write the alerting rules for a vLLM fleet and defend each threshold.

---

## Phase 4 — Features, as needed (week 4+)

Pick by what your workload actually needs. Each is roughly a day.

| Feature | Example | Note |
|---|---|---|
| Prefix caching | [04](examples/04_prefix_caching.py) | Highest ROI on real traffic (shared system prompts, RAG). Free throughput. |
| Quantization | [07](examples/07_quantization.py) | Decides what fits. Needed for anything above ~30B on this box. |
| Structured outputs | [05](examples/05_structured_outputs.py) | Removes parse failures from the error budget. Guarantees **structure, not accuracy**. |
| Speculative decoding | [06](examples/06_speculative_decoding.py) | Latency win, throughput cost. Measured **1.64x** here on grounded/repetitive prompts; measure acceptance on *your* traffic. |
| LoRA | — | Multi-tenant adapter serving on one base model |
| Disaggregation | — | Read-only here (needs >1 GPU); see [CODE_ANALYSIS.md](CODE_ANALYSIS.md) §4 |

---

## Phase 5 — eBPF (parallel track, start after Phase 2)

See [09_ebpf_observability.sh](examples/09_ebpf_observability.sh) for the full answer, verified working on this kernel (`bpftrace v0.20.2`, BTF present).

**The honest summary:** there is **no** direct vLLM/eBPF integration and there will not be one. vLLM has no eBPF hooks. The tie-in is real but it is at the layer *below and around* vLLM.

| | |
|---|---|
| **eBPF sees** | syscalls (ioctl to the NVIDIA driver, mmap during model load), page faults (critical on GB10 — unified memory means host pressure *is* GPU pressure), block I/O during weight load, TCP accept/retransmit, CPU run-queue delay causing ITL jitter |
| **eBPF cannot see** | anything inside the GPU. No kernel execution, no SM occupancy, no HBM bandwidth. It stops at the driver boundary. |

**Division of labor**

| Tool | Domain |
|---|---|
| DCGM / `nvidia-smi` | GPU-side truth |
| vLLM `/metrics` | Engine-side truth (queue depth, KV usage, TTFT) |
| eBPF | Everything between the NIC and the driver — where *"GPU looks idle but latency is bad"* lives |

**Learning path**
1. `bpftrace` one-liners — the 6 recipes in [example 09](examples/09_ebpf_observability.sh) (all verified here)
2. Gregg, *BPF Performance Tools* — ch. 6 (CPU), 8 (filesystems), 10 (networking)
3. Cilium + Hubble on this single-node cluster. You own the whole control plane, which makes it an unusually good place to learn it. Would replace the Calico CNI currently installed.
4. libbpf / CO-RE only if you end up writing custom collectors

> Sequence it **after** Phase 2. eBPF answers *"why is this slow"*, which is only useful once you know what normal looks like.

---

## Testing scenarios to cover

You said you expect to test a lot. This is the matrix worth building out.

### Capacity
- Throughput at saturation (`request-rate=inf`)
- Latency percentiles at fixed rates; sweep to find the knee
- Max concurrency before preemption
- Cold-start time (bounds autoscaling response)

### Memory
- OOM at startup — clean failure? fast? loud?
- Memory pressure *during* serving — survives? degrades how?
- KV exhaustion → preemption behavior
- Fragmentation over long runs

### Failure
- Engine core death → detection time
- Pod eviction / node pressure
- Client disconnect mid-stream → are blocks freed?
- Malformed and oversized requests → 4xx, not OOM
- Rolling restart with traffic in flight

### Correctness
- Output stability at `temperature=0` across batch sizes — batching should not change greedy results; see [batch_invariant.py](../vllm/model_executor/layers/batch_invariant.py) if this matters to you
- Quality delta after quantization — run real evals, [tests/evals/](../tests/evals/)

### Efficiency
- Prefix cache hit rate on production-shaped traffic
- Tokens per GPU-second per dollar
- Idle detection (the 37-day problem)

> Automate these with `vllm bench sweep` and keep results in version control. **A performance number without the config that produced it is noise.**

---

## Things specific to this box — do not forget

- **119.7 GiB unified memory, shared with the OS.** `gpu_memory_utilization` is a fraction of a pool the operating system also lives in. Page cache alone was holding ~60 GiB when we checked. **Start at 0.30, not the 0.9 default.**
- `sm_121` Blackwell: FP8 and NVFP4 available. arm64, so some wheels are missing.
- Single GPU: TP/PP are code-reading only.
- **This node IS its own k8s control plane:**
  ```bash
  sudo KUBECONFIG=/etc/kubernetes/admin.conf kubectl ...
  ```
  The k3s agent pointing at `192.168.6.55` is a red herring.
- `crictl stop` does not keep a pod down. **Scale the Deployment.**
- The cached `Llama-3.1-70B` (263 GB BF16) cannot run here unquantized. Use it as a worked example of why quantization is a *capacity decision*, not a tuning one.
- Per [AGENTS.md](../AGENTS.md): always use `uv` and `.venv/bin/python`. Never system `python3`/`pip`.

---

## Reading order if you only have a day

1. [docs/design/arch_overview.md](../docs/design/arch_overview.md)
2. [scheduler.py](../vllm/v1/core/sched/scheduler.py) — `schedule()`
3. [kv_cache_manager.py](../vllm/v1/core/kv_cache_manager.py)
4. [utils.py](../vllm/v1/worker/utils.py) — `request_memory()`
5. [CODE_ANALYSIS.md](CODE_ANALYSIS.md) §4 — how Dynamo and friends attach
