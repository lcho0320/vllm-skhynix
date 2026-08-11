# vLLM Code Analysis — where everything lives

Against the local checkout at commit `6c7e679`, version `0.26.1rc1.dev53`.
Paths are relative to `/home/leeksun/testingEnvironment/vLLM_testing/vllm/`.

---

## 1. The 60-second mental model

A request travels through five layers. Almost every question about vLLM is a
question about one of them:

```
  HTTP / OpenAI API          vllm/entrypoints/
        │                    FastAPI routes, chat templating, tool parsing
        ▼
  AsyncLLM / LLMEngine       vllm/v1/engine/
        │                    tokenize, validate, own the request lifecycle
        ▼  (ZMQ IPC, separate process)
  EngineCore                 vllm/v1/engine/core.py
        │                    the busy loop: schedule → execute → emit outputs
        ├── Scheduler        vllm/v1/core/sched/
        │                    who runs this step; admission, preemption
        ├── KVCacheManager   vllm/v1/core/
        │                    block allocation, prefix-cache hashing
        ▼
  Executor → Worker          vllm/v1/executor/, vllm/v1/worker/
        │                    one process per GPU; owns the device
        ▼
  ModelRunner → Model        vllm/v1/worker/gpu_model_runner.py,
                             vllm/model_executor/
                             build the batch, run forward, sample
```

**The process split matters operationally.** `EngineCore` runs in its own process
(you saw `EngineCore pid=3062140` in your stack trace). The API server talks to it
over ZMQ. This is why an engine crash shows up as an API server that is alive but
useless, and why the health probe must reflect engine state, not just HTTP liveness.

---

## 2. Directory map

### `vllm/v1/` — the current engine (this is where the action is)

V1 is a rewrite; V0 is gone from this tree. Anything you read online referencing
`vllm/core/` or `vllm/worker/` at the top level is outdated.

| Path | What it does | Why you'd open it |
|---|---|---|
| `v1/engine/core.py` | `EngineCore` / `EngineCoreProc` — the main busy loop | Startup failures land here; your day-one traceback passed through it |
| `v1/engine/async_llm.py` | async engine behind the API server | Streaming, cancellation, per-request lifecycle |
| `v1/engine/llm_engine.py` | sync engine behind `LLM()` | Offline batch path |
| `v1/engine/core_client.py` | client side of the ZMQ IPC | Debugging API-server↔engine communication |
| `v1/engine/detokenizer.py` `output_processor.py` | token IDs → text, stop strings | Wrong-looking output, stop-sequence bugs |
| `v1/core/sched/scheduler.py` | **the scheduler** — admission, batching, preemption | The single most important file for throughput/latency behavior |
| `v1/core/sched/request_queue.py` | queue policies (FCFS, priority) | Implementing QoS / priority tiers |
| `v1/core/kv_cache_manager.py` | KV block allocation per request | Preemption and cache-exhaustion behavior |
| `v1/core/block_pool.py` | the physical block pool + prefix-cache hash table | How APC actually works |
| `v1/core/kv_cache_utils.py` | block hashing, cache sizing math | Understanding `num_gpu_blocks` |
| `v1/core/kv_cache_coordinator.py` `single_type_kv_cache_manager.py` | hybrid models (attention + Mamba layers) | Hybrid/SSM model memory |
| `v1/worker/gpu_worker.py` | owns the device; memory profiling at startup | **`request_memory()` — the exact function that raised your ValueError** (`v1/worker/utils.py:408`) |
| `v1/worker/gpu_model_runner.py` | builds the batch tensors, runs forward, samples | Deepest "what is actually executed" file |
| `v1/worker/block_table.py` `gpu_input_batch.py` | per-step batch state | Batch construction bugs |
| `v1/executor/` | `uniproc_executor.py` (single GPU), multiproc, Ray | How workers are launched |
| `v1/attention/backends/` | FlashAttention, FlashInfer, Triton, MLA, Mamba | Backend selection and per-backend quirks |
| `v1/sample/` | sampling ops, logits processors | Temperature/top-p/penalties, custom logit processors |
| `v1/spec_decode/` | n-gram, EAGLE, Medusa proposers | Example 06 |
| `v1/structured_output/` | xgrammar / guidance / outlines / lm-format-enforcer backends | Example 05 |
| `v1/metrics/` | Prometheus metrics, stat loggers | Example 11 |
| `v1/kv_offload/`, `simple_kv_offload/` | CPU/disk KV offloading | Extending cache beyond GPU memory |
| `v1/fault_tolerance/` | failure handling | Chaos testing (example 10) |
| `v1/pool/` | pooling/embedding models | Non-generative serving |

### `vllm/entrypoints/` — the serving surface

| Path | What |
|---|---|
| `openai/` | OpenAI-compatible FastAPI server — `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/health`, `/metrics` |
| `anthropic/` | Anthropic-compatible API surface |
| `llm.py` | the `LLM` class — offline entry point used by every example here |
| `cli/` | `vllm serve`, `chat`, `complete`, `bench`, `run-batch` |
| `chat_utils.py` | chat template application, multimodal message parsing |
| `grpc_server.py` | gRPC surface |
| `mcp/` | Model Context Protocol support |
| `launcher.py` | server bootstrap |

`--api-key`, request validation, and the routes flooding your abandoned pods'
logs (`/health`, `/v1/models`) all live under `openai/`.

### `vllm/model_executor/` — the models themselves

| Path | What |
|---|---|
| `models/` | ~200 architecture implementations (llama.py, qwen.py, …). Adding a model = adding a file here + registry entry |
| `layers/` | building blocks: `linear.py`, `layernorm.py`, `activation.py`, `rotary_embedding/`, `vocab_parallel_embedding.py` |
| `layers/attention/` | attention layer wrappers over the v1 backends |
| `layers/fused_moe/` | MoE routing and fused expert kernels |
| `layers/quantization/` | **FP8, AWQ, GPTQ, compressed-tensors, bitsandbytes, mxfp4, modelopt** — one file per method |
| `layers/mamba/`, `conv.py` | SSM/Mamba layers for hybrid models |
| `model_loader/` | weight loading: safetensors, sharded state, tensorizer, runai streamer |
| `layers/pooler/` | embedding/classification heads |

### `vllm/config/` — every tunable, one file per domain

This is the best index of what vLLM can even do. `cache.py` (block size,
`kv_cache_dtype`), `scheduler.py` (`max_num_seqs`, `max_num_batched_tokens`),
`parallel.py` (TP/PP/DP/EP), `speculative.py`, `quantization.py`, `lora.py`,
`kv_transfer.py`, `compilation.py`, `observability.py`, `fault_tolerance.py`.
`vllm.py` composes them into the `VllmConfig` that example 02 introspects.

### `vllm/distributed/` — multi-GPU and multi-node

| Path | What |
|---|---|
| `parallel_state.py` | process groups; TP/PP/DP/EP rank assignment |
| `device_communicators/` | NCCL, custom all-reduce, shared-memory paths |
| `kv_transfer/` | **KV connectors — the third-party extension point that matters most** (see §4) |
| `kv_events.py` | ZMQ publisher of KV cache events (block stored/evicted) for external routers |
| `eplb/` | expert-parallel load balancing for MoE |
| `elastic_ep/` | elastic expert parallelism |

Not exercisable on this single-GPU box beyond code reading.

### `vllm/compilation/` — torch.compile integration

`backends.py`, `piecewise_backend.py`, `cuda_graph.py`, `passes/`. vLLM compiles
the model with a custom inductor backend and captures CUDA graphs for decode.
This is what `enforce_eager=True` disables, and why the first startup is slow and
subsequent ones hit a cache (`caching.py`).

### `csrc/` — the CUDA/C++ kernels

`attention/` (paged attention), `quantization/` (FP8/INT4/mxfp4 GEMMs), `moe/`,
`cutlass_extensions/`, `custom_all_reduce.cuh`, `cpu/`, `rocm/`. Compiled into the
`.so` files sitting in `vllm/` (`_C_stable_libtorch.abi3.so`, `_moe_C…so`).

**This is what `VLLM_USE_PRECOMPILED=1` skips downloading and what a source build
compiles.** Your original install failure was the wheel server having no `cu130`
variant for these.

### `vllm/platforms/` — hardware abstraction

`cuda.py`, `rocm.py`, `cpu.py`, `tpu.py`, `xpu.py`, `interface.py`. Capability
detection, backend selection, device-specific defaults. `current_platform` is
resolved here, and it is also the hook for out-of-tree hardware plugins (§4).

### Other notable dirs

- `vllm/lora/` — LoRA adapter loading and multi-adapter batched serving
- `vllm/multimodal/` — image/video/audio input processing and registry
- `vllm/transformers_utils/` — HF config/tokenizer glue
- `vllm/tracing/` — OpenTelemetry (`otel.py` appeared in your traceback)
- `vllm/profiler/` — torch profiler integration
- `vllm/plugins/` — the plugin loader (§4)
- `vllm/ray/` — Ray-based multi-node orchestration
- `vllm/benchmarks/` — implementation behind `vllm bench`
- `tests/`, `docs/design/` — `docs/design/` is genuinely good; start with
  `arch_overview.md`, `prefix_caching.md`, `paged_attention.md`

---

## 3. Reading order for someone in your role

1. `docs/design/arch_overview.md` — the map from the authors
2. `v1/core/sched/scheduler.py` — read `schedule()` end to end. Everything about
   throughput, fairness, and preemption is decided here.
3. `v1/core/kv_cache_manager.py` + `block_pool.py` — memory is the binding
   constraint in production; this is how it is spent
4. `v1/worker/gpu_worker.py` `init_device()` → `v1/worker/utils.py` `request_memory()`
   — the startup memory math, and your day-one error
5. `v1/engine/core.py` — the busy loop and the process boundary
6. `config/scheduler.py` + `config/cache.py` — the knobs, with their docstrings

---

## 4. Third-party extensions — how things run "on top of" vLLM

There are **four** distinct extension mechanisms. Knowing which one a given
project uses tells you its failure modes and upgrade risk.

### (a) Python entry-point plugins — `vllm/plugins/__init__.py`

vLLM discovers plugins via `importlib.metadata.entry_points()`. A third-party
package declares an entry point in its own `pyproject.toml` and vLLM loads it at
startup. The groups (from `vllm/plugins/__init__.py`):

| Group | Loaded where | Purpose |
|---|---|---|
| `vllm.general_plugins` | all processes | register custom models, layers, ops |
| `vllm.platform_plugins` | all processes | **out-of-tree hardware backends** |
| `vllm.io_processor_plugins` | front end only | custom input/output processing |
| `vllm.stat_logger_plugins` | front end, async mode | custom metrics sinks |
| `vllm.endpoint_plugins` | API server only | add HTTP routes |

Gated by the `VLLM_PLUGINS` env var (allowlist; unset = load all).

This is how hardware vendors ship support without merging into vLLM:
`vllm-ascend`, `vllm-spyre`, `vllm-openvino` register a `platform_plugins` entry
point and supply their own `Platform` class. Docs: `docs/design/plugin_system.md`.

```python
# in a third-party package's pyproject.toml
[project.entry-points."vllm.general_plugins"]
my_ext = "my_package:register"
```

### (b) KV connectors — `vllm/distributed/kv_transfer/kv_connector/v1/`

**The most important extension point for data-center-scale deployments.** A
connector plugs into the KV cache path with a scheduler-side half and a
worker-side half (documented in `base.py`):

- scheduler side: `get_num_new_matched_tokens()`, `update_state_after_alloc()`,
  `request_finished()`
- worker side: `start_load_kv()`, `wait_for_layer_load()`, `save_kv_layer()`,
  `get_finished()`

Registered through `kv_connector/factory.py` and selected via `--kv-transfer-config`.
Shipped connectors in this tree:

| Connector | What it enables |
|---|---|
| `nixl/` | **NVIDIA NIXL** — RDMA/GPUDirect KV transfer. The transport under disaggregated prefill/decode |
| `lmcache_connector.py`, `lmcache_integration/` | LMCache — multi-tier KV cache (GPU/CPU/disk/remote), cross-node prefix reuse |
| `mooncake/` | Mooncake — Moonshot's disaggregated KV store |
| `offloading_connector.py`, `moriio/`, `hf3fs/`, `flexkv_connector.py` | offloading to CPU/disk/distributed FS |
| `multi_connector.py` | compose several |
| `example_connector.py` | **read this first** — minimal reference implementation |

### (c) KV events — `vllm/distributed/kv_events.py`

vLLM publishes block-level cache events (stored/evicted, with hashes) over ZMQ.
External routers subscribe and route requests to whichever replica already holds
the relevant prefix. This is **KV-aware routing**, and it is how a fleet gets
prefix-cache hit rates that a single node cannot.

### (d) Library embedding

The simplest and most common: import `AsyncLLM` and drive it directly. The
project supplies its own HTTP layer, scheduling, and routing, using vLLM purely
as the inference engine.

---

### So: how does NVIDIA Dynamo run on top of vLLM?

[Dynamo](https://github.com/ai-dynamo/dynamo) is a distributed inference framework
(docs in-tree: `docs/deployment/integrations/dynamo.md`). It does **not** fork or
patch vLLM. It uses (b), (c), and (d) together:

1. **vLLM as an embedded engine** — Dynamo workers instantiate vLLM's `AsyncLLM`
   in-process. vLLM does single-replica inference; Dynamo owns everything above
   that. It supports TRT-LLM and SGLang the same way, which is why the boundary
   is a clean one.
2. **NIXL connector for disaggregation** — Dynamo's headline feature is splitting
   **prefill** and **decode** onto separate GPU pools (compute-bound vs
   bandwidth-bound; they want different hardware and scale independently). The KV
   cache produced by a prefill worker must reach a decode worker, and that
   transfer is `kv_connector/v1/nixl/` doing RDMA. NIXL is itself an NVIDIA
   project (`github.com/ai-dynamo/nixl`) — same org, deliberately co-designed.
3. **KV events for its smart router** — Dynamo subscribes to the ZMQ event stream
   from `kv_events.py` so its router knows which worker already has a given prefix
   cached, then routes accordingly. That is the KV-aware routing in (c).
4. **A planner** for scaling prefill/decode pools independently.

The architecture, then:

```
        client
          │
   Dynamo Frontend (OpenAI-compatible)
          │
   Dynamo Router  ◄── KV events (ZMQ) ── from every vLLM instance
          │
   ┌──────┴────────┐
   ▼               ▼
 Prefill pool    Decode pool
 [vLLM+NIXL] ══▶ [vLLM+NIXL]      ══▶ = KV blocks over RDMA
```

**The operational consequence:** vLLM stays a single-node engine. Everything
multi-node — routing, disaggregation, autoscaling — lives in the layer above and
talks to vLLM through documented interfaces. When you are debugging a Dynamo
deployment, the question "is this a vLLM problem or a Dynamo problem" is usually
answerable by checking whether a single vLLM replica behaves correctly in
isolation.

### Others that sit on vLLM the same way

| Project | Mechanism | Notes |
|---|---|---|
| **Dynamo** | embed + NIXL connector + KV events | NVIDIA; disaggregated serving |
| **LMCache** | KV connector | multi-tier KV cache; also usable standalone |
| **Mooncake** | KV connector | disaggregated KV store |
| **KServe / KubeRay / llm-d** | k8s orchestration around `vllm serve` | no code coupling; just a container |
| **Ray Serve** | library embedding + `vllm/ray/` | multi-node placement |
| **AIBrix, Envoy AI Gateway** | external routing, consumes `/metrics` + KV events | fleet-level routing |
| **vllm-ascend / spyre / openvino** | `vllm.platform_plugins` | out-of-tree hardware |

---

## 5. What is exercisable on this GB10 box

| Feature | Here? | Note |
|---|---|---|
| Continuous batching, PagedAttention | ✅ | examples 01–03 |
| Prefix caching | ✅ | example 04 |
| Structured outputs | ✅ | example 05 |
| Speculative decoding (ngram) | ✅ | example 06 |
| FP8 / NVFP4 quantization | ✅ | sm_121 Blackwell; example 07 |
| KV cache quantization | ✅ | `kv_cache_dtype="fp8"` |
| LoRA multi-adapter | ✅ | `enable_lora=True` |
| Multimodal | ✅ | memory-permitting |
| Metrics / OTel tracing | ✅ | example 11 |
| Tensor / pipeline parallel | ❌ | single GPU |
| Disaggregated prefill/decode | ⚠️ | code-readable; needs ≥2 GPUs to run |
| NIXL / RDMA transfer | ❌ | needs multi-node |
| Dynamo full topology | ❌ | needs a multi-GPU fleet |

The single-GPU constraint is not a blocker for learning the parts that matter
most operationally — scheduling, memory, and failure behavior are all
single-node concerns.
