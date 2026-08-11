# The examples, explained deeply

How each example works, what mechanism it exercises, and how to read its output.

The code follows upstream vLLM conventions (SPDX header, module-level constants,
`main()`, terse `#` comments, `-` separators) so it reads like the rest of the
repo. Explanation lives both in the comments and here.

```bash
source /home/leeksun/testingEnvironment/vLLM_testing/vllm/.venv/bin/activate
cd /home/leeksun/testingEnvironment/vLLM_testing/vllm/SKhynix/examples
```

Two models are cached locally. `facebook/opt-125m` is the default: tiny, fast, and
a **base** model with no chat template, so `chat()` fails on it.
`Qwen/Qwen2.5-0.5B-Instruct` works for chat — `export VLLM_SK_MODEL=Qwen/Qwen2.5-0.5B-Instruct`.

---

## Part 1 — What actually happens when you call `LLM(...)`

Every offline example spends most of its wall time in one line. Understanding that
line explains most of the output you'll see.

```python
llm = LLM(model=..., gpu_memory_utilization=0.30)
```

In rough order:

1. **Config resolution.** `EngineArgs` merges your kwargs with the model's HF
   config into a `VllmConfig` — the object example 02 introspects.
2. **Process fork.** `EngineCore` starts in a **separate process**, connected by
   ZMQ. This is why tracebacks show `(EngineCore pid=…)` and why a dead engine
   leaves a live-but-useless API server.
3. **Weight load.** Safetensors shards are read and sharded across workers.
4. **Memory profiling.** A real forward pass runs at maximum batch shape to
   *measure* peak activation memory rather than estimate it.
5. **KV cache sizing.** Whatever is left of the budget becomes KV cache:
   ```
   gpu_memory_utilization × TOTAL_POOL − weights − activations − graphs = KV cache
   ```
   That number is divided by the per-block size to yield `num_gpu_blocks`.
6. **CUDA graph capture** (unless `enforce_eager=True`). Decode steps are replayed
   from captured graphs to remove Python launch overhead.

**The failure you already hit lives at step 5.** vLLM compares
`gpu_memory_utilization × TOTAL` against **currently free** memory and refuses to
start if the request exceeds it — deliberately failing fast rather than OOMing
mid-serving.

### Why the examples always pass a list

```python
outputs = llm.generate(prompts, params)     # concurrent
for p in prompts: llm.generate([p], params) # serialized — wrong
```
`generate()` submits every prompt to the scheduler, which interleaves them. The
second form defeats continuous batching entirely.

---

## 00 — `env_check.py`

**Mechanism.** Starts no engine. Calls `torch.cuda.mem_get_info()`, which returns
`(free, total)` for the device — and on GB10 that reflects the *unified* pool,
including OS page cache.

```python
ceiling = free_bytes / total_bytes
suggested = ceiling - 0.15
```

**Why the ceiling is `free/total`.** vLLM's check is
`gpu_memory_utilization × total > free → fail`. Rearranged, the largest legal
utilization is exactly `free/total`. The 0.15 subtraction leaves room for
activations, graph capture and the OS.

**What you'll see that looks wrong but isn't.** Tens of GiB "Used" with no GPU
processes listed. That's page cache in the shared pool. It also means **the
ceiling moves between runs** — a value that worked an hour ago can fail now.

---

## 01 — `hello_offline.py`

**Mechanism.** The three-step shape, plus two details worth internalizing.

**`finish_reason`.** Each `CompletionOutput` carries `"stop"` (model chose to end),
`"length"` (hit `max_tokens`), or `"abort"`. All three prompts here return
`"length"` because opt-125m rambles. In production, all-`"length"` means you're
truncating users mid-sentence.

**`n=2` shares prefill.** Two samples from one prompt means the prompt's KV blocks
are computed **once** and both sequences decode from them. That's why `n` is much
cheaper than issuing two separate requests with the same prompt.

**The tok/s number is deliberately not a benchmark.** 144 tokens in ~0.09s on 3
prompts is latency-bound: the GPU is idle most of that time waiting for a
sequential decode loop. Example 03 measures throughput properly by saturating it.

---

## 02 — `memory_and_kv_cache.py`

The most important example for your role. **Offline only** — `num_gpu_blocks` has
no HTTP equivalent, by design.

**Mechanism.** Builds the same model at three utilizations and reads the resolved
config:

```python
config     = llm.llm_engine.vllm_config
num_blocks = config.cache_config.num_gpu_blocks     # decided at step 5 above
block_size = config.cache_config.block_size         # tokens per block, default 16
```

**The per-token cost, derived.** Every token must cache a key **and** a value
(hence `2`), for every layer, for every KV head:

```
KV bytes/token = 2 × num_layers × num_kv_heads × head_dim × dtype_bytes
```

For opt-125m: `2 × 12 × 12 × 64 × 2 = 36,864 bytes/token` — verified by running it.

**Why `num_kv_heads`, not `num_attention_heads`.** Modern models use grouped-query
attention: many query heads share one KV head. Llama-3-8B has 32 query heads but
only 8 KV heads, so its KV cache is 4× smaller than a naive calculation suggests.
Using the wrong field overestimates by that ratio.

**Measured at utilization 0.35:** 74,768 blocks × 16 tokens = **1,196,288 tokens**
of KV cache (41 GiB). That divides into:
- 584 concurrent sequences at full 2048 context
- 4,673 concurrent sequences at 256 tokens each

**The planning identity:**
```
concurrency = KV capacity (tokens) / average context length
```
Both halves are tunable. Halving `max_model_len` doubles concurrency at zero
memory cost — usually the cheapest capacity win available, and a product
negotiation rather than an engineering one.

**Caveat stated in the file:** `enforce_eager=True` removes CUDA graph memory from
the budget, so these are upper bounds. Production numbers are somewhat lower.

**Also worth knowing:** `kv_cache_memory_bytes` pins an exact cache size instead of
deriving it from a fraction — far more reproducible across heterogeneous nodes,
and what example 13 uses to force eviction deterministically.

---

## 03 — `batching_throughput.py`

**Mechanism.** vLLM's scheduler ([`vllm/v1/core/sched/scheduler.py`](../vllm/v1/core/sched/scheduler.py))
runs a loop: each step it decides which requests to run, admits waiting ones if
there's room, and evicts if there isn't. There is no fixed batch.

Each step is bounded by two independent limits:
- `max_num_seqs` — how many sequences may be resident
- `max_num_batched_tokens` — how many tokens may be processed

Prefill is token-heavy (a 512-token prompt is 512 tokens of work in one step);
decode is sequence-heavy (each sequence contributes exactly 1 token per step). So
**prefill hits the token limit and decode hits the sequence limit** — which is why
both knobs exist and why tuning one alone stalls.

**Measured on this box:**

| max_num_seqs | max_batched_tokens | output tok/s |
|---|---|---|
| 8 | 2048 | 4,622 |
| 64 | 4096 | 21,321 |
| 256 | 8192 | **34,135** |

**7.4x from scheduler configuration alone** — same model, same hardware, same
prompts. This is why "vLLM is slow" is usually a config statement.

**Why throughput rises with batch size.** Decode is memory-bandwidth bound: you
read the whole model to produce one token per sequence. Reading it once to serve
256 sequences amortizes that read 256 ways. The GPU was idle waiting on memory;
batching fills the gap.

**Where it stops rising — preemption.** Past some concurrency, admitted sequences
need more KV blocks than exist. The scheduler **preempts**: evicts a request's
blocks and re-queues it, so that work is done twice. Throughput doesn't plateau,
it *falls*, and p99 latency spikes while p50 looks fine — the classic signature.
Grep the engine log for `Preempted`.

**The limitation this example cannot escape.** It hands all 256 requests over at
once: a **closed** system where offered load is capped by your own throughput.
Production is an **open** system — arrivals continue regardless of whether you keep
up, so queues grow without bound past capacity. Latency percentiles from a closed
test are meaningless. Use `vllm bench serve` (example 08 §4) for the real thing.

---

## 04 — `prefix_caching.py`

**Mechanism.** vLLM hashes KV blocks by their content chain. Block *n*'s hash
covers its tokens **plus the hash of block *n−1***, so identical prefixes produce
identical hashes and diverging suffixes do not collide. On admission the scheduler
looks up each block hash in the pool ([`block_pool.py`](../vllm/v1/core/block_pool.py));
hits are mapped into the sequence's block table and skipped during prefill.

**Nothing is copied.** The blocks were already in GPU memory; a hit is a pointer
lookup. That's what distinguishes prefix caching from offloading (example 13).

**Measured:**

| | Prompt tokens | Served from cache | Wall time |
|---|---|---|---|
| APC off | 20,880 | 0 | 0.31s |
| APC on | 20,880 | **20,448 (98%)** | 0.14s |

**2.15x**, and the mechanism is visible: 98% of prompt tokens were never computed.

**Why the win is bounded by prefill share.** Only prefill is skipped; decode is
untouched. So the speedup ceiling is the fraction of total time spent in prefill.
Long shared prefix with short outputs → large win. Short prompt with long
generation → almost none.

**Three operational consequences:**
1. Cached blocks **occupy** KV cache and are evicted LRU. Hit rate therefore
   degrades exactly when you're busiest — the opposite of what you want.
2. The cache is **per-engine and in-memory**. Every pod restart cold-starts it,
   which makes rollouts a latency event, not just an availability one.
3. Sharing across replicas requires a **KV connector**; extending to CPU/disk on
   one node is **example 13**.

**Measuring it correctly:** call `llm.reset_prefix_cache()` between arms, or the
second run inherits the first's cache and you measure nothing.

---

## 05 — `structured_outputs.py` · dual mode

```bash
python 05_structured_outputs.py --mode offline
python 05_structured_outputs.py --mode online
```

**Mechanism.** A grammar is compiled to a state machine. At each decode step the
backend computes which tokens are legal **in the current state** and writes `-inf`
into the logits of every other token before sampling. Illegal tokens therefore have
zero probability — malformed output is not unlikely, it is *impossible*.

Backends live in [`vllm/v1/structured_output/`](../vllm/v1/structured_output/):
`xgrammar`, `guidance`, `outlines`, `lm-format-enforcer`.

**The lesson is demo 2.** With a small model the incident classifications come out
**semantically wrong** — and *always* exactly one of `critical`/`warning`/`healthy`.

> Constrained decoding guarantees **structure**, never **correctness**. It removes
> parse failures from your error budget. It does not remove hallucinations.

That distinction matters when someone proposes JSON mode as a hallucination fix.

**Reading demo 4 honestly.** The constrained run often measures *faster*. That is an
artifact: the grammar forces a closing brace so the request terminates, while the
unconstrained run rambles to `max_tokens`. Different output lengths — not a valid
comparison. To isolate the per-step mask cost, fix output length with `ignore_eos`
and warm the grammar cache first. Grammar compilation is one-time and cached, so
first-request cost badly overstates steady state.

**API trap.** This build uses `structured_outputs=StructuredOutputsParams(...)`.
Older vLLM used `guided_decoding=GuidedDecodingParams(...)`, so most examples you
find online raise `TypeError`.

---

## 06 — `speculative_decoding.py`

**Mechanism.** Decode reads the entire model to produce **one** token — enormously
memory-inefficient. Speculation:

1. A cheap proposer suggests *k* tokens.
2. The target model verifies all *k* in **one** forward pass (they're processed in
   parallel, like prefill).
3. Accepted tokens are kept; the first rejection discards the rest.

If all *k* are accepted you got *k* tokens for one model read.

**Why `ngram` needs no draft model.** It searches the existing context for a
matching n-gram and proposes the continuation that followed last time. Free, zero
extra memory — and it only works when output **echoes input**. That's why the
prompts here are a document plus questions whose answers quote it. That shape is
RAG, summarization, and code editing; it is not open-ended chat.

**Measured: 1.64x–1.88x** across runs on those grounded prompts.

**Acceptance rate decides everything.** Every rejected proposal is wasted
verification compute. Below roughly 30% acceptance, speculation is a net loss.
*(That threshold is my rule of thumb, not vLLM documentation — calibrate on your
own traffic.)*

**The production trade nobody mentions.** Speculation converts **compute into
latency**. On an idle server it's free latency improvement. On a **saturated**
server that compute was doing useful work for other requests, so total throughput
falls while individual latency improves. Decide which you're optimizing first.

Draft-model methods (eagle, medusa) load a second model, taking budget away from
KV cache — re-run example 02 after enabling one.

---

## 07 — `quantization.py`

**Starts no engine.** Pure arithmetic against live free memory.

**Mechanism.** `weights_GiB = params × bytes_per_param`. BF16 is 2 bytes, FP8 is 1,
INT4/NVFP4 is 0.5. The 60% ceiling reserves room for KV cache and activations.

**The worked case.** `meta-llama/Llama-3.1-70B` sits in your HF cache at **263 GB**
of BF16 shards:

| Format | Weights | Verdict |
|---|---|---|
| BF16 | 132 GiB | **impossible** — exceeds the 119.7 GiB pool |
| FP8 | 66 GiB | fits the pool, not today's free memory |
| INT4/NVFP4 | 33 GiB | comfortable |

**The point:** quantization is a **capacity decision, not a tuning knob**. It
determines what runs at all, before any performance discussion.

**KV cache quantization is the other half.** Weights are fixed; KV cache scales with
concurrency × context. On long-context serving it routinely exceeds the weights, so
`kv_cache_dtype="fp8"` roughly doubles concurrent sequences. Verify with example 02.

**Blackwell relevance:** this box is sm_121, so FP8 and NVFP4 kernels both exist.
Pre-Hopper hardware has neither and must use AWQ/GPTQ INT4.

---

## 08 — `online_serving.{txt,py}`

Two halves. **`.txt`** is the reference: every command written out for running by
hand. **`.py`** is a harness that does it for you:

```bash
python 08_online_serving.py                 # full run
python 08_online_serving.py --skip-bench    # faster
```

It launches a server, polls `/health` until ready, inspects `/v1/models`, sends
completion / chat / streaming requests, runs `vllm bench serve`, scrapes
`/metrics`, then tears the server down in a `finally` block.

Two implementation details worth copying:
- The server is launched with `start_new_session=True`, giving it its own process
  group. Teardown signals the whole group without touching the harness, so a
  stray Ctrl-C cannot orphan a GPU-holding process.
- Teardown sends **SIGTERM, not SIGKILL**. vLLM installs handlers for SIGTERM and
  SIGINT and unwinds the engine cleanly; SIGKILL skips that and can leave shared
  memory and the EngineCore child behind. SIGKILL is only the fallback after a
  60-second timeout.

It also invokes the server as `python -m vllm.entrypoints.cli.main serve` rather
than the `vllm` console script, so it works whether or not the venv is on PATH.

**Measured on this box** (Qwen2.5-0.5B, request-rate 5): mean TTFT 25 ms, p99
TTFT 31 ms, mean ITL 6.0 ms, p99 ITL 10.2 ms. Streaming TTFT measured
independently at 0.018 s.

**The section that matters is `vllm bench serve`.** Everything offline measures a
closed system. This models **Poisson arrivals at a fixed rate** and reports
TTFT/ITL/E2E percentiles, which is what SLAs are written against.

**Capacity planning procedure:**
1. Sweep `--request-rate` upward.
2. Find where p99 TTFT breaks your SLA — the knee.
3. `replicas = peak_RPS / rate_at_knee`, plus headroom for rollouts and failure.

`--request-rate inf` is a *saturation* test: offered load is unbounded, so
throughput is meaningful and latency is not.

**Diagnostic table:**

| Symptom | Cause | Action |
|---|---|---|
| TTFT rising, throughput flat | queueing | add replicas |
| ITL rising | batch too large | lower `max_num_seqs` |
| throughput collapsing | KV exhaustion | check `Preempted`, raise memory |

**Also flags:** `/health` and `/v1/models` are probe endpoints. Exclude them from
dashboards or an idle service is indistinguishable from a busy one — the exact
blind spot that hid two dead deployments for 37 days.

---

## 09 — `ebpf_observability.{txt,py}`

**`.txt`** explains what eBPF can and cannot see. **`.py`** runs a chosen probe
for a fixed duration:

```bash
python 09_ebpf_observability.py --list
sudo python 09_ebpf_observability.py --recipe runq-latency --seconds 15
```

Six recipes: `page-faults`, `runq-latency`, `gpu-ioctl`, `block-io`,
`tcp-accept`, `offcpu`. The harness checks for bpftrace and for root before
attaching, and enforces the time limit itself — bpftrace has no total-duration
flag, so the runner sends SIGTERM, which makes it print its maps and exit
cleanly.

**The honest framing, stated up front:** vLLM has **no eBPF hooks** and will not
get any. eBPF sees syscalls, page faults, block I/O, TCP, and scheduler latency —
and stops at the driver boundary. It sees **nothing** inside the GPU: no kernel
execution, no SM occupancy, no HBM bandwidth.

| Layer | Tool |
|---|---|
| Inside the GPU | DCGM, NSight, CUPTI |
| Inside the engine | vLLM `/metrics` |
| Between NIC and driver | eBPF |

That third row is where "GPU looks idle but latency is bad" lives.

**Recipe 2 is the GB10-specific one.** Unified memory means host memory pressure
*is* GPU memory pressure. Major faults during serving mean you have squeezed the
host and latency will be unpredictable — a failure mode that does not exist on
discrete-GPU boxes.

---

## 10 — `chaos_and_limits.{txt,py}`

**`.txt`** describes seven scenarios. **`.py`** automates five of them, each
managing its own server so nothing is left holding GPU memory:

```bash
python 10_chaos_and_limits.py --list
python 10_chaos_and_limits.py --scenario startup-oom
```

| Scenario | Verified result on this box |
|---|---|
| `startup-oom` | exits code 1 after 14 s with the exact `Free memory on device cuda:0` ValueError |
| `kv-exhaustion` | saturating load against a 64 MiB cache; reports preemption count |
| `engine-kill` | `/health` stopped answering 200 **1.0 s** after EngineCore was killed |
| `bad-requests` | oversized prompt, huge `max_tokens`, invalid temperature — all clean **HTTP 400**; `num_requests_running` returned to 0 after a mid-stream disconnect |
| `idle-detection` | probe-only traffic leaves `generation_tokens_total` flat |

That `engine-kill` number is directly actionable: it is your worst-case blackhole
window, so liveness probe `periodSeconds` should sit below it.

**Scenario 2 is the most instructive.** vLLM does **not** crash when memory pressure
appears *during* serving, because its KV cache was pre-allocated at startup. But a
**new** replica cannot start under that pressure.

> vLLM is resilient to memory pressure while running, and fragile to it at startup.
> **Rollouts are therefore the dangerous window**, not steady state.

**Scenario 7 — the 37-day problem.** Both abandoned deployments passed health checks
for a month while doing zero work. The detection that would have caught them:

```
GPU memory reserved > X  AND  rate(vllm:generation_tokens_total) == 0 for N hours
```

Request count cannot work — probes inflate it. `generation_tokens_total` is the only
honest liveness signal, and this alert generalizes to every model server you run.

---

## 11 — `metrics.py`

```bash
python 11_metrics.py            # snapshot with interpretation
python 11_metrics.py --watch    # live table
```

**Mechanism.** Scrapes `/metrics`, parses the Prometheus text format, sums across
label sets, and annotates ~13 series with what to *do* about each.

**Counter naming.** `prometheus_client` appends `_total` to Counters, so a metric
declared `vllm:num_preemptions` in
[`loggers.py`](../vllm/v1/metrics/loggers.py) is scraped as
`vllm:num_preemptions_total`. Gauges have no suffix; histograms expose
`_sum`/`_count`/`_bucket`.

**Two names I got wrong initially**, worth checking in source rather than docs:
- `vllm:kv_cache_usage_perc` — *not* `gpu_cache_usage_perc`
- `vllm:inter_token_latency_seconds` — *not* `time_per_output_token_seconds`

**The alerting shortlist:**

| Metric | Why |
|---|---|
| `num_requests_waiting` | sustained > 0 is the primary saturation signal |
| `num_preemptions_total` | leading indicator of the throughput cliff |
| `kv_cache_usage_perc` | near 1.0 means preemption is imminent |
| `generation_tokens_total` | the only honest "doing work" signal |
| `external_prefix_cache_hits_total` | nonzero only with offload/connector |

---

## 12 — `multiturn_chat.py` · dual mode

```bash
export VLLM_SK_MODEL=Qwen/Qwen2.5-0.5B-Instruct
python 12_multiturn_chat.py --mode offline
python 12_multiturn_chat.py --mode online
python 12_multiturn_chat.py --mode online --interactive
```

**The reference for the offline/online question.** Identical behaviour both ways.

**Mechanism.** `LLM.chat()` applies the model's Jinja chat template to your message
list, producing one flat prompt string, then calls the same path as `generate()`.
The online server does exactly the same thing inside the request handler. Base
models have no template, which is why opt-125m raises here.

**vLLM is stateless.** No session ids, no server-side history:

```python
messages.append({"role": "user", "content": turn})
reply = backend.chat(messages)                              # send EVERYTHING
messages.append({"role": "assistant", "content": reply})    # omit = amnesia
```

That last append is the most common bug in hand-rolled chat loops.

**The cost curve, and its rescue.** Turn *N* resends turns 1..*N*−1, so tokens sent
grow quadratically. Prefix caching converts that back to near-linear real compute,
because each turn's prompt is the previous turn's prompt plus more:

| Turn | Prompt tokens | From cache | Hit rate |
|---|---|---|---|
| 1 | 45 | 0 | 0% |
| 2 | 84 | 48 | **57%** |
| 3 | 125 | 96 | **77%** |

Measured identically offline and online. The rate climbs toward 100% as
conversations lengthen — which is precisely why APC is not optional for production
chat.

Online reporting needs `--enable-prompt-tokens-details`; without it
`prompt_tokens_details` is `null` and a working cache looks broken.

---

## 13 — `kv_offload.py`

**What offloading adds.** Prefix caching only reuses blocks **still in GPU memory**.
When the cache fills, blocks are evicted LRU and that history is gone. Offloading
copies completed blocks to **pinned host memory** as they are produced; a later
match is promoted back to GPU instead of recomputed. Transfers are DMA
(`cudaMemcpyAsync`) and overlap with compute.

**How this example forces the condition.** Demonstrating offloading requires
eviction to actually happen — and that is harder than it sounds:

```python
KV_CACHE_BYTES = 48 * 1024 * 1024   # pin a tiny cache; 256 blocks
```

My first attempt used `gpu_memory_utilization=0.10`, which on a 119.7 GiB pool is
still ~10 GiB of KV cache. Nothing was ever evicted, both arms showed identical GPU
hits, and offloading looked useless. **If your cache never fills, offloading does
nothing by definition.** Pinning `kv_cache_memory_bytes` makes eviction
deterministic rather than hoped-for.

Then: generate the target → flood with 40 unrelated long prompts → request the
target again. Anything cached now had to survive eviction.

**Measured:**

| | Prompt tokens | Cached after eviction | Recomputed |
|---|---|---|---|
| No offloading | 904 | 0 | **904** |
| With offloading | 904 | **896** | 8 |

**896 tokens recovered from the CPU tier** that would otherwise have been recomputed.

**Config anatomy:**
```python
{"kv_connector": "OffloadingConnector",       # delivered as a KV connector
 "kv_role": "kv_both",                        # this instance both stores and loads
 "kv_connector_extra_config": {
     "block_size": 64,                        # offload chunk, multiple of GPU block size
     "cpu_bytes_to_use": 4_000_000_000}}      # TOTAL across workers, not per-worker
```

**Sizing rule:** `cpu_bytes_to_use` must **exceed** the aggregate GPU KV cache.
Offloading is immediate, so a smaller CPU tier merely mirrors what the GPU already
holds and adds no hit rate at all.

**Two gotchas found while building this:**
- `get_metrics()` raises `AssertionError: Stat logging disabled` offline unless you
  pass `disable_log_stats=False`. Offline `LLM` disables stats by default.
- `external_prefix_cache_hits` reported **0** even though 896 tokens demonstrably
  came from the CPU tier. Trust the per-request `num_cached_tokens`.

**On GB10 this is a mechanism demo, not a win.** Unified memory means host RAM and
GPU memory are the same pool, so you pay a DMA copy to free nothing. It is a
discrete-GPU optimization. Full server configs, disk/object/P2P tiers and the
`PYTHONHASHSEED` trap are in [KV_OFFLOAD.md](KV_OFFLOAD.md); the authoritative
reference is [`docs/features/kv_offloading_usage.md`](../docs/features/kv_offloading_usage.md).

---

## `_backend.py` — not an example

Scaffolding letting 05 and 12 run both ways from one file. vLLM ships nothing like
it **deliberately**; upstream keeps the modes in separate files. It exists only to
prove the same conversation code works in both. Don't ship the pattern — pick a mode
and write to it directly. Rationale in [MODES.md](MODES.md).

---

## Suggested order

1. **00 → 01 → 02** — memory is the binding constraint; start there
2. **03 → 08** — scheduling, then real (open-system) serving benchmarks
3. **12** — how conversations actually work
4. **04 → 13** — caching in GPU, then beyond it
5. **05, 07** — the features you'll use most
6. **10, 11** — failure and monitoring
7. **06, 09** — specialist topics

**02, 03, and 12** are the three that most change how you think about running this
in production.
