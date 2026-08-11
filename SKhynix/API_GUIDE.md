# vLLM API Guide — the calls you will actually write

Verified against this checkout (`0.26.1rc1.dev53`, commit `6c7e679`) by introspecting
the installed package. Signatures here are what *this* build accepts; vLLM's API moves,
so re-check with `inspect.signature(...)` after an upgrade.

---

## 0. First: offline vs online is not single-turn vs multi-turn

This trips almost everyone. The two axes are independent.

| | **Offline** (`vllm.LLM`) | **Online** (`vllm serve` + HTTP) |
|---|---|---|
| Where the engine lives | your Python process | separate long-lived server |
| Weight load cost | once **per process** | once **per server lifetime** |
| Multi-turn chat | ✅ `LLM.chat(messages)` | ✅ `/v1/chat/completions` |
| Concurrent clients | ✗ one process, batched list | ✅ many |
| Streaming | ✗ | ✅ |
| Engine introspection | ✅ `llm.llm_engine.vllm_config` | ✗ (metrics only) |
| Use it for | evals, benchmarks, batch jobs | **applications** |

**vLLM is stateless in both modes.** No session ids, no server-side history. The
client resends the entire message list every turn — that list *is* the conversation
state, and you own it. This is also true of OpenAI's API generally.

Consequence: turn *N* re-sends turns 1..*N*-1, so cost grows quadratically over a
conversation. **Prefix caching is what makes multi-turn affordable** — each turn's
prompt is the previous turn's prompt plus more, a perfect cache hit. Verify it with
`RequestOutput.num_cached_tokens` (§3.3).

You want online serving for an interactive app — not because offline can't do turns,
but because reloading weights per process is fatal and you need concurrency.

---

## 1. Offline: the `LLM` class

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", gpu_memory_utilization=0.30)
```

### 1.1 Constructor parameters that matter

Named parameters (the rest pass through `**kwargs` to `EngineArgs`):

| Parameter | Default | Notes |
|---|---|---|
| `model` | — | HF id or local path |
| `tokenizer`, `tokenizer_mode` | auto | override tokenizer |
| `trust_remote_code` | `False` | needed for some architectures |
| `dtype` | `"auto"` | `bfloat16` / `float16` / `float32` |
| `quantization` | `None` | `fp8`, `awq`, `gptq`, `compressed-tensors` |
| `gpu_memory_utilization` | ~0.9 | **fraction of the whole pool. On GB10 start at 0.30** |
| `kv_cache_memory_bytes` | `None` | pin exact KV size — more reproducible than a fraction |
| `cpu_offload_gb` | 0 | offload weights to CPU |
| `enforce_eager` | `False` | `True` skips CUDA graph capture: faster start, slower steps |
| `tensor_parallel_size` | 1 | >1 needs multiple GPUs |
| `seed` | `None` | pin for reproducibility |
| `structured_outputs_config` | `None` | e.g. `{"backend": "xgrammar"}` |
| `compilation_config` | `None` | torch.compile / CUDA graph tuning |
| `logits_processors` | `None` | custom logit processors |
| `spec_method`, `spec_model`, `spec_tokens` | `None` | speculative decoding shorthand |
| `pooler_config` | `None` | embedding/classification models |

Common `**kwargs` (validated `EngineArgs` fields): `max_model_len`,
`max_num_seqs`, `max_num_batched_tokens`, `enable_prefix_caching`,
`kv_cache_dtype`, `enable_lora`, `max_loras`, `max_lora_rank`,
`disable_log_stats`, `kv_transfer_config`.

> `max_model_len` cannot exceed the model's own `max_position_embeddings`.
> `opt-125m` caps at 2048; asking for 4096 fails at startup.

### 1.2 Generation methods

```python
# Text completion. Pass a LIST — vLLM batches it concurrently.
outputs = llm.generate(["prompt one", "prompt two"], sampling_params)

# Multi-turn chat. Applies the model's chat template. Needs an INSTRUCT model —
# base models have no template and this raises.
outputs = llm.chat(
    [{"role": "system", "content": "..."},
     {"role": "user", "content": "..."},
     {"role": "assistant", "content": "..."},   # prior turn
     {"role": "user", "content": "..."}],       # current turn
    sampling_params,
    add_generation_prompt=True,   # default
    chat_template=None,           # override the model's template
    tools=None,                   # tool/function definitions
    chat_template_kwargs=None,    # e.g. {"enable_thinking": False}
)

# Batch of independent conversations
outputs = llm.chat([conv_a, conv_b], sampling_params)
```

Useful kwargs on both: `use_tqdm=False` (silence the progress bar),
`lora_request=LoRARequest(...)`.

### 1.3 Non-generative methods

```python
llm.embed(prompts)           # embedding models -> vectors
llm.classify(prompts)        # classification head
llm.score(data_1, data_2)    # cross-encoder / reranker scoring
llm.encode(prompts)          # raw pooler output
llm.beam_search(prompts, params)
```

### 1.4 Engine control and introspection

```python
cfg = llm.llm_engine.vllm_config      # the whole resolved config
cfg.cache_config.num_gpu_blocks       # KV blocks allocated at startup
cfg.cache_config.block_size           # tokens per block
cfg.model_config.max_model_len
cfg.model_config.dtype

llm.get_metrics()                     # list[Metric] — same data as /metrics
llm.get_tokenizer()
llm.get_default_sampling_params()     # from the model's generation_config.json

llm.reset_prefix_cache()              # clear APC — do this between A/B measurements
llm.sleep(level=1)                    # offload weights, keep the process alive
llm.wake_up()                         # reload them
llm.apply_model(fn)                   # run fn against the nn.Module directly
llm.collective_rpc(...)               # call into every worker
```

`sleep`/`wake_up` matter for RL and multi-model boxes: park a model without paying
full startup again.

### 1.5 Teardown — use the canonical helper

```python
from vllm.distributed import cleanup_dist_env_and_memory

del llm
cleanup_dist_env_and_memory()
```

Do **not** just `del llm; torch.cuda.empty_cache()`. The helper also destroys the
distributed process groups and calls `torch._C._host_emptyCache()`. On GB10 that
host-side release directly returns memory to the pool your next engine needs.
Required whenever one script builds more than one `LLM`.

---

## 2. `SamplingParams`

Per **request**, not per engine. Requests with different params still batch together.

```python
from vllm import SamplingParams

params = SamplingParams(
    temperature=0.0,        # 0 = greedy/deterministic
    top_p=1.0,
    top_k=-1,               # -1 = disabled
    max_tokens=256,         # output cap
    min_tokens=0,
    n=1,                    # samples per prompt (shares the prompt's KV)
    seed=None,
    stop=["\n\n"],          # stop strings
    stop_token_ids=None,
    ignore_eos=False,       # True = always emit max_tokens (benchmarking)
    repetition_penalty=1.0,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    logprobs=None,          # int = return top-N logprobs per token
    prompt_logprobs=None,
    detokenize=True,
    structured_outputs=None,
)
```

For benchmarking always set `temperature=0.0` **and** `ignore_eos=True` — otherwise
you measure sampling variance instead of the thing you changed.

### 2.1 Structured outputs

⚠️ **This build uses `structured_outputs=StructuredOutputsParams(...)`.** Older vLLM
used `guided_decoding=GuidedDecodingParams(...)`. Copy-pasted internet examples will
raise `TypeError`.

```python
from vllm.sampling_params import StructuredOutputsParams

StructuredOutputsParams(
    json=json.dumps(schema),   # or a dict
    # regex=r"...",
    # choice=["a", "b", "c"],
    # grammar="...",           # EBNF
    # json_object=True,        # any valid JSON
    # structural_tag=...,
    disable_any_whitespace=False,
)
```
Exactly one of `json` / `regex` / `choice` / `grammar` / `json_object` / `structural_tag`.

Guarantees **structure, never correctness** — it removes parse failures from your
error budget, not hallucinations.

---

## 3. Reading outputs

### 3.1 `RequestOutput`

```python
out.request_id
out.prompt                    # str
out.prompt_token_ids          # list[int]
out.outputs                   # list[CompletionOutput] — one per n
out.finished
out.num_cached_tokens         # ← prefix cache hits for this request
out.num_cache_creation_tokens
out.metrics
out.lora_request
```

### 3.2 `CompletionOutput`

```python
c = out.outputs[0]
c.index
c.text                        # the generated string
c.token_ids
c.cumulative_logprob
c.logprobs
c.finish_reason               # "stop" | "length" | "abort"
c.stop_reason                 # which stop string/token fired
```

`finish_reason` is worth monitoring: all-`length` means `max_tokens` is clipping
your users mid-sentence.

### 3.3 Proving prefix caching works

```python
outs = llm.chat(messages, params)
o = outs[0]
total = len(o.prompt_token_ids)
print(f"{o.num_cached_tokens}/{total} prompt tokens served from cache")
```
In a multi-turn loop this climbs every turn — direct evidence that the history is
not being recomputed.

⚠️ `num_cached_tokens` is `local + external` (§5.2). With no KV connector
configured it is purely GPU prefix cache; with one, it silently includes blocks
pulled over the network. Read `num_local_cached_tokens` if you need to tell them
apart.

---

## 4. Online: serving and calling

### 4.1 Start a server

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.30 \
  --max-model-len 2048 \
  --max-num-seqs 64 \
  --served-model-name my-model \
  --api-key SECRET
```
Most `LLM(...)` kwargs have a `--kebab-case` CLI equivalent.

### 4.2 Call it with the OpenAI SDK

> **`openai` is a vLLM dependency — you do not install it separately.**
> `requirements/common.txt:17` pins `openai >= 2.0.0`, so it arrives with vLLM
> (2.49.0 in this venv). vLLM doesn't merely *support* the OpenAI client, it
> **imports OpenAI's type definitions for its own API**: `LLM.chat()`'s signature
> references `openai.types.chat.ChatCompletionUserMessageParam`. The message dicts
> you pass to *offline* `chat()` are literally OpenAI's schema — which is why the
> same `messages` list works unchanged in both modes.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# multi-turn: you resend the whole list every time
messages = [{"role": "system", "content": "You are concise."}]
messages.append({"role": "user", "content": "Hello"})

r = client.chat.completions.create(
    model="my-model", messages=messages, temperature=0.0, max_tokens=256,
)
reply = r.choices[0].message.content
messages.append({"role": "assistant", "content": reply})   # ← don't forget this

# streaming
for chunk in client.chat.completions.create(..., stream=True):
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

### 4.3 vLLM-specific request fields

Anything not in the OpenAI spec goes in `extra_body`:

```python
client.chat.completions.create(
    model=..., messages=...,
    extra_body={
        "structured_outputs": {"json": schema},   # or regex / choice
        "ignore_eos": True,
        "top_k": 20,
        "repetition_penalty": 1.05,
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
```

### 4.4 `response_format` vs `structured_outputs` — both exist, they merge

Two ways to constrain output online, and they are **not** alternatives: vLLM
merges them, with `response_format` winning on conflict
(`vllm/entrypoints/openai/engine/protocol.py:179`):

```python
if structured_outputs is None:
    return StructuredOutputsParams(**overrides)   # response_format alone
return replace(structured_outputs, **overrides)   # response_format OVERRIDES
```

| | `response_format` | `structured_outputs` |
|---|---|---|
| Origin | OpenAI standard | vLLM extension |
| Portability | any OpenAI-compatible provider | vLLM only |
| Placement | top-level request field | inside `extra_body` |
| Supports | `text`, `json_object`, `json_schema`, `structural_tag` | `json`, **`regex`**, **`choice`**, **`grammar`**, `json_object`, `structural_tag` |
| Tuning | none | `disable_any_whitespace`, `whitespace_pattern`, `disable_additional_properties` |

```python
# portable — works against OpenAI, Together, vLLM, anything
response_format={"type": "json_schema",
                 "json_schema": {"name": "node", "schema": NODE_SCHEMA}}

# vLLM-only — regex and choice have NO OpenAI equivalent
extra_body={"structured_outputs": {"choice": ["critical", "warning", "healthy"]}}
```

**Rule:** reach for `response_format` when a JSON schema is all you need and you
might swap providers later. Reach for `structured_outputs` when you need regex,
choice, grammar, or the whitespace knobs. Setting both is legal — `response_format`
overrides the overlapping field, and the vLLM-only options survive.

Offline there is no `response_format`; it is `structured_outputs=StructuredOutputsParams(...)`
only (§2.1).

### 4.5 Discovering server flags

`vllm serve --help` does **not** dump every flag — it lists *config groups*, which
is far more usable than scrolling 284 options:

```bash
vllm serve --help                  # the config groups
vllm serve --help=all              # all 284 flags
vllm serve --help=Frontend         # server-facing only
vllm serve --help=CacheConfig      # KV cache
vllm serve --help=SchedulerConfig  # batching knobs
vllm serve --help=ModelConfig
```

Groups: `Frontend`, `ModelConfig`, `LoadConfig`, `AttentionConfig`, `MambaConfig`,
`StructuredOutputsConfig`, `ParallelConfig`, `CacheConfig`, `OffloadConfig`,
`MultiModalConfig`, `LoRAConfig`, `ObservabilityConfig`, `SchedulerConfig`,
`CompilationConfig`. They map 1:1 onto `vllm/config/*.py`, so the group name tells
you which source file to read for the real docstrings.

#### `--enable-prompt-tokens-details`

Off by default. Enables `usage.prompt_tokens_details.cached_tokens` in responses —
the **online equivalent of offline's `RequestOutput.num_cached_tokens`**. Without
it the field is `null` and every request appears to have 0 cached tokens, which
looks exactly like a broken prefix cache.

```bash
vllm serve MODEL --enable-prompt-tokens-details
```
```python
r = client.chat.completions.create(...)
cached = r.usage.prompt_tokens_details.cached_tokens   # null without the flag
```
Turn it on whenever you are measuring cache behavior; leave it off in steady state
if you want the extra per-request stat propagation gone.

### 4.6 Endpoints

| Endpoint | Purpose |
|---|---|
| `/v1/chat/completions` | multi-turn chat |
| `/v1/completions` | raw completion |
| `/v1/embeddings` | embeddings |
| `/v1/models` | what's served — also a readiness probe |
| `/health` | liveness |
| `/metrics` | Prometheus |
| `/tokenize`, `/detokenize` | tokenizer access |

> `/health` and `/v1/models` are what k8s probes hit. In dashboards, **exclude
> them** — otherwise an idle service looks identical to a busy one. That is exactly
> how two abandoned deployments held ~40 GiB on this box for 37 days.

---

## 5. Caching: three different features people conflate

"Cache hit" is ambiguous in vLLM. There are **three separate mechanisms**, in
different directories, solving different problems. Knowing which one you are
looking at changes the diagnosis.

| Feature | What moves | Scope | Lives in |
|---|---|---|---|
| **Prefix caching (APC)** | *nothing* — blocks reused in place | one GPU, one engine | `vllm/v1/core/block_pool.py` |
| **KV offload** | KV blocks → CPU RAM / disk | one node | `vllm/v1/kv_offload/` |
| **KV connector** | KV blocks → another node | cluster | `vllm/distributed/kv_transfer/` |

Plus a fourth thing with a confusingly similar name:

- **Weight offload** (`cpu_offload_gb`, `vllm/config/offload.py`) moves *model
  weights*, not KV cache. Different problem entirely — it is about fitting a model
  that does not fit, not about reusing computation.

### 5.1 Prefix caching — the one you almost always mean

Blocks stay in GPU memory; a hash lookup finds them and prefill is skipped. **Zero
data movement.** On by default in V1.

```python
LLM(model=..., enable_prefix_caching=True)   # default
llm.reset_prefix_cache()                     # clear between A/B measurements
```

This is what produced the multi-turn numbers in `examples/12_multiturn_chat.py`
(0% → 58% → 84% across three turns): each turn's prompt is the previous turn's
prompt plus more, so the blocks were simply still resident. Nothing was offloaded,
nothing crossed a bus.

Cost: cached blocks *occupy* KV cache. Under pressure they are evicted LRU — so hit
rate degrades exactly when you are busiest.

### 5.2 The local vs external split in the metric

`num_cached_tokens` is a **sum**, not a single source
(`vllm/v1/metrics/stats.py:284`):

```python
num_cached_tokens = num_local_cached_tokens + num_external_cached_tokens
```

`PrefillStats` breaks it down:

| Field | Meaning |
|---|---|
| `num_prompt_tokens` | total prompt length |
| `num_computed_tokens` | actually prefilled — **the real work** |
| `num_cached_tokens` | skipped (local + external) |
| `num_local_cached_tokens` | from **GPU prefix cache** |
| `num_external_cached_tokens` | from a **KV connector / offload tier** |
| `num_cache_creation_tokens` | computed *and written* to cache |

This matters diagnostically. A high `num_cached_tokens` that is mostly *external*
means you are pulling blocks over a network or off disk — cheaper than recompute,
but far from free, and with latency variance a local hit never has. With no
connector configured, `num_external_cached_tokens` is always 0 and the two numbers
are identical.

### 5.3 KV offload — extending cache beyond GPU memory

Spills KV blocks to CPU RAM or disk so more history stays cached than GPU memory
alone allows. Trades PCIe/disk latency for avoided recompute. Configured via
`vllm/config/offload.py`; implementation in `vllm/v1/kv_offload/` (`cpu/`,
`tiering/`).

> **On GB10 this is nearly pointless.** Unified memory means "CPU RAM" and "GPU
> memory" are the *same 119.7 GiB pool* — offloading moves blocks from the pool to
> the pool. It is a discrete-GPU optimization. Know it exists for the fleet; do not
> expect it to help here.

### 5.4 KV connectors — sharing cache across nodes

Prefix cache is **per-engine and in-memory**. It does not survive a pod restart,
and replica B cannot use replica A's cache. Connectors fix that:

```bash
vllm serve MODEL --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", ...}'
```

Shipped: `nixl/` (RDMA, the transport under disaggregated prefill/decode),
`lmcache_connector.py`, `mooncake/`, `offloading_connector.py`, `multi_connector.py`.
Start from `example_connector.py`. See `CODE_ANALYSIS.md` §4 for how Dynamo uses these.

### 5.5 Which one is your problem?

- Cache hit rate low on **multi-turn or shared system prompts** → prefix caching.
  Check `enable_prefix_caching`, and whether memory pressure is evicting blocks.
- Hit rate collapses **after every deploy** → per-engine cache is cold on restart.
  That is a connector problem, not an APC problem.
- Hit rate fine on one replica, **zero on the others** → no cross-node sharing;
  you need a connector or KV-aware routing.
- **Model does not fit at all** → that is weight offload or quantization, not any
  of the above.

---

## 6. In-process async: `AsyncLLM`

When you want concurrency *without* a separate server — a custom router, an agent
loop, or your own HTTP layer. This is the same class the OpenAI server uses
internally, and the one frameworks like Dynamo embed.

```python
import asyncio
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.engine.arg_utils import AsyncEngineArgs

engine = AsyncLLM.from_engine_args(
    AsyncEngineArgs(model="Qwen/Qwen2.5-0.5B-Instruct", gpu_memory_utilization=0.30)
)

async def run(prompt, rid):
    async for out in engine.generate(prompt, SamplingParams(max_tokens=64), rid):
        if out.finished:
            return out.outputs[0].text

# concurrent, one warm engine
await asyncio.gather(run("a", "r1"), run("b", "r2"))
```
`generate()` is an async generator yielding incremental `RequestOutput`s — that is
how you stream.

---

## 7. LoRA

```python
from vllm.lora.request import LoRARequest

llm = LLM(model=BASE, enable_lora=True, max_loras=4, max_lora_rank=16)
llm.generate(prompts, params, lora_request=LoRARequest("adapter1", 1, "/path/to/adapter"))
```
Online: `vllm serve BASE --enable-lora --lora-modules name=/path`, then call it as
`model="name"`. Adapters share the base weights, so N adapters ≈ 1 model's memory.

---

## 8. Patterns and gotchas

**Batch, don't loop.** `llm.generate(list_of_prompts)` schedules concurrently.
Calling `generate` once per prompt serializes and throws away vLLM's whole point.

**One `LLM` per process, ideally.** Building several sequentially requires
`cleanup_dist_env_and_memory()` between them. Two live at once will fight for memory.

**Reset the prefix cache between A/B runs.** Otherwise run 2 inherits run 1's cache
and you measure the wrong thing:
```python
llm.reset_prefix_cache()
```

**Append the assistant reply to `messages`.** The single most common multi-turn bug
is generating a reply, never appending it, and wondering why the model has amnesia.

**Trim history.** Context is finite. Truncate, summarize, or drop old turns before
you hit `max_model_len` — vLLM will not do it for you.

**Check `finish_reason`.** `"length"` means you clipped the output.

**On GB10:** `gpu_memory_utilization` is a fraction of a pool shared with the OS.
Start at 0.30 and check `examples/00_env_check.py` for the current ceiling.

---

## 9. Import cheat-sheet

```python
from vllm import LLM, SamplingParams, EngineArgs
from vllm.sampling_params import StructuredOutputsParams
from vllm.outputs import RequestOutput, CompletionOutput
from vllm.lora.request import LoRARequest
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser   # EngineArgs.add_cli_args
```
