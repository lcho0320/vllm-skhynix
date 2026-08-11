# Offline vs Online — which examples run in which mode, and why

## The correction first

**Multi-turn conversation does not require online serving.** These are independent axes:

```
offline  vs  online       =  WHERE the engine lives  (in-process vs HTTP server)
single   vs  multi-turn   =  HOW MANY messages you put in the list
```

`LLM.chat(messages)` does full multi-turn offline, in your own process. Proven in
[12_multiturn_chat.py](examples/12_multiturn_chat.py), which runs identically in both modes.

**vLLM is stateless in both modes.** It never stores a conversation. The client
resends the entire message list every turn; that list *is* the state. There is no
session id and no server-side memory — the same is true of OpenAI's own API.

### So when do you actually need online?

Not for turns — for these:

| Need | Why offline fails |
|---|---|
| An interactive application | weights reload on every process start (seconds to minutes) |
| More than one concurrent user | one process, one batch call |
| Streaming tokens to a UI | `generate()` returns only when finished |
| Language-agnostic clients | it's a Python object, not a protocol |
| Independent deploy/scale | engine lifetime is tied to your script |

**Rule of thumb: use offline to MEASURE, online to SERVE.**

---

## The measured payoff: prefix caching across turns

From `12_multiturn_chat.py` on this box (Qwen2.5-0.5B-Instruct), identical in both modes:

| Turn | Prompt tokens | From prefix cache | Hit rate |
|---|---|---|---|
| 1 | 45 | 0 | 0% |
| 2 | 83 | 48 | **58%** |
| 3 | 134 | 112 | **84%** |

Each turn resends everything before it, so the hit rate climbs toward 100% as the
conversation grows. That is what converts the quadratic re-send into near-linear
real compute — and it is why APC is not optional for production chat.

Measure it yourself:
- offline: `RequestOutput.num_cached_tokens`
- online: `usage.prompt_tokens_details.cached_tokens`, **but only if the server was
  started with `--enable-prompt-tokens-details`** (it is off by default and the field
  is `null` without it)

---

## Mode support per example

| # | Example | Offline | Online | Notes |
|---|---|---|---|---|
| 00 | `env_check` | n/a | n/a | Host preflight — no engine involved |
| 01 | `hello_offline` | ✅ | — | The core `LLM`/`SamplingParams` loop. Online equivalent is example 08 |
| 02 | `memory_and_kv_cache` | ✅ | ✗ | **Offline only by necessity** — reads `llm.llm_engine.vllm_config` for `num_gpu_blocks`. No HTTP endpoint exposes this |
| 03 | `batching_throughput` | ✅ | → 08 | Offline measures a fixed batch. Real serving load = `vllm bench serve` (§4 of example 08) |
| 04 | `prefix_caching` | ✅ | → 12 | Offline A/B needs two engines with different flags. For the online view, see the cache table above |
| 05 | `structured_outputs` | ✅ | ✅ | Dual mode via `--mode` |
| 06 | `speculative_decoding` | ✅ | ✗ | A/B requires building two engines with different configs. Online = a server flag (`--speculative-config`), not a per-request option |
| 07 | `quantization` | n/a | n/a | Memory arithmetic — no engine started |
| 08 | `online_serving` | — | ✅ | Server operation and benchmarks |
| 09 | `ebpf_observability` | — | ✅ | Traces a running server process |
| 10 | `chaos_and_limits` | — | ✅ | Failure injection against a server |
| 11 | `metrics` | ✗ | ✅ | `/metrics` is an HTTP endpoint. Offline has `llm.get_metrics()` instead |
| 12 | `multiturn_chat` | ✅ | ✅ | **Dual mode** — the reference for this whole question |

### Why some examples are deliberately single-mode

Mechanically dualizing everything would produce worse material, not better:

- **02 and 06 need engine internals or multiple engine configs.** The HTTP API
  intentionally hides `num_gpu_blocks` and won't let a request change the
  speculative-decoding config. An "online version" would be a different, weaker
  experiment, not the same one.
- **11 is online by definition.** `/metrics` is HTTP. The offline analogue is
  `llm.get_metrics()`, already noted in [API_GUIDE.md](API_GUIDE.md) §1.4.
- **00 and 07 never start an engine.** Nothing to switch.
- **03's honest online counterpart is `vllm bench serve`** — a Poisson arrival
  process with TTFT/ITL percentiles. Re-implementing that by hand would be strictly
  worse than the tool that ships in the repo. Example 08 §4 covers it.

---

## Running dual-mode examples

```bash
source /home/leeksun/testingEnvironment/vLLM_testing/vllm/.venv/bin/activate
cd /home/leeksun/testingEnvironment/vLLM_testing/vllm/SKhynix/examples

# offline — engine in this process
VLLM_SK_MODEL=Qwen/Qwen2.5-0.5B-Instruct python 12_multiturn_chat.py --mode offline

# online — start the server first
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --gpu-memory-utilization 0.30 --max-model-len 2048 \
  --port 8000 --enable-prompt-tokens-details

VLLM_SK_MODEL=Qwen/Qwen2.5-0.5B-Instruct python 12_multiturn_chat.py --mode online

# interactive REPL (streams token-by-token in online mode)
python 12_multiturn_chat.py --mode online --interactive
```

Flags provided by [`_backend.py`](examples/_backend.py) on every dual-mode example:
`--mode {offline,online}`, `--model`, `--url`, `--gpu-memory-utilization`,
`--max-model-len`. Environment overrides: `VLLM_SK_MODE`, `VLLM_SK_MODEL`, `VLLM_SK_URL`.

---

## Why `_backend.py` exists — and why vLLM has nothing like it

`_backend.py` is **ours, not vLLM's.** vLLM ships no offline/online abstraction, and
that is a deliberate design position on their part, not an oversight. Worth
understanding before you copy the pattern.

### How vLLM's own examples do it

Upstream keeps the two modes in **separate files that share no code**:

```
examples/basic/offline_inference/chat.py             -> builds vllm.LLM directly
examples/basic/online_serving/openai_chat_completion_client.py -> plain OpenAI()
```

The offline one plumbs engine config through vLLM's own CLI machinery:

```python
from vllm import LLM, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

parser = FlexibleArgumentParser()
EngineArgs.add_cli_args(parser)          # every engine flag, for free
args = parser.parse_args()
llm = LLM(**vars(args))
```

The online one is ~15 lines of `OpenAI(base_url=...)` with no vLLM import at all —
because a client genuinely does not need one.

### Why upstream is right to keep them apart

The two modes are **not the same API wearing different clothes**:

| | Offline | Online |
|---|---|---|
| Config | engine construction args (`gpu_memory_utilization`, `max_model_len`) | server-side, fixed before the client connects |
| Batching | you hand it a list | server schedules across clients |
| Errors | Python exceptions | HTTP status codes |
| Introspection | full (`vllm_config`, `get_metrics()`) | none beyond `/metrics` |
| Streaming | ✗ | ✅ |
| Lifecycle | you own it | someone else owns it |

An abstraction over both has to **hide the differences that actually matter**.
`_backend.py` cannot expose `num_gpu_blocks` (online has no such concept), cannot
expose `gpu_memory_utilization` online (it was fixed at server start), and has to
fake `chat_stream` as offline-unavailable.

### So why did we write one anyway

One reason only: **to prove that the same conversation code runs unchanged in both
modes.** That is a teaching claim, and the cleanest way to demonstrate it is one
file with a `--mode` switch producing identical output. `12_multiturn_chat.py`
returns the same three answers and the same cache-hit curve either way — which is
the whole argument that multi-turn is not an online-only capability.

### Do not ship this pattern

For real code, **pick a mode and write to it directly**:

```python
# application code — just use the client
from openai import OpenAI
client = OpenAI(base_url="http://vllm.internal/v1", api_key=KEY)

# eval / benchmark harness — just use LLM
from vllm import LLM
llm = LLM(model=..., gpu_memory_utilization=0.30)
```

The indirection costs you: real `SamplingParams` (we expose ~6 of ~20 fields),
real error types (we collapse them), engine introspection, and streaming.
`_backend.py` is scaffolding for a demonstration, not an architecture.

If you *do* want one interface over a local and a remote engine in production, the
right shape is the reverse of ours: **write to the OpenAI client interface only**,
and run a local `vllm serve` for development. Then there is no abstraction at all —
just a different `base_url`.

## Use an instruct model for chat

`facebook/opt-125m` is a **base** model with no chat template — `LLM.chat()` raises
`"default chat template is no longer allowed"`. The examples catch this and tell you
what to do. `Qwen/Qwen2.5-0.5B-Instruct` is downloaded on this box and works in both modes.
