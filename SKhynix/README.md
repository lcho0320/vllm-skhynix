# vLLM on GB10 — Study & Test Kit

Working notes and runnable examples for learning vLLM well enough to run it as
data-center infrastructure. Built against the local editable checkout.

## This machine

| Property | Value | Why it matters |
|---|---|---|
| Host | `spark-fc72`, NVIDIA DGX Spark (GB10) | arm64 / aarch64 — some wheels are x86-only |
| Compute capability | `sm_121` (Blackwell) | FP8 and NVFP4 paths are available; Hopper-only kernels are not |
| Memory | **119.7 GiB unified** (CPU+GPU share one pool) | The single most important difference from an A100/H100 box — see below |
| GPUs | 1 | `tensor_parallel_size` > 1 is not testable here; DP/PP only in simulation |
| CUDA / torch | 13.x / `2.13.0+cu132` | `nvidia-smi` reports `memory.total = [N/A]` on GB10 |
| vLLM | editable install @ `6c7e679` | `pip install -e .` — edits to `vllm/` take effect on next run |
| venv | `../.venv` | `source /home/leeksun/testingEnvironment/vLLM_testing/vllm/.venv/bin/activate` |

### Unified memory is the headline difference

On a discrete GPU, `gpu_memory_utilization=0.9` means "90% of dedicated VRAM,"
and host RAM is unaffected. On GB10 there is **one 119.7 GiB pool shared with the
operating system**. So:

- `gpu_memory_utilization` is a fraction of a pool the OS is also living in.
  0.92 (the default) asks for ~110 GiB and leaves ~10 GiB for everything else.
- vLLM's preflight check compares your request against *currently free* memory,
  not total. Anything else resident — another pod, page cache, your own Python —
  shrinks the ceiling. This is the exact error you hit on day one:
  `Free memory on device cuda:0 (56.65/119.7 GiB) ... less than desired (0.92, 110.12 GiB)`.
- Practical rule here: **start at 0.30–0.50** for small models, raise
  deliberately, and check `free -g` and `nvidia-smi` first. `examples/00_env_check.py`
  does this for you.
- Model weights that would need host↔device copies on a discrete GPU do not here.
  Load times are fast; bandwidth characteristics differ from an H100 — don't
  extrapolate GB10 throughput numbers to a discrete-GPU fleet.

### Models already in the HF cache

- `facebook/opt-125m` — 250 MB, the default for every example here. Fast enough
  to iterate on knobs without waiting on model load.
- `meta-llama/Llama-3.1-70B` — 263 GB of BF16 weights. **This does not fit.** 70B
  in BF16 needs ~140 GiB of weights alone against a 119.7 GiB shared pool. It is
  a useful teaching artifact: to run a 70B here you need quantization (FP8 ≈ 70 GiB,
  INT4/NVFP4 ≈ 35–40 GiB). See `examples/07_quantization.py`.

## Layout

```
SKhynix/
├── README.md          ← you are here
├── guide.md           ← study roadmap: what to learn, in what order (start here)
├── API_GUIDE.md       ← the vLLM calls you will actually write, verified.
│                        also: response_format vs structured_outputs, server-flag
│                        discovery, and the 3 caching mechanisms people conflate (§5)
├── EXAMPLES.md        ← what every example does and what to look at. READ THIS
│                        if the example files are opaque
├── MODES.md           ← offline vs online per example (multi-turn works in BOTH),
│                        plus why _backend.py is scaffolding and not a ship pattern
├── KV_OFFLOAD.md      ← server setup for KV offloading (CPU/disk tiers), verified
├── CODE_ANALYSIS.md   ← where every subsystem lives, and how Dynamo/LMCache attach
└── examples/
    ├── 00_env_check.py            preflight: memory, capability, safe utilization
    ├── 01_hello_offline.py        LLM() + SamplingParams, the core loop
    ├── 02_memory_and_kv_cache.py  how memory knobs become KV cache blocks
    ├── 03_batching_throughput.py  continuous batching under concurrency
    ├── 04_prefix_caching.py       APC — measure the TTFT win yourself
    ├── 05_structured_outputs.py   constrained decoding (JSON/regex/choice)
    ├── 06_speculative_decoding.py n-gram spec decode, acceptance rate
    ├── 07_quantization.py         FP8/NVFP4 on Blackwell, memory math
    ├── 08_online_serving.{txt,py} server + benchmarks — notes, and a runnable
    │                              harness that starts, exercises and tears down
    ├── 09_ebpf_observability.{txt,py}  eBPF notes, and a runnable probe runner
    ├── 10_chaos_and_limits.{txt,py}    failure notes, and 5 runnable scenarios
    ├── 11_metrics.py              Prometheus metrics that matter on-call
    ├── 12_multiturn_chat.py       multi-turn, OFFLINE and ONLINE (+ --interactive)
    ├── 13_kv_offload.py           KV offload to CPU: forces eviction, proves recovery
    └── _backend.py                shared dual-mode layer (--mode offline|online)
```

**Multi-turn works offline too.** See [MODES.md](MODES.md) — `LLM.chat()` handles
full conversations in-process. Online serving is what you need for *applications*
(persistent weights, concurrency, streaming), not for turns as such.

Every example is executable. 08, 09 and 10 come in two halves: a `.txt` with the
full explanation and every command spelled out for running by hand, and a `.py`
that actually runs it (starting and tearing down its own server where needed).

## Running

```bash
source /home/leeksun/testingEnvironment/vLLM_testing/vllm/.venv/bin/activate
cd /home/leeksun/testingEnvironment/vLLM_testing/vllm/SKhynix

python examples/00_env_check.py        # always start here
python examples/01_hello_offline.py
```

Every example takes `VLLM_SK_MODEL` to override the model and prints the knobs it
used, so you can diff behavior across runs:

```bash
VLLM_SK_MODEL=Qwen/Qwen2.5-1.5B-Instruct python examples/03_batching_throughput.py
```

## A note on the k8s deployments on this box

This node is a **single-node kubeadm cluster** and it is its own control plane.
`sudo KUBECONFIG=/etc/kubernetes/admin.conf kubectl ...` is cluster-admin. The
`sglang-phi` and `vllm-qwen` deployments in the `llm-inference` namespace were
holding ~40 GiB of the shared pool. If they come back and you need the memory:

```bash
sudo KUBECONFIG=/etc/kubernetes/admin.conf \
  kubectl scale deployment vllm-qwen sglang-phi -n llm-inference --replicas=0
```

Stopping containers with `crictl` does not work — kubelet restarts them. Desired
state lives in the API server, so scale/delete the Deployment instead.
