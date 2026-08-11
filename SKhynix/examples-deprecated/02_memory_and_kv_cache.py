# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory

# vLLM's startup memory budget, in order:
#
#   gpu_memory_utilization * TOTAL_POOL
#     - model weights
#     - peak activation      (measured by a real profiling forward pass)
#     - CUDA graph memory    (skipped when enforce_eager=True)
#     = KV CACHE
#
# The KV cache is the leftover, and it alone decides how many sequences you can
# serve concurrently. This is the core capacity equation.

# Kept low because the GB10 pool is shared with the OS. Do not exceed the
# ceiling printed by 00_env_check.py.
UTILIZATIONS = [0.15, 0.25, 0.35]

MODEL = "facebook/opt-125m"


def kv_bytes_per_token(llm: LLM) -> int:
    # Every token caches a key and a value, for each layer, for each KV head.
    # GQA/MQA models share KV heads across query heads, so num_kv_heads (not
    # num_attention_heads) is what counts here.
    config = llm.llm_engine.vllm_config
    text_config = config.model_config.hf_text_config
    num_layers = config.model_config.get_num_layers(config.parallel_config)
    num_kv_heads = config.model_config.get_num_kv_heads(config.parallel_config)
    head_dim = getattr(
        text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads
    )
    return 2 * num_layers * num_kv_heads * head_dim * config.model_config.dtype.itemsize


def probe(utilization: float):
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print("-" * 60)
    print(f"gpu_memory_utilization = {utilization}")
    print(f"Free before load: {free_bytes / 2**30:.2f} GiB of {total_bytes / 2**30:.2f} GiB")

    try:
        # enforce_eager=True keeps the probe fast and removes CUDA graph memory
        # from the budget, so the KV numbers below are an upper bound on what
        # you would actually get in production.
        llm = LLM(
            model=MODEL,
            gpu_memory_utilization=utilization,
            max_model_len=2048,
            enforce_eager=True,
        )
    except Exception as exc:
        # vLLM compares utilization * TOTAL against FREE, so a request above
        # free/total fails here rather than OOMing later. Triggering it on
        # purpose is worth doing once.
        print(f"Engine refused to start: {type(exc).__name__}: {str(exc)[:200]}")
        return

    config = llm.llm_engine.vllm_config
    cache_config = config.cache_config
    per_token = kv_bytes_per_token(llm)

    # num_gpu_blocks is decided at startup, after the profiling forward pass.
    # Each block holds block_size tokens. This is the number to report when
    # someone asks "how much can this replica serve".
    num_blocks = cache_config.num_gpu_blocks or 0
    kv_tokens = num_blocks * cache_config.block_size
    max_len = config.model_config.max_model_len

    print(f"KV bytes per token: {per_token:,}")
    print(f"Block size: {cache_config.block_size} tokens")
    print(f"GPU KV blocks: {num_blocks:,}")
    print(f"KV capacity: {kv_tokens:,} tokens ({kv_tokens * per_token / 2**30:.2f} GiB)")

    # Concurrency is KV capacity divided by average context length. Both halves
    # are tunable, and context length is usually the cheaper one to negotiate.
    print(f"Concurrent sequences at {max_len} tokens: {kv_tokens // max_len:,}")
    print(f"Concurrent sequences at 256 tokens: {kv_tokens // 256:,}")

    # Building several engines in one process leaks distributed state unless you
    # use this helper. It also releases host memory, which on GB10 is the same
    # pool the next engine needs.
    del llm
    cleanup_dist_env_and_memory()


def main():
    # The free number moves between runs because OS page cache shares the
    # unified pool. Re-run 00_env_check.py if a probe unexpectedly fails.
    for utilization in UTILIZATIONS:
        probe(utilization)
    print("-" * 60)


if __name__ == "__main__":
    main()
