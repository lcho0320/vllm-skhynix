# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

# A tiny GPU KV cache, pinned exactly so eviction is guaranteed rather than hoped for.
# 48 MiB holds roughly 4000 tokens for this model, so a few long prompts overflow it.
KV_CACHE_BYTES = 48 * 1024 * 1024

# CPU tier for offloaded blocks. Must exceed the GPU KV cache or it only mirrors it.
CPU_BYTES_TO_USE = 4_000_000_000

OFFLOADING_CONFIG = {
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "block_size": 64,
        "cpu_bytes_to_use": CPU_BYTES_TO_USE,
    },
}

# The prompt whose KV cache we want to survive eviction.
target_prompt = "SYSTEM CONTEXT. " + ("data center telemetry node monitoring " * 180)

# Enough unrelated traffic to push the target out of the GPU cache.
filler_prompts = [
    f"filler request {i} " + ("padding tokens here " * 180) for i in range(40)
]

sampling_params = SamplingParams(temperature=0.0, max_tokens=4)


def prefix_cache_stats(llm: LLM) -> dict[str, float]:
    # Requires disable_log_stats=False; offline LLM disables stats by default.
    # Note: external_prefix_cache_hits reads 0 here even when blocks are served
    # from the CPU tier. The per-request num_cached_tokens is the reliable signal.
    stats = {}
    for metric in llm.get_metrics():
        name = getattr(metric, "name", "")
        if "prefix_cache" in name:
            stats[name] = getattr(metric, "value", None)
    return stats


def run(use_offloading: bool) -> int:
    kwargs = dict(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        gpu_memory_utilization=0.30,
        kv_cache_memory_bytes=KV_CACHE_BYTES,
        max_model_len=4096,
        enforce_eager=True,
        disable_log_stats=False,
        seed=0,
    )
    if use_offloading:
        kwargs["kv_transfer_config"] = OFFLOADING_CONFIG

    llm = LLM(**kwargs)

    label = "with offloading" if use_offloading else "without offloading"
    print("-" * 60)
    print(f"Run {label}")
    print(f"GPU KV blocks: {llm.llm_engine.vllm_config.cache_config.num_gpu_blocks}")

    # Populate the cache with the target prompt.
    llm.generate([target_prompt], sampling_params)

    # Flood the cache so the target's blocks are evicted from GPU memory.
    llm.generate(filler_prompts, sampling_params)

    # Ask for the target again. Anything cached now had to survive eviction.
    output = llm.generate([target_prompt], sampling_params)[0]

    prompt_tokens = len(output.prompt_token_ids)
    cached_tokens = output.num_cached_tokens or 0
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Cached tokens after eviction: {cached_tokens}")
    print(f"Recomputed tokens: {prompt_tokens - cached_tokens}")
    for name, value in prefix_cache_stats(llm).items():
        print(f"  {name}: {value}")

    del llm
    cleanup_dist_env_and_memory()
    return cached_tokens


def main():
    # Baseline: evicted blocks are gone and must be recomputed.
    baseline_cached = run(use_offloading=False)

    # With offloading: evicted blocks were copied to host memory and come back.
    offloaded_cached = run(use_offloading=True)

    print("-" * 60)
    print(f"Cached tokens without offloading: {baseline_cached}")
    print(f"Cached tokens with offloading:    {offloaded_cached}")
    recovered = offloaded_cached - baseline_cached
    print(f"Tokens recovered from the CPU tier: {recovered}")
    print("-" * 60)


if __name__ == "__main__":
    main()
