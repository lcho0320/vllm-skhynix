# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

# When many requests share a leading prefix — a system prompt, a few-shot block,
# a retrieved document — vLLM hashes the KV blocks for that prefix and reuses
# them instead of recomputing. Nothing moves: the blocks are simply still
# resident in GPU memory. The saving is entirely in PREFILL; decode is untouched.
#
# This is usually the highest-leverage optimization in production, because real
# traffic is full of shared prefixes and the win scales with prefix length.
# Automatic prefix caching (APC) is on by default in V1, so this example turns
# it OFF to measure the difference.

# Stand-in for a long system prompt or retrieved document. Production prompts
# commonly carry 1k-10k tokens of this on every single request.
shared_prefix = (
    "You are an infrastructure assistant for a large AI data center. "
    "You have access to telemetry from thousands of GPU nodes, including "
    "utilization, memory pressure, thermal readings, NVLink bandwidth, power "
    "draw per rack, and job scheduling history. When answering, you consider "
    "capacity planning, failure domains, cost per token, and SLA commitments. "
    "You are precise, quantitative, and you flag uncertainty explicitly. "
) * 8

questions = [
    "What is the first metric to check for a latency regression?",
    "How should we size the KV cache for a 70B model?",
    "Which failure mode causes throughput cliffs under load?",
    "When is disaggregated prefill worth the complexity?",
]

NUM_QUERIES = 32

prompts = [
    f"{shared_prefix}\n\nQuestion: {questions[i % len(questions)]}\nAnswer:"
    for i in range(NUM_QUERIES)
]

sampling_params = SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True)


def run(enable_prefix_caching: bool) -> float:
    print("-" * 60)
    print(f"enable_prefix_caching={enable_prefix_caching}")

    # opt-125m caps at 2048 positions; max_model_len cannot exceed a model's own
    # max_position_embeddings or the engine refuses to start.
    llm = LLM(
        model="facebook/opt-125m",
        gpu_memory_utilization=0.30,
        max_model_len=2048,
        enable_prefix_caching=enable_prefix_caching,
        seed=0,
    )

    # Warm-up populates the cache with the shared prefix. With caching off this
    # changes nothing, which is why both arms run it.
    llm.generate(prompts[:2], sampling_params)

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start

    prompt_tokens = sum(len(output.prompt_token_ids) for output in outputs)

    # num_cached_tokens is the direct evidence: prompt tokens served from cache
    # rather than computed. It is local + external, but with no KV connector
    # configured it is purely the GPU prefix cache.
    cached_tokens = sum(output.num_cached_tokens or 0 for output in outputs)

    print(f"Prompt tokens offered: {prompt_tokens:,}")
    print(f"Prompt tokens served from cache: {cached_tokens:,}")
    print(f"Wall time: {elapsed:.2f}s")

    del llm
    cleanup_dist_env_and_memory()
    return elapsed


def main():
    without_caching = run(enable_prefix_caching=False)
    with_caching = run(enable_prefix_caching=True)

    print("-" * 60)
    print(f"Without prefix caching: {without_caching:.2f}s")
    print(f"With prefix caching:    {with_caching:.2f}s")
    print(f"Speedup: {without_caching / with_caching:.2f}x")
    print("-" * 60)

    # Operational notes:
    #  - Cached blocks OCCUPY KV cache and are evicted LRU, so hit rate degrades
    #    exactly when you are busiest.
    #  - The cache is per-engine and in-memory: a pod restart cold-starts it,
    #    which matters for rollout planning.
    #  - Sharing prefixes ACROSS nodes needs a KV connector (LMCache, Mooncake,
    #    NIXL). Extending it to CPU/disk on one node is 13_kv_offload.py.


if __name__ == "__main__":
    main()
