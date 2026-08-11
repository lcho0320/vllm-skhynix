# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

# vLLM does not assemble a fixed batch and wait. Every scheduler step it
# re-decides which requests run, admitting new ones as others finish. That is
# continuous (in-flight) batching, and it is the main reason vLLM beats naive
# per-request serving.
#
# Two knobs shape it:
#   max_num_seqs            ceiling on sequences resident in a step
#   max_num_batched_tokens  ceiling on tokens processed in a step
#
# Raising max_num_seqs raises throughput until KV cache runs out. Past that,
# requests are PREEMPTED (evicted and recomputed later), throughput falls off a
# cliff and p99 latency explodes. Finding that cliff is the point of capacity
# testing.

# (max_num_seqs, max_num_batched_tokens)
CONFIGS = [
    (8, 2048),
    (64, 4096),
    (256, 8192),
]

NUM_PROMPTS = 256
OUTPUT_TOKENS = 64

base_prompts = [
    "Summarize the operational risks of running LLM inference at rack scale:",
    "In a data center, GPU utilization is best measured by",
    "Describe how paged attention reduces memory fragmentation:",
    "The difference between prefill and decode phases is",
]

# Non-uniform prompts so the scheduler has uneven work, closer to real traffic.
prompts = [f"[req {i:04d}] {base_prompts[i % len(base_prompts)]}" for i in range(NUM_PROMPTS)]

# ignore_eos forces every request to emit exactly max_tokens. Without it you are
# measuring how early the model chose to stop, not the scheduler.
sampling_params = SamplingParams(temperature=0.0, max_tokens=OUTPUT_TOKENS, ignore_eos=True)


def run(max_num_seqs: int, max_num_batched_tokens: int) -> dict:
    print("-" * 60)
    print(f"max_num_seqs={max_num_seqs} max_num_batched_tokens={max_num_batched_tokens}")

    llm = LLM(
        model="facebook/opt-125m",
        gpu_memory_utilization=0.30,
        max_model_len=1024,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        seed=0,
    )

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start

    generated = sum(len(output.outputs[0].token_ids) for output in outputs)
    prompt_tokens = sum(len(output.prompt_token_ids) for output in outputs)

    print(f"Wall time: {elapsed:.2f}s")
    print(f"Prompt tokens: {prompt_tokens:,}  Generated tokens: {generated:,}")
    print(f"Output throughput: {generated / elapsed:.0f} tok/s")
    print(f"Request throughput: {len(prompts) / elapsed:.1f} req/s")

    del llm
    cleanup_dist_env_and_memory()
    return {
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "elapsed": elapsed,
        "output_tps": generated / elapsed,
        "requests_per_s": len(prompts) / elapsed,
    }


def main():
    results = [run(seqs, tokens) for seqs, tokens in CONFIGS]

    print("-" * 60)
    print(f"{'max_seqs':>9} {'max_tok':>9} {'wall(s)':>9} {'out tok/s':>11} {'req/s':>8}")
    for result in results:
        print(
            f"{result['max_num_seqs']:>9} {result['max_num_batched_tokens']:>9} "
            f"{result['elapsed']:>9.2f} {result['output_tps']:>11.0f} "
            f"{result['requests_per_s']:>8.1f}"
        )

    best = max(results, key=lambda r: r["output_tps"])
    worst = min(results, key=lambda r: r["output_tps"])
    print(f"Best/worst ratio: {best['output_tps'] / worst['output_tps']:.1f}x")
    print("-" * 60)

    # Watch the engine logs above for "Preempted": that is KV cache exhaustion,
    # and it is the throughput cliff described at the top of this file.
    #
    # This is a CLOSED system: all 256 requests are handed over at once. Real
    # serving is OPEN, with arrivals independent of your ability to keep up.
    # For TTFT/ITL percentiles under an arrival rate use `vllm bench serve`
    # (see 08_online_serving.sh).


if __name__ == "__main__":
    main()
