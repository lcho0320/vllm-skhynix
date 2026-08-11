# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

# Decode is memory-bandwidth bound: each step reads the entire model to produce
# one token. Speculative decoding proposes k tokens cheaply, then verifies all k
# in a single forward pass. Accepted proposals are k tokens for the price of one
# step.
#
# Methods in this build (see vllm/config/speculative.py):
#   ngram        no draft model; proposes by matching n-grams already in the
#                context. Costs no extra memory. Wins on grounded/repetitive
#                text (RAG, summarization, code), useless on open-ended prose.
#   eagle/eagle3 trained draft head, higher acceptance, needs a checkpoint
#   medusa       multiple decode heads
#   draft model  a small model of the same family
#
# The number that decides whether it pays is ACCEPTANCE RATE. Rejected
# proposals cost a wasted verification, so low acceptance is a net loss.

SPEC_TOKENS = 4

# Deliberately grounded: the answers largely quote the document, which is the
# shape n-gram proposals hit. Swap in open-ended prompts and the win vanishes.
document = (
    "Incident report INC-4471. At 02:14 UTC, node spark-fc72 reported GPU memory "
    "exhaustion. The vLLM engine core failed to start because free memory on device "
    "cuda:0 was 56.65 GiB while the requested gpu_memory_utilization of 0.92 required "
    "110.12 GiB. Two workloads were resident: sglang-phi holding 13.6 GiB and "
    "vllm-qwen holding 26.8 GiB. Mitigation was to scale both deployments to zero "
    "replicas. Root cause was an abandoned deployment left running for 37 days."
)

prompts = [
    f"{document}\n\nQuestion: Which node reported the failure, and what were the two "
    f"memory figures involved?\nAnswer:",
    f"{document}\n\nQuestion: What was the mitigation and the root cause?\nAnswer:",
    f"{document}\n\nRepeat the incident ID and the exact utilization figure:",
]

sampling_params = SamplingParams(temperature=0.0, max_tokens=96, ignore_eos=True)


def run(use_speculation: bool) -> float:
    print("-" * 60)
    print(f"speculative_decoding={use_speculation}")

    kwargs = dict(
        model="facebook/opt-125m",
        gpu_memory_utilization=0.30,
        max_model_len=2048,
        seed=0,
    )
    if use_speculation:
        # Top-level aliases for speculative_config, available in this build.
        # ngram needs no draft model, so KV cache is unaffected; draft-model
        # methods load a second model and take budget away from the cache.
        kwargs.update(spec_method="ngram", spec_tokens=SPEC_TOKENS)

    llm = LLM(**kwargs)

    # Warm up so graph capture and first-call costs are out of the timing.
    llm.generate(prompts[:1], sampling_params)

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start

    tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Generated tokens: {tokens}")
    print(f"Output throughput: {tokens / elapsed:.0f} tok/s")

    del llm
    cleanup_dist_env_and_memory()
    return elapsed


def main():
    baseline = run(use_speculation=False)
    speculative = run(use_speculation=True)

    print("-" * 60)
    print(f"Baseline:    {baseline:.2f}s")
    print(f"Speculative: {speculative:.2f}s")
    print(f"Speedup: {baseline / speculative:.2f}x")
    print("-" * 60)

    # Speculative decoding trades COMPUTE for LATENCY. On a saturated server it
    # can lower total throughput while improving per-request latency, so decide
    # which you are optimizing before enabling it.
    #
    # Acceptance is workload-dependent: a benchmark on someone else's prompts
    # tells you nothing about yours. Implementation: vllm/v1/spec_decode/.


if __name__ == "__main__":
    main()
