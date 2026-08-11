# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

from vllm import LLM, SamplingParams

# Every offline vLLM program is these three steps. Everything else is an
# elaboration on one of them.
#   1. build the engine   (expensive: weights, profiling, KV sizing, graphs)
#   2. describe sampling  (cheap, per-request)
#   3. run a batch        (pass a LIST; vLLM schedules it concurrently)

prompts = [
    "The role of a data center in AI infrastructure is",
    "Explain GPU memory fragmentation in one sentence:",
    "A rack-scale inference cluster needs",
]

# temperature=0 makes decode deterministic, so reruns are comparable.
greedy_params = SamplingParams(temperature=0.0, max_tokens=48)

# n=2 draws two samples from one prompt. They share the prompt's KV cache, so
# prefill happens once and only decode is duplicated.
sampling_params = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=48, n=2)


def main():
    # gpu_memory_utilization is a fraction of the WHOLE unified pool on GB10,
    # which the OS also lives in. 0.30 is conservative; run 00_env_check.py for
    # the current ceiling. enforce_eager=False keeps CUDA graph capture on:
    # slower to start, faster per decode step.
    start = time.perf_counter()
    llm = LLM(
        model="facebook/opt-125m",
        gpu_memory_utilization=0.30,
        max_model_len=2048,
        enforce_eager=False,
        seed=0,
    )
    print(f"\nEngine ready in {time.perf_counter() - start:.1f}s")

    # Passing the whole list at once is what gets you continuous batching.
    # Looping one prompt per call serializes everything and wastes the engine.
    start = time.perf_counter()
    outputs = llm.generate(prompts, greedy_params)
    elapsed = time.perf_counter() - start

    print("\nGreedy outputs:\n" + "-" * 60)
    generated_tokens = 0
    for output in outputs:
        completion = output.outputs[0]
        generated_tokens += len(completion.token_ids)
        print(f"Prompt: {output.prompt!r}")
        print(f"Output: {completion.text.strip()!r}")
        # finish_reason "length" means max_tokens truncated the answer;
        # "stop" means the model chose to end. Worth watching in production.
        print(f"Tokens: {len(completion.token_ids)}  Finish: {completion.finish_reason}")
        print("-" * 60)

    # A tiny model on a tiny batch is latency-bound, so this number says nothing
    # about throughput. Example 03 measures that properly.
    print(f"{generated_tokens} tokens in {elapsed:.2f}s = {generated_tokens / elapsed:.0f} tok/s")

    outputs = llm.generate(prompts[:1], sampling_params)

    print("\nSampled outputs (n=2, one shared prefill):\n" + "-" * 60)
    for output in outputs:
        for i, completion in enumerate(output.outputs):
            print(f"Sample {i}: {completion.text.strip()!r}")
            print("-" * 60)


if __name__ == "__main__":
    main()
