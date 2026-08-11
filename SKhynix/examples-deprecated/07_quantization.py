# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

# Starts no engine. This is the arithmetic that decides what you can run at all.
#
# Quantization is a CAPACITY decision, not a tuning knob. On this box the
# 119.7 GiB unified pool is the binding constraint, and the format you pick
# determines whether a model fits before any performance question arises.
#
# GB10 is sm_121 (Blackwell), so FP8 and NVFP4 kernels are both available.

# (name, billions of parameters)
MODELS = [
    ("Llama-3.2-1B", 1.24),
    ("Qwen2.5-7B", 7.6),
    ("Llama-3.1-8B", 8.03),
    ("Qwen2.5-32B", 32.8),
    ("Llama-3.3-70B", 70.6),
    ("Llama-3.1-405B", 405.0),
]

# (label, bytes per parameter)
FORMATS = [
    ("BF16", 2.0),
    ("FP8", 1.0),
    ("INT4/NVFP4", 0.5),
]

# Weights are only part of the budget; KV cache and activations come on top.
# Treating 60% of free memory as the weight ceiling leaves room for them.
WEIGHT_BUDGET_FRACTION = 0.6


def weight_gib(params_billions: float, bytes_per_param: float) -> float:
    return params_billions * 1e9 * bytes_per_param / 2**30


def main():
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gib = free_bytes / 2**30
    total_gib = total_bytes / 2**30

    print("-" * 72)
    print(f"Unified pool: {total_gib:.1f} GiB total, {free_gib:.1f} GiB free")
    print(f"Weight ceiling used below: {WEIGHT_BUDGET_FRACTION:.0%} of free memory")
    print("-" * 72)

    header = f"{'model':<18}" + "".join(f"{label:>14}" for label, _ in FORMATS)
    print(header)
    for name, params in MODELS:
        row = f"{name:<18}"
        for _, bytes_per_param in FORMATS:
            size = weight_gib(params, bytes_per_param)
            fits = size < free_gib * WEIGHT_BUDGET_FRACTION
            row += f"{size:>12.1f}{' ' if fits else '*'} "
        print(row)
    print("* = weights alone exceed the ceiling; not practically servable here")

    # The 70B in the local HF cache is a useful worked example: it is 263 GB of
    # BF16 weights on disk, which cannot fit this pool in any configuration.
    print("-" * 72)
    print("meta-llama/Llama-3.1-70B (cached locally, 263 GB of BF16 shards):")
    for label, bytes_per_param in FORMATS:
        size = weight_gib(70.6, bytes_per_param)
        print(
            f"  {label:<12} needs {size:>6.0f} GiB; "
            f"{total_gib - size:>6.0f} GiB left of the pool, "
            f"{free_gib - size:>+6.0f} GiB of what is free now"
        )
    print("  BF16 is impossible here at any time. FP8 fits the pool but not")
    print("  today's free memory, and the remainder still has to cover KV cache,")
    print("  activations and the OS. INT4 is the only comfortable option.")

    print("-" * 72)
    print("Loading a quantized checkpoint (usually auto-detected from config.json):")
    print('  llm = LLM(model="RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic")')
    print('  llm = LLM(model=..., quantization="fp8")   # force a method')
    print()
    # Weights are a fixed cost. KV cache scales with concurrency x context, so on
    # long-context serving it frequently exceeds the weights. Quantizing it
    # roughly doubles the sequences you can hold; re-run 02 to see the blocks.
    print("KV cache quantization is the other half of the budget:")
    print('  llm = LLM(model=..., kv_cache_dtype="fp8")')
    print("-" * 72)


if __name__ == "__main__":
    main()
