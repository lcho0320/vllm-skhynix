# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import time

from _backend import add_mode_args, get_backend

# vLLM can force output to match a JSON schema, regex or fixed choice set by
# masking invalid tokens at each decode step. The model CANNOT emit malformed
# output, because illegal tokens are removed from the distribution before
# sampling.
#
# This guarantees STRUCTURE, never CORRECTNESS. It removes parse failures from
# your error budget; it does not remove hallucinations. Demo 2 below makes that
# concrete: a small model gets the classification wrong every time, but the
# answer is always one of the three permitted strings.
#
# API note (offline): this build uses structured_outputs=StructuredOutputsParams.
# Older vLLM used guided_decoding=GuidedDecodingParams, so copied internet
# examples raise TypeError.
# API note (online): vLLM-specific fields ride in extra_body; the server also
# accepts standard OpenAI response_format.

node_schema = {
    "type": "object",
    "properties": {
        "node_id": {"type": "string"},
        "healthy": {"type": "boolean"},
        "gpu_util_pct": {"type": "integer", "minimum": 0, "maximum": 100},
        "action": {"type": "string", "enum": ["none", "drain", "restart", "page_oncall"]},
    },
    "required": ["node_id", "healthy", "gpu_util_pct", "action"],
    "additionalProperties": False,
}

incidents = [
    "GPU memory exhausted, engine core crashed, requests failing.",
    "Latency p99 up 15% over baseline but within SLA.",
    "All nodes reporting nominal.",
]


def demo_json_schema(backend):
    print("-" * 60)
    print("JSON schema: output is guaranteed parseable and schema-valid")

    prompt = (
        "Report the status of node spark-fc72, which is at 94% GPU utilization "
        "and thermally throttling. Respond as JSON."
    )
    # The first constrained request pays a one-time grammar compilation cost.
    start = time.perf_counter()
    text = backend.chat(
        [{"role": "user", "content": prompt}],
        json_schema=json.dumps(node_schema),
        max_tokens=128,
    )
    print(f"Elapsed (includes grammar compilation): {time.perf_counter() - start:.2f}s")
    print(f"Raw: {text!r}")
    try:
        print(f"Parsed: {json.loads(text)}")
    except json.JSONDecodeError as exc:
        # Small models can still hit max_tokens mid-object. The grammar keeps
        # every emitted token legal; it cannot make the model finish in time.
        print(f"Parse failed: {exc}")


def demo_choice(backend):
    print("-" * 60)
    print("Choice: output is exactly one of the listed strings")

    for incident in incidents:
        text = backend.chat(
            [{"role": "user", "content": f"Classify this incident: {incident}"}],
            choice=["critical", "warning", "healthy"],
            max_tokens=16,
        )
        print(f"  {incident[:48]:<48} -> {text.strip()!r}")

    # Expect these to be semantically wrong with a small model, and structurally
    # perfect every time. That contrast is the lesson.


def demo_regex(backend):
    print("-" * 60)
    print("Regex: force a rigid format")

    text = backend.chat(
        [{"role": "user", "content": "What is the control plane node's IP address?"}],
        regex=r"(25[0-5]|2[0-4]\d|[01]?\d?\d)(\.(25[0-5]|2[0-4]\d|[01]?\d?\d)){3}",
        max_tokens=24,
    )
    print(f"  {text.strip()!r}")


def demo_overhead(backend):
    print("-" * 60)
    print("Overhead: constrained vs unconstrained")

    messages = [{"role": "user", "content": "Report status of node n01 as JSON."}]

    start = time.perf_counter()
    for _ in range(8):
        backend.chat(messages, max_tokens=64)
    unconstrained = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(8):
        backend.chat(messages, json_schema=json.dumps(node_schema), max_tokens=64)
    constrained = time.perf_counter() - start

    print(f"  Unconstrained: {unconstrained:.2f}s")
    print(f"  Constrained:   {constrained:.2f}s ({constrained / unconstrained:.2f}x)")

    # Read that ratio skeptically. Constrained runs often finish FASTER because
    # the grammar forces a closing brace and the request stops, while the
    # unconstrained run rambles to max_tokens. Different output lengths means
    # this is not an apples-to-apples measurement. To isolate the per-step mask
    # cost, hold output length constant with ignore_eos and warm the grammar
    # cache first.


def main():
    parser = argparse.ArgumentParser()
    add_mode_args(parser)
    args = parser.parse_args()

    print(f"Mode: {args.mode}, model: {args.model}")
    backend = get_backend(args)
    try:
        demo_json_schema(backend)
        demo_choice(backend)
        demo_regex(backend)
        demo_overhead(backend)
        print("-" * 60)
    except Exception as exc:
        if "chat template" in str(exc).lower():
            print(f"\n{exc}\nUse an instruct model:")
            print("  VLLM_SK_MODEL=Qwen/Qwen2.5-0.5B-Instruct \\")
            print(f"    python 05_structured_outputs.py --mode {args.mode}")
            return
        raise
    finally:
        backend.close()

    # Backends live in vllm/v1/structured_output/: auto (default), xgrammar,
    # guidance, outlines, lm-format-enforcer.
    #   offline: LLM(..., structured_outputs_config={"backend": "xgrammar"})
    #   online:  vllm serve ... --structured-outputs-config '{"backend":"xgrammar"}'


if __name__ == "__main__":
    main()
