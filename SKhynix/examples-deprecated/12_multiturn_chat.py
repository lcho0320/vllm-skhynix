# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import sys

from _backend import add_mode_args, get_backend

# Multi-turn does NOT require online serving. LLM.chat() does full multi-turn
# offline, in this process. The two axes are independent:
#
#   offline vs online     = where the engine lives (in-process vs HTTP server)
#   single vs multi-turn  = how many messages are in the list
#
# What IS true: vLLM is STATELESS in both modes. It stores no conversation. Every
# turn the client resends the entire message list and the model re-reads it.
# There is no session id and no server-side memory.
#
# Three consequences to design around:
#   1. Cost grows quadratically. Turn N resends turns 1..N-1.
#   2. Prefix caching is what makes that affordable: each turn's prompt is the
#      previous turn's prompt plus more, so it is a near-perfect cache hit.
#   3. You must trim history eventually. Context is finite and vLLM will not
#      truncate for you.
#
# Run with an INSTRUCT model; base models have no chat template:
#   VLLM_SK_MODEL=Qwen/Qwen2.5-0.5B-Instruct python 12_multiturn_chat.py

system_prompt = (
    "You are a concise data-center infrastructure assistant. "
    "Answer in two sentences or fewer."
)

turns = [
    "I run a GPU cluster with 8 nodes. What should I monitor first?",
    "Which of those would catch an abandoned deployment wasting memory?",
    "How would I alert on that specifically?",
]


def scripted(backend):
    # This list is the entire conversation state. Nothing else persists.
    messages = [{"role": "system", "content": system_prompt}]

    for i, turn in enumerate(turns, start=1):
        messages.append({"role": "user", "content": turn})
        print("-" * 60)
        print(f"Turn {i}, sending {len(messages)} messages")
        print(f"User: {turn}")

        reply = backend.chat(messages, max_tokens=96)
        print(f"Assistant: {reply.strip()}")

        # Measured, not assumed: how much of this turn's prompt came from cache.
        total = backend.last_prompt_tokens
        cached = backend.last_cached_tokens
        share = 100 * cached / total if total else 0.0
        print(f"Prompt tokens: {total}, from cache: {cached} ({share:.0f}%)")

        # Appending the reply is what makes the NEXT turn aware of this one.
        # Omitting it is the most common bug in hand-rolled chat loops.
        messages.append({"role": "assistant", "content": reply})

    print("-" * 60)
    print(f"Final history: {len(messages)} messages")
    # The cache share climbs every turn as the shared history grows. That is
    # prefix caching turning a quadratic re-send into near-linear real compute.


def interactive(backend, args):
    messages = [{"role": "system", "content": system_prompt}]
    print("Interactive. 'exit' to quit, 'reset' to clear history.")

    while True:
        try:
            user_input = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        if user_input.lower() == "reset":
            messages = [{"role": "system", "content": system_prompt}]
            print("History cleared")
            continue

        messages.append({"role": "user", "content": user_input})

        if args.mode == "online" and hasattr(backend, "chat_stream"):
            print("bot > ", end="", flush=True)
            chunks = []
            for delta in backend.chat_stream(messages, max_tokens=200):
                print(delta, end="", flush=True)
                chunks.append(delta)
            reply = "".join(chunks)
            print()
        else:
            reply = backend.chat(messages, max_tokens=200)
            print(f"bot > {reply.strip()}")

        messages.append({"role": "assistant", "content": reply})


def main():
    parser = argparse.ArgumentParser()
    add_mode_args(parser)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    print(f"Mode: {args.mode}, model: {args.model}")
    backend = get_backend(args)
    try:
        if args.interactive:
            interactive(backend, args)
        else:
            scripted(backend)
    except Exception as exc:
        if "chat template" in str(exc).lower():
            print(f"\n{exc}\n")
            print("Base model with no chat template. Use an instruct model:")
            print("  VLLM_SK_MODEL=Qwen/Qwen2.5-0.5B-Instruct \\")
            print(f"    python 12_multiturn_chat.py --mode {args.mode}")
            sys.exit(1)
        raise
    finally:
        backend.close()


if __name__ == "__main__":
    main()
