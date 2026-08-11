# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

# Starts an OpenAI-compatible server, exercises it, benchmarks it, and shuts it
# down again. The narrative version of all this is 08_online_serving.txt.
#
# Offline generate() measures a CLOSED system: you hand over a fixed batch and
# time it. Production is OPEN — requests arrive whether or not you can keep up —
# so the metrics that matter are per-request percentiles, not aggregate tok/s:
#
#   TTFT  time to first token   dominated by prefill + queueing
#   ITL   inter-token latency   dominated by decode step time + batch size
#   E2E   end-to-end latency
#
# The server is launched with start_new_session=True so it gets its own process
# group. That means we can signal the whole group on teardown without killing
# this script, and a stray Ctrl-C does not orphan a GPU-holding process.

DEFAULT_MODEL = os.environ.get("VLLM_SK_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
STARTUP_TIMEOUT = 600


def wait_for_health(port: int, process: subprocess.Popen, timeout: int) -> bool:
    # Poll /health until the engine is up. Also watch the child: if it exits
    # early there is no point waiting out the full timeout.
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            print(f"Server exited early with code {process.returncode}")
            return False
        try:
            with urllib.request.urlopen(url, timeout=2):
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    print(f"Server did not become healthy within {timeout}s")
    return False


def post_json(url: str, payload: dict, timeout: int = 60) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def show_endpoints(port: int):
    print("-" * 60)
    print("Endpoints")

    # /v1/models doubles as a readiness probe and tells you the served name,
    # which need not equal the checkpoint path (see --served-model-name).
    with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=10) as response:
        models = json.loads(response.read().decode())
    served = [entry["id"] for entry in models.get("data", [])]
    print(f"/v1/models -> {served}")

    # /health is what a k8s liveness probe hits. Note that both this and
    # /v1/models look identical to real traffic in a request-count graph, which
    # is how an idle deployment hides for weeks. Exclude them from dashboards.
    with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=10) as response:
        print(f"/health -> {response.status}")

    return served[0] if served else DEFAULT_MODEL


def run_requests(port: int, model: str):
    print("-" * 60)
    print("Completion request")
    result = post_json(
        f"http://localhost:{port}/v1/completions",
        {"model": model, "prompt": "A data center is", "max_tokens": 32},
    )
    print(f"  {result['choices'][0]['text'].strip()!r}")
    print(f"  usage: {result.get('usage')}")

    print("Chat request")
    result = post_json(
        f"http://localhost:{port}/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Name one GPU metric to alert on."}],
            "max_tokens": 32,
        },
    )
    print(f"  {result['choices'][0]['message']['content'].strip()!r}")

    print("Streaming request (deltas arrive as they are produced)")
    request = urllib.request.Request(
        f"http://localhost:{port}/v1/completions",
        data=json.dumps(
            {"model": model, "prompt": "Explain paged attention:", "max_tokens": 48, "stream": True}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    first_token_at = None
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=60) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            if first_token_at is None:
                # This gap is TTFT: prefill plus any time spent queued.
                first_token_at = time.perf_counter() - start
            chunk = json.loads(line[6:])
            sys.stdout.write(chunk["choices"][0]["text"])
            sys.stdout.flush()
    print(f"\n  TTFT: {first_token_at:.3f}s")


def show_metrics(port: int):
    print("-" * 60)
    print("Metrics worth watching")
    with urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=10) as response:
        text = response.read().decode()
    wanted = (
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
        "vllm:num_preemptions_total",
        "vllm:generation_tokens_total",
        "vllm:prefix_cache_hits_total",
    )
    for line in text.splitlines():
        if line.startswith(wanted):
            print(f"  {line}")


def run_benchmark(port: int, model: str, request_rate: str, num_prompts: int):
    # The important measurement. Unlike offline timing, this generates Poisson
    # arrivals at a fixed rate and reports TTFT/ITL/E2E percentiles. Sweep
    # request_rate upward until p99 TTFT breaks your SLA: that knee is the
    # per-replica capacity you divide expected traffic by.
    print("-" * 60)
    print(f"vllm bench serve at request-rate={request_rate}")
    command = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
        "--backend", "vllm",
        "--model", model,
        "--host", "localhost",
        "--port", str(port),
        "--dataset-name", "random",
        "--random-input-len", "256",
        "--random-output-len", "64",
        "--num-prompts", str(num_prompts),
        "--request-rate", request_rate,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if any(key in line for key in ("Throughput", "TTFT", "TPOT", "ITL", "latency", "===")):
            print(f"  {line}")
    if result.returncode != 0:
        print(f"  benchmark exited {result.returncode}")
        print(result.stderr[-500:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--request-rate", default="5")
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--skip-bench", action="store_true")
    args = parser.parse_args()

    # Invoke through the module rather than the `vllm` console script, so this
    # works whether or not the venv is activated on PATH.
    command = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", args.model,
        "--port", str(args.port),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
        "--enable-prompt-tokens-details",
    ]
    print("-" * 60)
    print(" ".join(command))

    # start_new_session puts the server in its own process group, so teardown
    # below can signal the group without touching this script.
    start = time.perf_counter()
    server = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    try:
        if not wait_for_health(args.port, server, STARTUP_TIMEOUT):
            return 1
        # Cold-start time bounds how fast you can scale out: a replica is
        # useless until weights are loaded and graphs captured.
        print(f"Server healthy after {time.perf_counter() - start:.1f}s")

        model = show_endpoints(args.port)
        run_requests(args.port, model)
        if not args.skip_bench:
            run_benchmark(args.port, model, args.request_rate, args.num_prompts)
        show_metrics(args.port)
        print("-" * 60)
    finally:
        # SIGTERM, not SIGKILL: vLLM installs handlers for SIGTERM/SIGINT and
        # unwinds the engine cleanly. SIGKILL skips that and can leave shared
        # memory segments and the EngineCore child behind.
        print("Stopping server")
        try:
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(server.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        print("Stopped")

    return 0


if __name__ == "__main__":
    sys.exit(main())
