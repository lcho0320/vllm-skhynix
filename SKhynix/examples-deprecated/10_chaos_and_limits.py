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

# Runs one failure scenario at a time and reports what vLLM actually did. The
# narrative version, with scenarios this harness does not automate, is
# 10_chaos_and_limits.txt.
#
# For data-center work these behaviours matter more than peak throughput,
# because they decide what happens at 3am. Each scenario manages its own server
# and tears it down again, so nothing is left holding GPU memory.

DEFAULT_MODEL = os.environ.get("VLLM_SK_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
SERVE_MODULE = ["-m", "vllm.entrypoints.cli.main", "serve"]


def start_server(model: str, port: int, extra: list[str], quiet: bool = True):
    # start_new_session gives the server its own process group so stop_server
    # can signal the whole tree without touching this script.
    command = [sys.executable, *SERVE_MODULE, model, "--port", str(port), *extra]
    print(f"  launching: {' '.join(command[2:])}")
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE if quiet else None,
        stderr=subprocess.STDOUT if quiet else None,
        start_new_session=True,
        text=True,
    )


def stop_server(process: subprocess.Popen):
    # SIGTERM, never SIGKILL first: vLLM handles SIGTERM and unwinds cleanly.
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def wait_for_health(port: int, process: subprocess.Popen, timeout: int = 600) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2):
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    return False


def get_metrics(port: int, names: tuple[str, ...]) -> dict[str, float]:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=10) as response:
            text = response.read().decode()
    except (urllib.error.URLError, OSError):
        return {}
    found = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for name in names:
            if line.startswith(name):
                try:
                    found[name] = found.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
                except (ValueError, IndexError):
                    pass
    return found


def scenario_startup_oom(args):
    # Ask for more memory than exists. Expected: a fast, loud, clean failure
    # from request_memory() in vllm/v1/worker/utils.py rather than an OOM
    # partway through serving. This is the failure you met on day one.
    print("Requesting gpu_memory_utilization=0.99 on a shared pool")
    start = time.perf_counter()
    server = start_server(args.model, args.port, ["--gpu-memory-utilization", "0.99"])
    try:
        healthy = wait_for_health(args.port, server, timeout=180)
        elapsed = time.perf_counter() - start
        if healthy:
            print(f"  UNEXPECTED: server started in {elapsed:.0f}s; the pool was free enough")
            return
        output = server.stdout.read() if server.stdout else ""
        print(f"  server exited after {elapsed:.0f}s with code {server.returncode}")
        for line in output.splitlines():
            if "Free memory on device" in line or "ValueError" in line:
                print(f"  {line.strip()[:160]}")
        # Fast and loud is the correct behaviour: in k8s this becomes
        # CrashLoopBackOff rather than a pod that half-serves.
        print("  Verdict: fails fast and loudly, before accepting any traffic")
    finally:
        stop_server(server)


def scenario_kv_exhaustion(args):
    # Force the scheduler past its KV cache with high concurrency, and watch for
    # preemption: requests evicted and recomputed. Expected: graceful
    # degradation (a throughput cliff), never a crash.
    print("Tiny KV cache plus high concurrency, then saturating load")
    server = start_server(
        args.model,
        args.port,
        [
            "--gpu-memory-utilization", "0.20",
            "--max-model-len", "2048",
            "--max-num-seqs", "512",
            "--kv-cache-memory-bytes", str(64 * 1024 * 1024),
        ],
    )
    try:
        if not wait_for_health(args.port, server):
            print("  server failed to start")
            return
        command = [
            sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
            "--backend", "vllm", "--model", args.model,
            "--host", "localhost", "--port", str(args.port),
            "--dataset-name", "random",
            "--random-input-len", "1024", "--random-output-len", "256",
            "--num-prompts", "120", "--request-rate", "inf",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if any(key in line for key in ("Throughput", "TTFT", "ITL", "===")):
                print(f"  {line}")
        metrics = get_metrics(args.port, ("vllm:num_preemptions_total", "vllm:kv_cache_usage_perc"))
        print(f"  metrics: {metrics}")
        preemptions = metrics.get("vllm:num_preemptions_total", 0.0)
        if preemptions > 0:
            print(f"  Verdict: {preemptions:.0f} preemptions. Cache undersized; alert on this rate.")
        else:
            print("  Verdict: no preemptions; cache absorbed the load. Lower the cache to force it.")
    finally:
        stop_server(server)


def scenario_engine_kill(args):
    # Kill the EngineCore child and measure how long the API server keeps
    # answering /health. That window is your worst-case blackhole, and what
    # liveness probe periodSeconds should be tuned against.
    print("Killing the EngineCore process out from under a healthy server")
    server = start_server(args.model, args.port, ["--gpu-memory-utilization", "0.30"])
    try:
        if not wait_for_health(args.port, server):
            print("  server failed to start")
            return
        print("  server healthy")
        found = subprocess.run(
            ["pgrep", "-f", "VLLM::EngineCore"], capture_output=True, text=True
        ).stdout.split()
        if not found:
            print("  no EngineCore process found")
            return
        pid = int(found[0])
        print(f"  killing EngineCore pid {pid}")
        os.kill(pid, signal.SIGKILL)

        start = time.perf_counter()
        for _ in range(60):
            try:
                with urllib.request.urlopen(f"http://localhost:{args.port}/health", timeout=2) as r:
                    if r.status != 200:
                        break
            except (urllib.error.URLError, OSError):
                break
            time.sleep(0.5)
        detection = time.perf_counter() - start
        print(f"  /health stopped answering 200 after {detection:.1f}s")
        print("  Verdict: set liveness probe periodSeconds below this to bound the blackhole")
    finally:
        stop_server(server)


def scenario_bad_requests(args):
    # Malformed and oversized requests must be clean 4xx, never OOM. Any of
    # these crashing the server is a denial-of-service vector in a shared
    # service.
    print("Oversized context, absurd max_tokens, and a mid-stream disconnect")
    server = start_server(
        args.model, args.port, ["--gpu-memory-utilization", "0.30", "--max-model-len", "2048"]
    )
    try:
        if not wait_for_health(args.port, server):
            print("  server failed to start")
            return

        def post(payload, label):
            request = urllib.request.Request(
                f"http://localhost:{args.port}/v1/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    print(f"  {label}: HTTP {response.status} (accepted)")
            except urllib.error.HTTPError as exc:
                print(f"  {label}: HTTP {exc.code} (rejected cleanly)")
            except Exception as exc:
                print(f"  {label}: {type(exc).__name__}")

        post({"model": args.model, "prompt": "word " * 100000, "max_tokens": 16}, "oversized prompt")
        post({"model": args.model, "prompt": "hi", "max_tokens": 1000000}, "huge max_tokens")
        post({"model": args.model, "prompt": "hi", "temperature": -5}, "invalid temperature")

        # Client disconnect mid-stream: the server must free the KV blocks.
        request = urllib.request.Request(
            f"http://localhost:{args.port}/v1/completions",
            data=json.dumps(
                # Keep prompt + max_tokens under max_model_len, or this is
                # rejected as a 400 and never exercises the disconnect path.
                {"model": args.model, "prompt": "long story:", "max_tokens": 1500, "stream": True}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            response = urllib.request.urlopen(request, timeout=30)
            response.read(200)
            response.close()
            print("  mid-stream disconnect: client closed")
        except Exception as exc:
            print(f"  mid-stream disconnect: {type(exc).__name__}")

        time.sleep(5)
        metrics = get_metrics(args.port, ("vllm:num_requests_running",))
        running = metrics.get("vllm:num_requests_running", 0.0)
        print(f"  num_requests_running after disconnect: {running:.0f}")
        print("  Verdict: blocks freed" if running == 0 else "  Verdict: request may be leaked")
    finally:
        stop_server(server)


def scenario_idle_detection(args):
    # The 37-day problem. A deployment that serves nothing still passes every
    # health check. Request counts cannot distinguish it, because probes inflate
    # them; generation_tokens_total is the only honest liveness signal.
    print("Distinguishing an idle-but-healthy server from a working one")
    server = start_server(args.model, args.port, ["--gpu-memory-utilization", "0.30"])
    try:
        if not wait_for_health(args.port, server):
            print("  server failed to start")
            return

        names = ("vllm:generation_tokens_total", "vllm:prompt_tokens_total")
        print("  probing /health and /v1/models for 15s, sending no real work")
        for _ in range(15):
            for path in ("/health", "/v1/models"):
                try:
                    urllib.request.urlopen(f"http://localhost:{args.port}{path}", timeout=2).read()
                except (urllib.error.URLError, OSError):
                    pass
            time.sleep(1)
        idle = get_metrics(args.port, names)
        print(f"  after probe-only traffic: {idle}")

        urllib.request.urlopen(
            urllib.request.Request(
                f"http://localhost:{args.port}/v1/completions",
                data=json.dumps(
                    {"model": args.model, "prompt": "real work", "max_tokens": 32}
                ).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=30,
        ).read()
        time.sleep(3)
        busy = get_metrics(args.port, names)
        print(f"  after one real request:   {busy}")
        print("  Verdict: alert on rate(vllm:generation_tokens_total) == 0 while memory is held")
    finally:
        stop_server(server)


SCENARIOS = {
    "startup-oom": ("Request more memory than exists; expect a fast clean failure", scenario_startup_oom),
    "kv-exhaustion": ("Saturate a tiny KV cache; expect preemption, not a crash", scenario_kv_exhaustion),
    "engine-kill": ("Kill EngineCore; measure how long /health lies", scenario_engine_kill),
    "bad-requests": ("Oversized and malformed requests; expect 4xx, not OOM", scenario_bad_requests),
    "idle-detection": ("Prove probe traffic cannot reveal an idle service", scenario_idle_detection),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()

    if args.list or not args.scenario:
        print("-" * 60)
        for name, (description, _) in SCENARIOS.items():
            print(f"{name:<16} {description}")
        print("-" * 60)
        print("python 10_chaos_and_limits.py --scenario startup-oom")
        return 0

    description, function = SCENARIOS[args.scenario]
    print("-" * 60)
    print(f"Scenario: {args.scenario}")
    print(description)
    print("-" * 60)
    function(args)
    print("-" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
