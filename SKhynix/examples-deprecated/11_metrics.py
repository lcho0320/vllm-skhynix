# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import re
import sys
import time
import urllib.error
import urllib.request

# Scrapes /metrics from a running server and filters to the series worth
# alerting on. Start a server first:
#   vllm serve facebook/opt-125m --gpu-memory-utilization 0.30 --port 8000
#
# Metric names were read out of vllm/v1/metrics/loggers.py rather than docs,
# because two of them are easy to get wrong: the KV gauge is
# kv_cache_usage_perc (not gpu_cache_usage_perc) and ITL is
# inter_token_latency_seconds (not time_per_output_token_seconds).
#
# prometheus_client appends _total to Counters, so a Counter declared as
# "vllm:num_preemptions" is scraped as "vllm:num_preemptions_total".

DEFAULT_URL = "http://localhost:8000/metrics"

# name -> (what it measures, what to do about it)
KEY_METRICS = {
    "vllm:num_requests_running": (
        "requests currently in the batch",
        "pinned at max_num_seqs means saturated; add replicas",
    ),
    "vllm:num_requests_waiting": (
        "queue depth",
        "sustained above 0 is the primary saturation signal; alert on this",
    ),
    "vllm:kv_cache_usage_perc": (
        "KV cache occupancy, 0-1",
        "near 1.0 means preemption is imminent",
    ),
    "vllm:num_preemptions_total": (
        "requests evicted and recomputed",
        "any sustained rate means the cache is undersized; leading indicator",
    ),
    "vllm:prompt_tokens_total": (
        "cumulative prefill tokens",
        "flat while healthy means an idle service",
    ),
    "vllm:generation_tokens_total": (
        "cumulative decode tokens",
        "the honest 'is this doing work' metric; probes inflate request counts",
    ),
    "vllm:time_to_first_token_seconds": (
        "TTFT histogram",
        "p99 is usually the user-facing SLA; rises with queueing",
    ),
    "vllm:inter_token_latency_seconds": (
        "ITL histogram",
        "rises with batch size; the throughput/latency tradeoff measured",
    ),
    "vllm:e2e_request_latency_seconds": (
        "end-to-end latency histogram",
        "what the client actually experiences",
    ),
    "vllm:request_success_total": (
        "completed requests by finish reason",
        "all-length means max_tokens is clipping answers",
    ),
    "vllm:prefix_cache_hits_total": (
        "prefix cache hits served from GPU",
        "hit rate degrades under memory pressure, exactly when you are busiest",
    ),
    "vllm:prefix_cache_queries_total": (
        "prefix cache lookups",
        "denominator for hit rate",
    ),
    "vllm:external_prefix_cache_hits_total": (
        "hits served from a KV connector or offload tier",
        "only nonzero when offloading or a connector is configured",
    ),
}


def fetch(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.read().decode()
    except urllib.error.URLError as exc:
        print(f"Cannot reach {url}: {exc}")
        print("Start a server: vllm serve facebook/opt-125m --port 8000")
        sys.exit(1)


def parse(text: str) -> dict[str, float]:
    # Minimal Prometheus text parser. Sums across label sets, so a multi-engine
    # server reports totals rather than per-engine values.
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)$", line)
        if not match:
            continue
        name, _labels, raw_value = match.groups()
        try:
            values[name] = values.get(name, 0.0) + float(raw_value)
        except ValueError:
            continue
    return values


def snapshot(url: str):
    metrics = parse(fetch(url))
    print("-" * 60)

    for name, (meaning, action) in KEY_METRICS.items():
        value = metrics.get(name)
        if value is None:
            # Histograms expose _sum/_count/_bucket rather than a bare name.
            total, count = metrics.get(f"{name}_sum"), metrics.get(f"{name}_count")
            if total is not None and count:
                print(f"{name}: mean {total / count:.4f}s over {int(count)} observations")
                print(f"  {meaning} | {action}")
            continue
        print(f"{name}: {value:,.4f}")
        print(f"  {meaning} | {action}")

    print("-" * 60)
    hits = metrics.get("vllm:prefix_cache_hits_total")
    queries = metrics.get("vllm:prefix_cache_queries_total")
    if hits is not None and queries:
        print(f"Prefix cache hit rate: {100 * hits / queries:.1f}%")

    # An abandoned deployment passes every health check while generating nothing.
    # This is the check that would have caught two pods idling for 37 days.
    if metrics.get("vllm:generation_tokens_total", 0.0) == 0:
        print("generation_tokens_total is 0: this server has served no real inference")
    print("-" * 60)


def watch(url: str, interval: float):
    print(f"{'time':>8} {'running':>8} {'waiting':>8} {'kv_used':>8} {'tok/s':>8} {'preempt':>8}")
    previous_tokens, previous_time = None, None
    try:
        while True:
            metrics = parse(fetch(url))
            now = time.time()
            generated = metrics.get("vllm:generation_tokens_total", 0.0)
            rate = 0.0
            if previous_tokens is not None and now > previous_time:
                rate = (generated - previous_tokens) / (now - previous_time)
            previous_tokens, previous_time = generated, now
            print(
                f"{time.strftime('%H:%M:%S'):>8} "
                f"{metrics.get('vllm:num_requests_running', 0):>8.0f} "
                f"{metrics.get('vllm:num_requests_waiting', 0):>8.0f} "
                f"{100 * metrics.get('vllm:kv_cache_usage_perc', 0):>7.1f}% "
                f"{rate:>8.0f} "
                f"{metrics.get('vllm:num_preemptions_total', 0):>8.0f}"
            )
            time.sleep(interval)
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()

    if args.watch:
        watch(args.url, args.interval)
    else:
        snapshot(args.url)


if __name__ == "__main__":
    main()
