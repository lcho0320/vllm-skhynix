# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os
import shutil
import subprocess
import sys
import time

import torch

# Preflight for GB10. Starts no engine. Run it before anything else, and any
# time an engine fails to start.
#
# The failure this exists to prevent:
#   Free memory on device cuda:0 (56.65/119.7 GiB) on startup is less than
#   desired GPU memory utilization (0.92, 110.12 GiB)
#
# vLLM compares gpu_memory_utilization * TOTAL against FREE, so the largest
# legal value is free/total. On GB10 that ceiling moves between runs because
# CPU and GPU share one memory pool and OS page cache counts against it.

# vLLM consumes MORE than gpu_memory_utilization * total. That product is its
# internal budget (weights + activations + KV cache); on top of it sit the CUDA
# context, cuBLAS/cuDNN workspaces and allocator fragmentation, which are not
# counted. Measured on this box with Qwen2.5-0.5B:
#
#   util 0.20 -> budget 23.9 GiB, actually consumed 26.5 GiB  (+2.6)
#   util 0.40 -> budget 47.9 GiB, actually consumed 50.7 GiB  (+2.8)
#
# The overshoot is roughly CONSTANT, not proportional, which is what you would
# expect from a fixed CUDA context plus workspaces. Budget for it separately.
ENGINE_OVERHEAD_GIB = 3.0

# What is left for the operating system after the engine has taken its share.
HOST_RESERVE_GIB = 5.0

# Memory to leave unclaimed, in GiB. Absolute, not a fraction of the pool.
#
# What it has to absorb, measured on this box rather than assumed:
#
#   short-term drift    negligible. Sampling torch free memory 20 times over 30s
#                       on an idle box gave a 0.03 percentage-point spread, so
#                       there is nothing here to hedge against.
#   OS growth           the host keeps running while vLLM serves. On a discrete
#                       GPU this costs you nothing; here it comes out of the
#                       same pool.
#   allocator slack     you cannot hand the CUDA allocator 100% of free memory
#                       and expect one contiguous arena.
#   page cache churn    reading weights fills cache, which torch counts as used.
#                       Reclaimable, but there is a window during load.
#
# Why absolute. A fraction of TOTAL is the wrong shape, because what the host
# needs does not scale with how much happens to be free. Reserving 0.15 of a
# 119.7 GiB pool means holding back ~18 GiB — fine when 110 GiB is free, absurd
# when 20 GiB is, where it would reserve 70% of what you have. An absolute
# reserve degrades gracefully at both ends.
#
# It does NOT protect you from another workload appearing: the k8s pods on this
# node took ~40 GiB with no warning. For that, read what is resident (this
# script prints it) rather than trusting any margin.
RESERVE_GIB = ENGINE_OVERHEAD_GIB + HOST_RESERVE_GIB


def gib(num_bytes: float) -> str:
    return f"{num_bytes / 2**30:.1f} GiB"


def report_device():
    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    print(f"torch: {torch.__version__}")
    print(f"Device: {props.name}")
    print(f"Capability: sm_{major}{minor}")
    print(f"Total memory: {gib(props.total_memory)}")

    # Compute capability gates which quantization kernels exist. sm_120/121 is
    # Blackwell (this box); sm_90 is Hopper.
    if major >= 12:
        print("Blackwell: FP8 and NVFP4 kernels available")
    elif (major, minor) == (9, 0):
        print("Hopper: FP8 available, NVFP4 is not")
    else:
        print("Pre-Hopper: no FP8; use AWQ/GPTQ INT4")


def report_reclaimable(torch_free_bytes: int):
    # torch's "free" and the kernel's "available" are different questions.
    # torch reports memory not currently allocated; the kernel reports memory a
    # new allocation could obtain, which INCLUDES page cache it would evict.
    # On GB10 the gap is large because model files you have read sit in cache.
    #
    # vLLM's startup check uses the torch number, so the ceiling is computed
    # from that. But knowing the gap tells you whether a failed check is real
    # scarcity or just cache you can drop.
    try:
        with open("/proc/meminfo") as handle:
            info = {
                line.split(":")[0]: int(line.split()[1]) * 1024
                for line in handle
                if ":" in line
            }
    except OSError:
        return

    available = info.get("MemAvailable")
    cached = info.get("Cached", 0) + info.get("Buffers", 0)
    if available is None:
        return

    print(f"  Kernel MemAvailable: {gib(available)}")
    print(f"  Page cache (reclaimable): {gib(cached)}")
    gap = available - torch_free_bytes
    if gap > 2 * 2**30:
        print(f"  -> the kernel could free ~{gib(gap)} more than torch reports.")
        print("     If a run fails the memory check, dropping caches may be enough:")
        print("     sync && sudo sysctl -w vm.drop_caches=3")


def report_other_processes():
    if not shutil.which("nvidia-smi"):
        return
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory,process_name",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    listing = result.stdout.strip()
    print("\nOther GPU processes:")
    print(listing if listing else "  (none)")
    if listing:
        # crictl stop will not keep a k8s pod down; kubelet restarts it.
        # Desired state lives in the API server, so scale the Deployment.
        print("  If these are k8s pods, scale the Deployment rather than killing them:")
        print("  sudo KUBECONFIG=/etc/kubernetes/admin.conf \\")
        print("    kubectl scale deploy <name> -n llm-inference --replicas=0")


def reclaim_page_cache() -> bool:
    # Page cache is reclaimable: the kernel evicts it under pressure anyway, so
    # dropping it by hand changes nothing about what is POSSIBLE. It only
    # changes what torch REPORTS free — which matters because vLLM's startup
    # check reads that number and is not reclaim-aware.
    #
    # sync first: drop_caches will not discard dirty pages, so unwritten data
    # would otherwise stay resident.
    if os.geteuid() != 0:
        print("Reclaiming page cache needs root. Re-run with sudo:")
        print(f"  sudo {sys.executable} {' '.join(sys.argv)}")
        return False

    before = torch.cuda.mem_get_info()[0]
    subprocess.run(["sync"], check=False)
    try:
        with open("/proc/sys/vm/drop_caches", "w") as handle:
            handle.write("3")
    except OSError as exc:
        print(f"Could not drop caches: {exc}")
        return False
    time.sleep(2)
    after = torch.cuda.mem_get_info()[0]

    print(f"\nDropped page cache: {gib(before)} free -> {gib(after)} free")
    print(f"  reclaimed {gib(after - before)}")
    # The cost is a cold cache: the next model load re-reads weights from disk.
    print("  next model load will be slower (weights re-read from disk)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reclaim",
        action="store_true",
        help="drop page cache before reporting (needs root)",
    )
    args = parser.parse_args()

    print("-" * 60)
    if not torch.cuda.is_available():
        print("No CUDA device. Activate the venv:")
        print("  source /home/leeksun/testingEnvironment/vLLM_testing/vllm/.venv/bin/activate")
        return

    report_device()

    if args.reclaim:
        reclaim_page_cache()

    try:
        import vllm

        print(f"vllm: {vllm.__version__}")
    except Exception as exc:
        print(f"vllm import failed: {exc}")

    # mem_get_info reflects everything resident in the unified pool, including
    # OS page cache. Expect tens of GiB "in use" with no GPU processes at all.
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print("\nUnified memory pool (shared by CPU and GPU):")
    print(f"  Total: {gib(total_bytes)}")
    print(f"  Used:  {gib(total_bytes - free_bytes)}")
    print(f"  Free:  {gib(free_bytes)} ({100 * free_bytes / total_bytes:.0f}%)")

    report_reclaimable(free_bytes)
    report_other_processes()

    # This is the number vLLM actually tests against, so the ceiling is exact:
    # a request above free/total fails, one below it passes.
    ceiling = free_bytes / total_bytes
    reserve_bytes = RESERVE_GIB * 2**30
    suggested = max(0.05, (free_bytes - reserve_bytes) / total_bytes)

    print("\ngpu_memory_utilization guidance:")
    print(f"  Hard ceiling: {ceiling:.2f} (anything above fails immediately)")
    print(
        f"  Suggested:    {suggested:.2f} "
        f"({ENGINE_OVERHEAD_GIB:.0f} GiB engine overhead + {HOST_RESERVE_GIB:.0f} GiB host)"
    )
    print(f"  vLLM default 0.90+ {'will fail here' if ceiling < 0.90 else 'fits'}")

    if free_bytes <= reserve_bytes * 1.5:
        # Below roughly 12 GiB free there is nothing meaningful left after the
        # reserve. Reclaim memory instead of shaving the fraction.
        print("\n  Free memory is close to the reserve. Free some first:")
        print("    sync && sudo sysctl -w vm.drop_caches=3")
        print("    check for other tenants above")

    print(f"\n  llm = LLM(model=..., gpu_memory_utilization={suggested:.2f})")
    print("-" * 60)


if __name__ == "__main__":
    main()
