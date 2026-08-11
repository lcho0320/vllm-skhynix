# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import shutil
import subprocess
import sys

# Runs a chosen bpftrace program against the running system for a fixed
# duration. The narrative version, including what eBPF can and cannot see, is
# 09_ebpf_observability.txt.
#
# The short version: vLLM has no eBPF hooks and will not get any. eBPF sees
# syscalls, page faults, block I/O, TCP and scheduler latency, and stops at the
# driver boundary — it can see nothing inside the GPU. Use DCGM for GPU-side
# truth, /metrics for engine-side truth, and eBPF for the gap between the NIC
# and the driver, which is where "GPU looks idle but latency is bad" lives.
#
# Every program here needs root. Run vLLM in one terminal and this in another:
#   sudo python 09_ebpf_observability.py --list
#   sudo python 09_ebpf_observability.py --recipe page-faults --seconds 15

# Each recipe is (description, bpftrace program). The programs use an
# interval probe to print periodically; the harness supplies the time limit.
RECIPES = {
    "page-faults": (
        "Page faults per process. The GB10-specific one: unified memory means "
        "host memory pressure IS GPU memory pressure, so major faults during "
        "serving mean you have squeezed the host and latency will be erratic.",
        """
        software:major-faults:1 { @major[comm] = count(); }
        software:minor-faults:1 { @minor[comm] = count(); }
        interval:s:5 {
          printf("--- major faults (nothing listed below = zero, which is good) ---\\n");
          print(@major); clear(@major);
          printf("--- top minor-fault processes (normal, informational) ---\\n");
          print(@minor, 5); clear(@minor);
        }
        """,
    ),
    "runq-latency": (
        "Scheduler run-queue delay. The engine's Python threads must be "
        "scheduled promptly to launch the next decode step; a tail into "
        "milliseconds shows up to users as inter-token latency jitter.",
        """
        tracepoint:sched:sched_wakeup { @qtime[args->pid] = nsecs; }
        tracepoint:sched:sched_switch
        /@qtime[args->next_pid]/
        {
          @runq_delay_us = hist((nsecs - @qtime[args->next_pid]) / 1000);
          delete(@qtime[args->next_pid]);
        }
        interval:s:10 { print(@runq_delay_us); clear(@runq_delay_us); clear(@qtime); }
        """,
    ),
    "gpu-ioctl": (
        "NVIDIA driver ioctl rate and latency. Every CUDA operation crossing "
        "into the driver is an ioctl, so this is a proxy for how hard the "
        "engine drives the GPU. A long tail with flat GPU utilization means "
        "driver-side contention rather than compute.",
        """
        tracepoint:syscalls:sys_enter_ioctl
        /comm == "VLLM::EngineCor" || comm == "pt_main_thread" || comm == "python3"/
        { @start[tid] = nsecs; @count = count(); }

        tracepoint:syscalls:sys_exit_ioctl
        /@start[tid]/
        { @ioctl_us = hist((nsecs - @start[tid]) / 1000); delete(@start[tid]); }

        interval:s:5 { print(@count); clear(@count); print(@ioctl_us); clear(@ioctl_us); }
        """,
    ),
    "block-io": (
        "Block device throughput. Run this while a large model loads to see "
        "where cold-start time goes. Well under the device's rated throughput "
        "means the bottleneck is deserialization, not the disk.",
        """
        tracepoint:block:block_rq_issue { @bytes = sum(args->bytes); @reqs = count(); }
        interval:s:2 {
          printf("read %8d MB/s  %6d reqs/s\\n", @bytes / 1024 / 1024 / 2, @reqs / 2);
          clear(@bytes); clear(@reqs);
        }
        """,
    ),
    "tcp-accept": (
        "TCP connection rate. vLLM's own metrics start counting when the "
        "engine sees a request; the kernel sees it earlier. The gap is time "
        "lost in the API server, invisible from inside vLLM.",
        """
        kretprobe:inet_csk_accept { @accepts = count(); }
        kprobe:tcp_v4_connect { @connects[comm] = count(); }
        interval:s:5 { print(@accepts); clear(@accepts); print(@connects); clear(@connects); }
        """,
    ),
    "offcpu": (
        "Kernel stacks where the engine blocks. Expect futex (IPC with the API "
        "server process) and GPU driver waits; anything else deserves an "
        "explanation.",
        """
        kprobe:finish_task_switch
        /comm == "VLLM::EngineCor"/
        { @off[kstack] = count(); }
        interval:s:15 { print(@off, 5); clear(@off); }
        """,
    ),
}


def check_prerequisites() -> bool:
    if shutil.which("bpftrace") is None:
        print("bpftrace not installed: sudo apt install bpftrace")
        return False
    # bpftrace needs CAP_BPF/CAP_SYS_ADMIN; in practice that means root.
    if hasattr(__import__("os"), "geteuid") and __import__("os").geteuid() != 0:
        print("These probes need root. Re-run with sudo:")
        print(f"  sudo {sys.executable} {' '.join(sys.argv)}")
        return False
    return True


def list_recipes():
    print("-" * 60)
    for name, (description, _program) in RECIPES.items():
        print(f"{name}")
        for line in description.split(". "):
            if line.strip():
                print(f"    {line.strip().rstrip('.')}.")
    print("-" * 60)
    print("sudo python 09_ebpf_observability.py --recipe page-faults --seconds 15")


def run_recipe(name: str, seconds: int):
    description, program = RECIPES[name]
    print("-" * 60)
    print(f"Recipe: {name}")
    print(description)
    print(f"Running for {seconds}s (Ctrl-C to stop early)")
    print("-" * 60)

    # The child writes to the same stdout we do. When that stdout is a pipe
    # rather than a terminal it is block-buffered, so everything printed above
    # would still be sitting in our buffer while bpftrace writes directly to the
    # file descriptor — and the header would appear AFTER the trace output.
    # Flush explicitly here rather than relying on flush=True on whichever print
    # happens to be last, which silently breaks when a line is added below it.
    sys.stdout.flush()
    sys.stderr.flush()

    # bpftrace has no built-in total time limit, so the harness enforces one and
    # sends SIGTERM, which makes bpftrace print its maps and exit cleanly.
    try:
        subprocess.run(["bpftrace", "-e", program], timeout=seconds)
    except subprocess.TimeoutExpired:
        pass
    except KeyboardInterrupt:
        pass

    # Same hazard in reverse: flush the child's inherited output before we add
    # our footer, so the two do not interleave.
    sys.stdout.flush()
    print("-" * 60)
    print("Done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--recipe", choices=sorted(RECIPES))
    parser.add_argument("--seconds", type=int, default=15)
    args = parser.parse_args()

    if args.list or not args.recipe:
        list_recipes()
        return 0
    if not check_prerequisites():
        return 1
    run_recipe(args.recipe, args.seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
