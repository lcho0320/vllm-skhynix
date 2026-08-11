# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dual-mode helper so one example can run offline or online.

Teaching scaffolding, not a pattern to ship. vLLM deliberately keeps offline and
online examples in separate files that share no code, because the modes differ in
config, batching, error handling, introspection and lifecycle. Rationale in
../MODES.md.
"""

import argparse
import os
import signal
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

DEFAULT_MODEL = os.environ.get("VLLM_SK_MODEL", "facebook/opt-125m")
DEFAULT_URL = os.environ.get("VLLM_SK_URL", "http://localhost:8000/v1")

SERVER_STARTUP_TIMEOUT = 600


def add_mode_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--mode",
        choices=["offline", "online"],
        default=os.environ.get("VLLM_SK_MODE", "offline"),
        help="offline = in-process vllm.LLM; online = HTTP to a `vllm serve`",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument(
        "--no-autostart",
        action="store_true",
        help="in online mode, fail instead of starting a server when none is found",
    )
    return parser


def server_is_up(base_url: str) -> bool:
    # /health is the cheapest readiness signal and needs no model name.
    health = base_url.rsplit("/v1", 1)[0] + "/health"
    try:
        with urllib.request.urlopen(health, timeout=2):
            return True
    except (urllib.error.URLError, OSError):
        return False


VENV_PYTHON = "/home/leeksun/testingEnvironment/vLLM_testing/vllm/.venv/bin/python"


def check_interpreter() -> None:
    # Every path here needs a working vLLM install, and online mode also needs
    # the openai client. Verify both up front, because the failure modes
    # otherwise are misleading:
    #
    #   - running from the repo's PARENT directory makes `import vllm` resolve
    #     to the SOURCE FOLDER as a namespace package. It imports "fine" but has
    #     no submodules, so the server dies with a bare exit code 1.
    #   - the system python has neither package, producing a raw traceback from
    #     whichever import happens to run first.
    #
    # Both really mean "wrong interpreter", so say that once and clearly.
    import importlib.util

    missing = []
    for module in ("vllm", "openai"):
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            missing.append(module)
        elif module == "vllm" and spec.origin is None:
            # origin is None for a NAMESPACE package. That is the parent-directory
            # trap: the repo folder is named `vllm`, so from its parent `import
            # vllm` binds to the directory. Submodules still resolve, which makes
            # a naive find_spec check pass, but nothing defined in vllm/__init__.py
            # exists — so `from vllm import SamplingParams` fails deep inside the
            # server with a confusing ImportError.
            missing.append("vllm (shadowed by the source directory)")

    if not missing:
        return

    argv = " ".join(shlex.quote(part) for part in sys.argv)
    sys.exit(
        f"This interpreter cannot run the examples: {sys.executable}\n"
        f"Cannot import: {', '.join(missing)}\n\n"
        f"Use the project venv:\n"
        f"  {VENV_PYTHON} {argv}\n"
        f"or activate it first:\n"
        f"  source /home/leeksun/testingEnvironment/vLLM_testing/vllm/.venv/bin/activate\n\n"
        f"Note: running from the repo's PARENT directory also breaks this, because\n"
        f"`import vllm` then resolves to the source folder instead of the install."
    )


def start_server(args: argparse.Namespace) -> subprocess.Popen:
    port = urlparse(args.url).port or 8000
    # Invoke through the module rather than the `vllm` console script, so this
    # works whether or not the venv is activated on PATH.
    #
    # --enable-prompt-tokens-details is included because without it
    # usage.prompt_tokens_details is null and every prefix-cache measurement
    # reports 0%, which looks exactly like a broken cache.
    command = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", args.model,
        "--port", str(port),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
        "--enable-prompt-tokens-details",
    ]
    print(f"No server at {args.url}; starting one (this takes ~15-60s)")

    # Keep the server's output. Sending it to DEVNULL makes a failed start
    # impossible to diagnose: all you get is an exit code. A temp file keeps it
    # out of our stdout while the server runs, and lets us show the tail if it
    # dies.
    log = tempfile.NamedTemporaryFile(
        mode="w+", suffix=".log", prefix="vllm-server-", delete=False
    )

    # start_new_session gives the server its own process group, so teardown can
    # signal the whole group without touching this script, and a stray Ctrl-C
    # cannot orphan a process holding GPU memory.
    process = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        # Run from a neutral directory. `python -m` puts the working directory
        # on sys.path, so launching from the repo's parent would let the source
        # folder named `vllm` shadow the installed package inside the server
        # process. Model arguments are HF ids or absolute paths, so nothing here
        # depends on the caller's cwd.
        cwd=tempfile.gettempdir(),
    )
    # Quote the command so it is directly copy-pasteable, including the
    # interpreter. Omitting sys.executable produced a command that could not run.
    printable = " ".join(shlex.quote(part) for part in command)

    deadline = time.time() + SERVER_STARTUP_TIMEOUT
    while time.time() < deadline:
        if process.poll() is not None:
            log.flush()
            tail = _log_tail(log.name)
            sys.exit(
                f"Server exited with code {process.returncode} before becoming healthy.\n\n"
                f"Last lines of its output:\n{tail}\n"
                f"Full log: {log.name}\n"
                f"Reproduce with:\n  {printable}"
            )
        if server_is_up(args.url):
            print("Server ready")
            return process
        time.sleep(2)

    stop_server(process)
    sys.exit(
        f"Server did not become healthy within {SERVER_STARTUP_TIMEOUT}s\n"
        f"Last lines of its output:\n{_log_tail(log.name)}\n"
        f"Full log: {log.name}"
    )


def _log_tail(path: str, lines: int = 15) -> str:
    try:
        with open(path, errors="replace") as handle:
            content = handle.read().splitlines()
    except OSError:
        return "  (log unavailable)"
    if not content:
        return "  (no output)"
    return "\n".join(f"  {line}" for line in content[-lines:])


def stop_server(process: subprocess.Popen) -> None:
    # SIGTERM, not SIGKILL: vLLM installs handlers for SIGTERM/SIGINT and
    # unwinds the engine cleanly. SIGKILL skips that and can leave shared memory
    # and the EngineCore child behind. SIGKILL is only the fallback.
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


class Backend:
    mode: str
    # Set by chat() so an example can prove prefix caching rather than claim it.
    last_prompt_tokens: int = 0
    last_cached_tokens: int = 0

    def chat(self, messages: list[dict], **kwargs: Any) -> str:
        raise NotImplementedError

    def close(self) -> None:
        pass


class OfflineBackend(Backend):
    mode = "offline"

    def __init__(self, args: argparse.Namespace, **llm_kwargs: Any):
        from vllm import LLM

        kwargs = dict(
            model=args.model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            seed=0,
        )
        kwargs.update(llm_kwargs)
        self.llm = LLM(**kwargs)

    def _sampling_params(self, kwargs: dict):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        structured = None
        if any(key in kwargs for key in ("json_schema", "choice", "regex")):
            structured = StructuredOutputsParams(
                json=kwargs.pop("json_schema", None),
                choice=kwargs.pop("choice", None),
                regex=kwargs.pop("regex", None),
            )
        return SamplingParams(
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 64),
            structured_outputs=structured,
        )

    def chat(self, messages: list[dict], **kwargs: Any) -> str:
        # LLM.chat applies the model's chat template and takes the FULL message
        # list. Genuine multi-turn, offline. It is stateless: nothing is
        # remembered between calls, so the history you pass IS the state.
        params = self._sampling_params(kwargs)
        output = self.llm.chat(messages, params, use_tqdm=False)[0]
        self.last_prompt_tokens = len(output.prompt_token_ids)
        self.last_cached_tokens = output.num_cached_tokens or 0
        return output.outputs[0].text

    def close(self) -> None:
        from vllm.distributed import cleanup_dist_env_and_memory

        del self.llm
        cleanup_dist_env_and_memory()


class OnlineBackend(Backend):
    mode = "online"

    def __init__(self, args: argparse.Namespace, **_: Any):
        from openai import OpenAI

        # Only tear down a server we started ourselves. Reusing one you already
        # have running is the common case, and killing it on exit would be rude.
        self._server = None

        if not server_is_up(args.url):
            if args.no_autostart:
                sys.exit(
                    f"No vLLM server at {args.url} and --no-autostart was given.\n"
                    f"Start one:\n  vllm serve {args.model} "
                    f"--gpu-memory-utilization {args.gpu_memory_utilization} "
                    f"--max-model-len {args.max_model_len} "
                    f"--port {urlparse(args.url).port or 8000} "
                    f"--enable-prompt-tokens-details"
                )
            self._server = start_server(args)
        else:
            print(f"Reusing the server already running at {args.url}")

        self.client = OpenAI(base_url=args.url, api_key="EMPTY")
        self.model = args.model
        served = [model.id for model in self.client.models.list().data]
        # The server names the model; honour it rather than guessing. This also
        # covers reusing a server that happens to be serving something else.
        if served and self.model not in served:
            print(f"Server serves {served[0]!r}; using that")
            self.model = served[0]

    def _extra_body(self, kwargs: dict) -> dict:
        # Anything outside the OpenAI spec rides in extra_body.
        structured = {}
        for key, field in (("json_schema", "json"), ("choice", "choice"), ("regex", "regex")):
            if key in kwargs:
                structured[field] = kwargs.pop(key)
        return {"structured_outputs": structured} if structured else {}

    def chat(self, messages: list[dict], **kwargs: Any) -> str:
        extra_body = self._extra_body(kwargs)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 64),
            extra_body=extra_body or None,
        )
        # Online equivalent of RequestOutput.num_cached_tokens. Requires the
        # server to run with --enable-prompt-tokens-details; without it the
        # field is null and a working cache looks broken.
        usage = response.usage
        self.last_prompt_tokens = usage.prompt_tokens if usage else 0
        details = getattr(usage, "prompt_tokens_details", None) if usage else None
        self.last_cached_tokens = getattr(details, "cached_tokens", 0) or 0
        return response.choices[0].message.content or ""

    def close(self) -> None:
        if self._server is not None:
            print("Stopping the server this example started")
            stop_server(self._server)
            self._server = None

    def chat_stream(self, messages: list[dict], **kwargs: Any):
        # Streaming needs an HTTP connection to push deltas, so it is online
        # only. Offline generate() returns everything at once.
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 64),
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def get_backend(args: argparse.Namespace, **llm_kwargs: Any) -> Backend:
    # Fail with a clear message before any import blows up in a confusing way.
    check_interpreter()
    if args.mode == "online":
        return OnlineBackend(args)
    return OfflineBackend(args, **llm_kwargs)
