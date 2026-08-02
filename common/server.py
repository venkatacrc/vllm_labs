"""Lifecycle management for a `vllm serve` subprocess, used by every lab.

Every lab follows the same pattern:

    with VLLMServer(model, extra_args=["--dtype", "float32"]) as base_url:
        result = bench.run_benchmark(base_url=base_url, model=model, ...)

`VLLMServer` launches `vllm serve` in its own process group, polls `/health`
until the engine is ready (or raises `ServerStartError` with a pointer to the
log file), and on teardown sends SIGTERM (then SIGKILL if needed) to the
whole process group so vLLM's worker subprocesses don't linger and hold GPU
memory into the next lab.
"""
from __future__ import annotations

import itertools
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request

from . import config, gpu_utils

_port_counter = itertools.count(config.DEFAULT_PORT)


def next_port() -> int:
    return next(_port_counter)


class ServerStartError(RuntimeError):
    pass


class VLLMServer:
    def __init__(
        self,
        model: str,
        *,
        port: int | None = None,
        gpus: str = config.DEFAULT_GPU,
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,
        name: str | None = None,
        startup_timeout: int = config.SERVER_STARTUP_TIMEOUT_S,
    ) -> None:
        self.model = model
        self.port = port if port is not None else next_port()
        self.gpus = gpus
        self.extra_args = extra_args or []
        self.env_overrides = env or {}
        self.name = name or model.split("/")[-1]
        self.startup_timeout = startup_timeout
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.proc: subprocess.Popen | None = None
        self._log_file = None
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in self.name)
        self.log_path = config.LOGS_DIR / f"{safe_name}_{self.port}.log"

    # -- lifecycle -----------------------------------------------------
    def start(self) -> str:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpus)
        env["HF_HOME"] = config.HF_HOME
        env.setdefault("VLLM_LOGGING_LEVEL", "INFO")
        env.update(self.env_overrides)

        cmd = [
            "vllm", "serve", self.model,
            "--port", str(self.port),
            "--host", "127.0.0.1",
            *self.extra_args,
        ]

        print(f"\n[server] Launching '{self.name}' (gpus={self.gpus}, port={self.port})")
        print(f"[server] cmd: {' '.join(cmd)}")
        if self.env_overrides:
            print(f"[server] env overrides: {self.env_overrides}")
        print(f"[server] log: {self.log_path}")

        self._log_file = open(self.log_path, "w")
        self.proc = subprocess.Popen(
            cmd,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,
        )
        try:
            self._wait_healthy()
        except ServerStartError:
            self.stop()
            raise
        return self.base_url

    def _wait_healthy(self) -> None:
        assert self.proc is not None
        t0 = time.time()
        deadline = t0 + self.startup_timeout
        url = f"{self.base_url}/health"
        while time.time() < deadline:
            ret = self.proc.poll()
            if ret is not None:
                raise ServerStartError(
                    f"'{self.name}' exited early (code={ret}) before becoming healthy. "
                    f"Tail of log: {self._log_tail()}\nFull log: {self.log_path}"
                )
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        print(f"[server] '{self.name}' healthy after {time.time() - t0:.0f}s")
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                pass
            time.sleep(3)
        raise ServerStartError(
            f"'{self.name}' did not become healthy within {self.startup_timeout}s. "
            f"Tail of log: {self._log_tail()}\nFull log: {self.log_path}"
        )

    def _log_tail(self, n_lines: int = 25) -> str:
        try:
            with open(self.log_path) as f:
                lines = f.readlines()
            return "".join(lines[-n_lines:])
        except OSError:
            return "(log unavailable)"

    def stop(self) -> None:
        if self.proc is not None:
            try:
                pgid = os.getpgid(self.proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                for _ in range(30):
                    if self.proc.poll() is not None:
                        break
                    time.sleep(1)
                if self.proc.poll() is None:
                    os.killpg(pgid, signal.SIGKILL)
                    self.proc.wait(timeout=10)
            except ProcessLookupError:
                pass
            except Exception as exc:  # noqa: BLE001
                print(f"[server] WARNING: error stopping '{self.name}': {exc}")
            finally:
                self.proc = None
                print(f"[server] '{self.name}' stopped.")
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        gpu_utils.wait_for_free(self.gpus)

    def __enter__(self) -> str:
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
