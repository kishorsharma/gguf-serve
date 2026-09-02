"""Host inspection and small console helpers.

Originally cell "01 - Verify Kaggle hardware/runtime".
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def banner(text: str) -> None:
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def step(text: str) -> None:
    print(f"\n>> {text}")


def ok(text: str) -> None:
    print(f"   [ok] {text}")


def warn(text: str) -> None:
    print(f"   [!] {text}")


def run(cmd: list[str] | str, check: bool = False) -> subprocess.CompletedProcess:
    """Run a command and capture its output. Only raises when `check` is set."""
    proc = subprocess.run(
        cmd,
        shell=isinstance(cmd, str),
        text=True,
        capture_output=True,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {cmd}\n{proc.stderr.strip()}"
        )
    return proc


def gpu_summary() -> str:
    """One line per GPU, or an empty string when nvidia-smi is unavailable."""
    if shutil.which("nvidia-smi") is None:
        return ""
    proc = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader",
        ]
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def gpu_count() -> int:
    summary = gpu_summary()
    return len(summary.splitlines()) if summary else 0


def free_disk_gib(path: Path) -> float:
    """Free space on the filesystem holding `path`, walking up if it is absent."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1024**3


def report(model_dir: Path) -> None:
    """Print the hardware summary used to sanity-check a run before loading."""
    step("Checking host")

    print(f"   python : {sys.version.split()[0]}")

    nvcc = run(["nvcc", "--version"]) if shutil.which("nvcc") else None
    if nvcc is not None and nvcc.returncode == 0:
        release = next(
            (ln.strip() for ln in nvcc.stdout.splitlines() if "release" in ln),
            "unknown",
        )
        print(f"   cuda   : {release}")
    else:
        print("   cuda   : nvcc not found")

    summary = gpu_summary()
    if summary:
        for line in summary.splitlines():
            print(f"   gpu    : {line}")
    else:
        warn("No NVIDIA GPU detected. Qwen-K needs CUDA to be useful.")

    print(f"   models : {model_dir} ({free_disk_gib(model_dir):.1f} GiB free)")
