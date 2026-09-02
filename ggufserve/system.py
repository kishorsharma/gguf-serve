"""Host inspection and small console helpers."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
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


@dataclass
class GpuStat:
    index: int
    name: str
    used_gib: float
    total_gib: float
    util_pct: int | None
    temp_c: int | None

    def describe(self) -> str:
        text = f"{self.name}  {self.used_gib:.1f} / {self.total_gib:.1f} GiB"
        if self.total_gib:
            text += f" ({self.used_gib / self.total_gib * 100:.0f}%)"
        if self.util_pct is not None:
            text += f"  util {self.util_pct}%"
        if self.temp_c is not None:
            text += f"  {self.temp_c}C"
        return text


def gpu_stats() -> list[GpuStat]:
    """Current per-GPU memory, utilisation and temperature.

    Returns an empty list when nvidia-smi is missing or fails, so callers can
    treat "no GPU" and "cannot tell" the same way.
    """
    if shutil.which("nvidia-smi") is None:
        return []

    proc = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,"
            "utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if proc.returncode != 0:
        return []

    stats = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue
        try:
            stats.append(
                GpuStat(
                    index=int(parts[0]),
                    name=parts[1],
                    used_gib=float(parts[2]) / 1024,
                    total_gib=float(parts[3]) / 1024,
                    # These read "[N/A]" on some cards and in some VMs.
                    util_pct=_maybe_int(parts[4]),
                    temp_c=_maybe_int(parts[5]),
                )
            )
        except ValueError:
            continue
    return stats


def _maybe_int(value: str) -> int | None:
    try:
        return int(float(value))
    except ValueError:
        return None


def gpu_count() -> int:
    return len(gpu_stats())


def human_size(num_bytes: int) -> str:
    """Byte count as GiB, dropping to MiB so small files do not read as 0.00."""
    gib = num_bytes / 1024**3
    if gib < 1:
        return f"{num_bytes / 1024**2:.0f} MiB"
    return f"{gib:.2f} GiB"


def ram_stats(meminfo: Path | None = None) -> tuple[float, float] | None:
    """(used, total) system RAM in GiB, or None when it cannot be determined.

    Reads /proc/meminfo, so this works on Linux — which covers Kaggle, Colab
    and any CUDA host — and returns None elsewhere.
    """
    meminfo = meminfo or Path("/proc/meminfo")
    if not meminfo.exists():
        return None

    values = {}
    try:
        for line in meminfo.read_text().splitlines():
            key, _, rest = line.partition(":")
            fields = rest.split()
            if fields:
                values[key] = int(fields[0]) / 1024**2  # kB -> GiB
    except (OSError, ValueError):
        return None

    total = values.get("MemTotal")
    if not total:
        return None

    # MemAvailable is the kernel's own estimate of what a new workload can
    # actually claim, which is a truer "free" than MemFree.
    available = values.get("MemAvailable", values.get("MemFree", 0.0))
    return total - available, total


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

    gpus = gpu_stats()
    if gpus:
        for gpu in gpus:
            print(f"   gpu {gpu.index}  : {gpu.describe()}")
        print(f"   vram   : {sum(g.total_gib for g in gpus):.1f} GiB total")
    else:
        warn("No NVIDIA GPU detected. gguf-serve needs CUDA to be useful.")

    ram = ram_stats()
    if ram:
        print(f"   ram    : {ram[1]:.1f} GiB total")

    print(f"   models : {model_dir} ({free_disk_gib(model_dir):.1f} GiB free)")
