"""Acquiring and loading the GGUF.

Originally cells "03 - Find/download/validate exact Q5_K_XL GGUF", "04 - Load
Qwen3.8-27B Q5_K_XL across both T4s" and "05 - Verify GPU allocation".

The download deliberately uses curl against the public Hugging Face `resolve`
URL rather than `huggingface_hub.hf_hub_download`. Pulling the Hub into the
process was the original source of a broken `huggingface_hub` import once
Gradio upgraded it mid-session.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from qwenk import config
from qwenk.system import free_disk_gib, gpu_count, gpu_summary, ok, step, warn


def validate(path: Path) -> tuple[bool, str]:
    """Check that `path` is a complete, plausible GGUF file."""
    if not path.exists():
        return False, "missing"

    size_gib = path.stat().st_size / 1024**3
    if size_gib < config.MIN_MODEL_GIB:
        return False, f"incomplete ({size_gib:.2f} GiB, expected >= {config.MIN_MODEL_GIB})"

    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            return False, "not a GGUF file (bad magic bytes)"

    return True, f"{size_gib:.2f} GiB"


def acquire() -> Path:
    """Return a validated model path, downloading it only if necessary."""
    step("Locating model")

    path = config.model_path()

    valid, detail = validate(path)
    if valid:
        ok(f"already present: {path} ({detail})")
        return path

    print(f"   {path}: {detail}")

    if path.exists():
        print("   removing the incomplete file before retrying")
        path.unlink()

    needed = config.MIN_MODEL_GIB * 1.05
    available = free_disk_gib(path.parent)
    if available < needed:
        raise SystemExit(
            f"Not enough disk space in {path.parent}: {available:.1f} GiB free, "
            f"{needed:.1f} GiB needed.\n"
            "Set QWENK_MODEL_DIR to a larger filesystem."
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"   downloading {config.MODEL_FILE} (~{config.MIN_MODEL_GIB:.0f} GiB)")
    print("   this takes a few minutes on a fast connection")
    _download(config.model_url(), path)

    valid, detail = validate(path)
    if not valid:
        raise SystemExit(f"Downloaded model failed validation: {detail}")

    ok(f"downloaded and validated ({detail})")
    return path


def _download(url: str, dest: Path) -> None:
    """Fetch `url` to `dest`, resuming a partial transfer if one exists.

    Downloads to a `.part` sibling so an interrupted run can never leave a
    truncated file that looks like a finished model.
    """
    part = dest.parent / (dest.name + ".part")

    if shutil.which("curl") is None:
        raise SystemExit(
            "curl was not found on PATH and is required to download the model.\n"
            f"Download it manually and place it at {dest}:\n  {url}"
        )

    subprocess.run(
        [
            "curl",
            "--location",
            "--fail",
            "--retry", "5",
            "--retry-delay", "3",
            "--continue-at", "-",
            "--progress-bar",
            "--output", str(part),
            url,
        ],
        check=True,
    )

    os.replace(part, dest)


def load(path: Path):
    """Load the GGUF across the available GPUs and return the Llama handle."""
    step("Loading model")

    from llama_cpp import Llama

    gpus = gpu_count()
    tensor_split = config.TENSOR_SPLIT

    # A tensor_split longer than the number of visible GPUs makes llama.cpp
    # address a device that is not there.
    if tensor_split and gpus and len(tensor_split) != gpus:
        warn(
            f"TENSOR_SPLIT has {len(tensor_split)} entries but {gpus} GPU(s) "
            f"are visible; splitting evenly instead"
        )
        tensor_split = [1.0] * gpus if gpus > 1 else None

    print(f"   {path.name}")
    print(f"   context {config.CTX_SIZE}, gpu_layers {config.N_GPU_LAYERS}, "
          f"tensor_split {tensor_split}")
    print("   this takes a few minutes")

    threads = max(4, os.cpu_count() or 4)

    llm = Llama(
        model_path=str(path),
        n_gpu_layers=config.N_GPU_LAYERS,
        split_mode=1,  # split by layer across GPUs
        tensor_split=tensor_split,
        n_ctx=config.CTX_SIZE,
        n_batch=config.N_BATCH,
        n_ubatch=config.N_UBATCH,
        offload_kqv=config.OFFLOAD_KQV,
        n_threads=threads,
        n_threads_batch=threads,
        verbose=False,
    )

    ok("model loaded")

    summary = gpu_summary()
    if summary:
        for line in summary.splitlines():
            print(f"   gpu    : {line}")

    return llm
