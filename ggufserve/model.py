"""Acquiring and loading the GGUF.

The download deliberately uses curl against the public `resolve` URL rather than
`huggingface_hub.hf_hub_download`. Pulling the Hub into the process was the
original source of a broken `huggingface_hub` import once Gradio upgraded it
mid-session.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from ggufserve import config
from ggufserve.system import (
    free_disk_gib,
    gpu_count,
    gpu_stats,
    heartbeat,
    human_size,
    info,
    ok,
    step,
    warn,
)

GGUF_MAGIC = b"GGUF"

# ggml type ids, from llama_cpp.GGML_TYPE_*. Hardcoded so the mapping is
# readable here rather than hidden behind an import of the CUDA extension.
KV_CACHE_TYPES = {"f16": 1, "q8_0": 8}


def remote_size(url: str) -> int | None:
    """The size the server reports for `url`, or None if it cannot be determined.

    This is what lets validation work for any model without anyone having to
    look up a byte count. Best-effort by design: offline runs and hosts that
    omit `Content-Length` simply skip the size check.
    """
    if shutil.which("curl") is None:
        return None

    proc = subprocess.run(
        ["curl", "--silent", "--head", "--location", "--max-time", "30", url],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        return None

    # Follow the redirect chain and keep the last Content-Length, which belongs
    # to the file itself rather than to a redirect stub.
    size = None
    for line in proc.stdout.splitlines():
        match = re.match(r"^\s*content-length\s*:\s*(\d+)\s*$", line, re.IGNORECASE)
        if match:
            size = int(match.group(1))
    return size


def validate(path: Path, expected_bytes: int | None = None) -> tuple[bool, str]:
    """Check that `path` looks like a complete GGUF file."""
    if not path.exists():
        return False, "not found"

    size = path.stat().st_size
    human = f"{size / 1024**3:.2f} GiB"

    if size == 0:
        return False, "empty file"

    with path.open("rb") as handle:
        if handle.read(4) != GGUF_MAGIC:
            return False, "not a GGUF file (bad magic bytes)"

    if expected_bytes is not None and size != expected_bytes:
        return False, (
            f"incomplete ({human} of {expected_bytes / 1024**3:.2f} GiB)"
        )

    return True, human


def find_attached_model(
    filename: str, root: Path | None = None, dataset: str | None = None
) -> Path | None:
    """A matching GGUF already attached as a Kaggle dataset, if there is one.

    Datasets land under `/kaggle/input/<slug>/` and survive session restarts,
    which is the only practical way to keep a ~19 GB file on Kaggle — working
    disk is capped at 20 GB and `/tmp` is wiped.

    `dataset` (or `GGUF_SERVE_KAGGLE_DATASET`) pins the search to one slug.
    Leave it unset to use any attached copy whose filename matches.
    """
    if dataset is None:
        dataset = (os.environ.get("GGUF_SERVE_KAGGLE_DATASET") or "").strip() or None
    base = root if root is not None else Path("/kaggle/input")
    search = base / dataset if dataset else base
    if not search.is_dir():
        return None
    for candidate in sorted(search.rglob(filename)):
        valid, _ = validate(candidate)
        if valid:
            return candidate
    # Named dataset whose single GGUF was uploaded under a different filename.
    if dataset:
        ggufs = [path for path in sorted(search.rglob("*.gguf")) if validate(path)[0]]
        if len(ggufs) == 1:
            return ggufs[0]
    return None


def _download_dest(path: Path) -> Path:
    """Keep downloads out of read-only Kaggle input mounts."""
    kaggle_input = Path("/kaggle/input")
    try:
        path.resolve().relative_to(kaggle_input.resolve())
    except ValueError:
        return path
    fallback = Path("/tmp/gguf-serve/models") / path.name
    warn(f"{path.parent} is a read-only Kaggle input; downloading to {fallback.parent}")
    return fallback


def acquire() -> Path:
    """Return a validated model path, downloading it only if necessary."""
    step("Locating model")

    path = config.model_path()
    url = config.model_url()

    info(config.MODEL_FILE)

    expected = remote_size(url)
    if expected:
        info(f"expected size {expected / 1024**3:.2f} GiB (from the server)")

    attached = find_attached_model(config.MODEL_FILE)
    if attached is not None:
        ok(f"using attached dataset copy at {attached}")
        return attached

    valid, detail = validate(path, expected)
    if valid:
        ok(f"already present at {path} ({detail})")
        if expected is None:
            info("note: could not reach the server, so size was not verified")
        return path

    info(f"{path}: {detail}")
    path = _download_dest(path)

    if path.exists():
        info("removing the unusable file before retrying")
        path.unlink()

    if expected:
        needed = expected / 1024**3 * 1.05
        available = free_disk_gib(path.parent)
        if available < needed:
            raise SystemExit(
                f"Not enough disk space in {path.parent}: "
                f"{available:.1f} GiB free, {needed:.1f} GiB needed.\n"
                "Set GGUF_SERVE_MODEL_DIR or pass --model-dir to use a larger "
                "filesystem."
            )

    path.parent.mkdir(parents=True, exist_ok=True)

    info("downloading; this takes a few minutes on a fast connection")
    _download(url, path, expected)

    valid, detail = validate(path, expected)
    if not valid:
        raise SystemExit(f"Downloaded model failed validation: {detail}")

    ok(f"downloaded and validated ({detail})")
    return path


def _report_download(proc, part: Path, expected: int | None, interval: float = 15.0) -> None:
    """Print how far a running download has got, then wait for it."""
    while True:
        try:
            proc.wait(timeout=interval)
            return
        except subprocess.TimeoutExpired:
            done = part.stat().st_size if part.exists() else 0
            if expected:
                info(f"{human_size(done)} of {human_size(expected)} "
                     f"({done / expected * 100:.0f}%)")
            else:
                info(human_size(done))


def _download(url: str, dest: Path, expected: int | None = None) -> None:
    """Fetch `url` to `dest`, resuming a partial transfer if one exists.

    Downloads to a `.part` sibling and renames on success, so an interrupted run
    can never leave a truncated file that looks like a finished model.
    """
    if shutil.which("curl") is None:
        raise SystemExit(
            "curl was not found on PATH and is required to download the model.\n"
            f"Fetch it manually and place it at {dest}:\n  {url}"
        )

    part = dest.parent / (dest.name + ".part")

    # curl's own progress bar redraws itself with carriage returns, which a
    # notebook renders as one accumulating line of garbage rather than a bar.
    # Reporting from the growing .part file gives clean, log-shaped output.
    proc = subprocess.Popen(
        [
            "curl",
            "--location",
            "--fail",
            "--retry", "5",
            "--retry-delay", "3",
            "--continue-at", "-",
            "--no-progress-meter",
            "--output", str(part),
            url,
        ]
    )

    try:
        _report_download(proc, part, expected)
    except KeyboardInterrupt:
        proc.terminate()
        raise

    if proc.returncode != 0:
        raise SystemExit(
            f"curl exited with code {proc.returncode}. The partial download is "
            f"kept at {part}, so re-running resumes rather than starting over."
        )

    os.replace(part, dest)


def load_failure_hint(detail: str) -> str:
    """Explain a failed load in terms of the setting that causes it.

    Worth spelling out because the cause is nearly always VRAM, which llama.cpp
    reports in a log suppressed unless VERBOSE is on — so by default the load
    dies saying only that it failed. The settings themselves are not repeated
    here; this step already printed them.
    """
    if config.KV_CACHE_TYPE == "f16":
        remedy = 'set KV_CACHE = "q8_0", which halves it'
    else:
        remedy = f"lower CTX below {config.CTX_SIZE:,}"

    return "\n".join(
        [
            "",
            "  [x] the model could not be loaded",
            f"  llama.cpp said: {detail}",
            "",
            "  The KV cache is allocated for the whole context up front, so this",
            f"  is nearly always VRAM. To fit, {remedy}.",
            "  docs/configuration.md lists what each context length costs.",
            "  Set VERBOSE = True to see llama.cpp's own error.",
            "",
        ]
    )


def load(path: Path):
    """Load the GGUF across the available GPUs and return the Llama handle."""
    step("Loading model")

    # Checked before the expensive llama_cpp import so a bad value in
    # config.py reports itself instead of surfacing as an import error.
    kv_type = config.KV_CACHE_TYPE
    if kv_type not in KV_CACHE_TYPES:
        raise SystemExit(
            f"Unknown KV_CACHE_TYPE {kv_type!r}. "
            f"Choose one of: {', '.join(KV_CACHE_TYPES)}"
        )

    from llama_cpp import Llama

    gpus = gpu_count()
    tensor_split = config.TENSOR_SPLIT

    # A tensor_split longer than the number of visible GPUs makes llama.cpp
    # address a device that is not there.
    if tensor_split and gpus and len(tensor_split) != gpus:
        warn(
            f"TENSOR_SPLIT has {len(tensor_split)} entries but {gpus} GPU(s) "
            "are visible; splitting evenly instead"
        )
        tensor_split = [1.0] * gpus if gpus > 1 else None

    print(f"   {path.name}")
    print(
        f"   context {config.CTX_SIZE:,}, gpu_layers {config.N_GPU_LAYERS}, "
        f"tensor_split {tensor_split}, kv cache {kv_type}"
    )
    # Determine thread counts for llama.cpp
    if config.N_THREADS is not None:
        n_threads = config.N_THREADS
        n_threads_batch = config.N_THREADS_BATCH if config.N_THREADS_BATCH is not None else config.N_THREADS
    else:
        n_threads = max(4, os.cpu_count() or 4)
        n_threads_batch = n_threads

    extra = {}
    if kv_type != "f16":
        ggml_type = KV_CACHE_TYPES[kv_type]
        # llama.cpp can only read a quantized KV cache through flash attention,
        # so selecting one implies enabling it.
        extra = {
            "type_k": ggml_type,
            "type_v": ggml_type,
            "flash_attn": True,
        }

    # llama.cpp is silent while it reads the weights, so without a heartbeat
    # this looks identical to a hang for several minutes.
    try:
        with heartbeat("loading weights onto the GPU"):
            llm = Llama(
                model_path=str(path),
                n_gpu_layers=config.N_GPU_LAYERS,
                split_mode=1,  # split by layer across GPUs
                tensor_split=tensor_split,
                n_ctx=config.CTX_SIZE,
                n_batch=config.N_BATCH,
                n_ubatch=config.N_UBATCH,
                offload_kqv=config.OFFLOAD_KQV,
                n_threads=n_threads,
                n_threads_batch=n_threads_batch,
                verbose=config.VERBOSE,
                **extra,
            )
    except Exception as error:
        print(load_failure_hint(str(error) or type(error).__name__), flush=True)
        raise SystemExit(1)

    ok("model loaded")

    # Shows how much VRAM was actually claimed, which is how you spot llama.cpp
    # having quietly spilled layers into system RAM.
    for gpu in gpu_stats():
        print(f"   gpu {gpu.index}  : {gpu.describe()}")

    return llm
