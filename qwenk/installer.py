"""Dependency setup, skipped entirely when the environment is already correct.

Originally cells "02 - Install/verify llama-cpp-python", "09 - Install/verify
Gradio web stack", "10 - Clean-import test for Gradio" and "11 - Verify the
Hugging Face issue is not active".
"""

from __future__ import annotations

import importlib.metadata as metadata
import subprocess
import sys

from qwenk import config
from qwenk.system import ok, step, warn


def _pip(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", *args], check=True)


def _installed(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def ensure_llama_cpp() -> None:
    """Install a CUDA-enabled llama-cpp-python, unless one is already present.

    `--only-binary=:all:` is load-bearing. Without it pip silently falls back to
    building from source, which takes 20+ minutes and produces a CPU-only build
    that never touches the GPU.
    """
    step("Checking llama-cpp-python")

    if _gpu_offload_available():
        ok(f"CUDA build already installed (llama-cpp-python {_installed('llama-cpp-python')})")
        return

    print("   installing the pre-built CUDA wheel (~1.7 GB, takes a few minutes)")
    _pip(
        "--upgrade",
        "--only-binary=:all:",
        "llama-cpp-python",
        "--extra-index-url",
        config.LLAMA_CPP_WHEEL_INDEX,
    )

    if not _gpu_offload_available():
        raise SystemExit(
            "llama-cpp-python imported but reports no CUDA GPU offload support.\n"
            "Refusing to continue, because the model would load onto the CPU and\n"
            "run far too slowly to be usable.\n"
            "See docs/troubleshoot.md for how to fix the wheel selection."
        )

    ok(f"CUDA build installed (llama-cpp-python {_installed('llama-cpp-python')})")


def _gpu_offload_available() -> bool:
    try:
        import llama_cpp
    except Exception:
        return False
    try:
        return bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:
        return False


def ensure_web_stack() -> None:
    """Pin the Gradio versions that are known to work together.

    Deliberately does not touch FastAPI or Starlette: Gradio manages its own
    web-server dependencies, and pinning them here breaks Kaggle's preinstalled
    environment.
    """
    step("Checking Gradio")

    wanted = {
        "gradio": config.GRADIO_VERSION,
        "gradio_client": config.GRADIO_CLIENT_VERSION,
    }

    missing = []
    for package, want in wanted.items():
        have = _installed(package)
        print(f"   {package}: {have or 'not installed'} (want {want})")
        if have != want:
            missing.append(f"{package}=={want}")

    if not missing:
        ok("web stack already correct")
        return

    print(f"   installing {' '.join(missing)}")
    _pip("--upgrade", "--no-cache-dir", *missing)
    ok("web stack installed")


def import_gradio():
    """Import Gradio, retrying once after evicting stale huggingface_hub modules.

    Installing Gradio also upgrades huggingface_hub. If an older version was
    already imported in this process the two disagree about their own internals
    and the import fails; dropping the cached modules lets the retry pick up the
    freshly installed package. Only reachable when running inside a long-lived
    process such as a notebook kernel.
    """
    try:
        import gradio
    except ImportError as first_error:
        warn(f"Gradio import failed ({first_error}); retrying with a clean cache")

        for name in list(sys.modules):
            if name == "huggingface_hub" or name.startswith("huggingface_hub."):
                del sys.modules[name]

        try:
            import gradio
        except ImportError as second_error:
            raise SystemExit(
                f"Gradio could not be imported: {second_error}\n"
                "Restart the runtime and re-run launch.py. The model file on "
                "disk is unaffected, so only the load is repeated.\n"
                "See docs/troubleshoot.md."
            ) from second_error

    ok(f"gradio {gradio.__version__}")
    return gradio


def setup() -> None:
    ensure_llama_cpp()
    ensure_web_stack()
