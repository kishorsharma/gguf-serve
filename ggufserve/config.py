"""All tunable settings for gguf-serve. This is the only file most people edit.

The defaults serve Qwen3.8-27B on two 16 GB GPUs, which is the combination this
project is verified against. Any GGUF that llama.cpp can load works the same
way — change `MODEL_REPO` and `MODEL_FILE` and everything else adapts.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# A Hugging Face repo and one file inside it. The file has to be a single
# self-contained .gguf; multi-part splits are not handled.
MODEL_REPO = "unsloth/Qwen3.8-27B-GGUF"
MODEL_FILE = "Qwen3.8-27B-UD-Q5_K_XL.gguf"

# Set this to fetch from somewhere other than Hugging Face. When None the URL is
# built from MODEL_REPO and MODEL_FILE.
MODEL_URL: str | None = None

# The name this server advertises over the API, and what clients put in their
# `model` field. When None it is derived from MODEL_FILE, so it always tracks
# whatever you are actually serving.
MODEL_ID: str | None = None

# Where the .gguf lives. Kaggle's /kaggle/working is capped at 20 GB and counts
# against the notebook output quota, so the default is the much larger scratch
# overlay. Scratch is wiped when the runtime restarts; set the environment
# variable (or edit this line) to keep the download across restarts.
MODEL_DIR = Path(os.environ.get("GGUF_SERVE_MODEL_DIR") or "/tmp/gguf-serve/models")

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
CTX_SIZE = 16384
N_BATCH = 512
N_UBATCH = 512

# KV cache precision, "f16" or "q8_0".
#
# The cache is allocated for the full context up front and is what makes long
# contexts expensive. "q8_0" halves its size for a small quality cost, which is
# what makes the longest contexts fit at all — see docs/configuration.md for the
# numbers. It needs a llama.cpp build with flash attention for your GPU, which
# gguf-serve enables automatically when you select it.
KV_CACHE_TYPE = "f16"

# -1 offloads every layer to the GPU. Lower it to keep some layers on the CPU
# when a model is slightly too big for your VRAM.
N_GPU_LAYERS = -1

# How to divide the model across GPUs. `None` means a single GPU. `[1.0, 1.0]`
# splits evenly across two, which is right for a matched pair; weight it by
# VRAM for a mismatched one, e.g. `[1.0, 2.0]` for 16 GB plus 32 GB.
TENSOR_SPLIT: list[float] | None = [1.0, 1.0]

# Keep the KV cache on the GPU.
OFFLOAD_KQV = True

# Sampling defaults, applied when a request does not override them.
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 20
MAX_TOKENS = 2048

# ---------------------------------------------------------------------------
# Reasoning models
# ---------------------------------------------------------------------------
# Reasoning models (Qwen3, DeepSeek-R1, QwQ and friends) emit a scratchpad and
# close it with `</think>` before the real answer. When True the server splits
# that off so clients get clean answers.
#
# Harmless for ordinary models, which simply never emit the tag, but you can set
# it False to pass output through completely untouched.
PARSE_REASONING = True

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
SERVER_NAME = "0.0.0.0"
SERVER_PORT = 7860

# Create a public *.gradio.live URL. This is the whole point of running on a
# hosted notebook, so it defaults on. Disable with `--no-share`.
SHARE = True

# ---------------------------------------------------------------------------
# Dependency pins
# ---------------------------------------------------------------------------
# Verified working together on Kaggle. Gradio 6 is required for `gradio.Server`,
# which is what lets the OpenAI routes and the public share URL live on one
# origin instead of needing two tunnels.
GRADIO_VERSION = "6.26.0"
GRADIO_CLIENT_VERSION = "2.6.1"

# Pre-built CUDA wheel index for llama-cpp-python. cu125 wheels run fine on the
# CUDA 12.8 driver that Kaggle ships.
LLAMA_CPP_WHEEL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cu125"


def model_url() -> str:
    if MODEL_URL:
        return MODEL_URL
    return f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}?download=true"


def model_path() -> Path:
    return MODEL_DIR / MODEL_FILE


def model_id() -> str:
    """The API-facing model name, derived from the filename unless overridden."""
    if MODEL_ID:
        return MODEL_ID
    return derive_model_id(MODEL_FILE)


def derive_model_id(filename: str) -> str:
    """Turn a GGUF filename into a lowercase, URL-safe model id.

    `Qwen3.8-27B-UD-Q5_K_XL.gguf` becomes `qwen3.8-27b-ud-q5-k-xl`.
    """
    stem = re.sub(r"\.gguf$", "", filename, flags=re.IGNORECASE)
    stem = stem.replace("_", "-").lower()
    return re.sub(r"-+", "-", stem).strip("-")
