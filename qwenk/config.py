"""All tunable settings for Qwen-K. This is the only file most people edit.

Originally cell "00 - Configuration" of the notebook.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_REPO = "unsloth/Qwen3.8-27B-GGUF"
MODEL_FILE = "Qwen3.8-27B-UD-Q5_K_XL.gguf"

# The name this server advertises over the API. Point your OpenAI client at it.
MODEL_ID = "qwen3.8-27b-q5-k-xl"

# A complete download is 19.44 GiB. Anything smaller is a truncated transfer,
# so we refuse to load it.
MIN_MODEL_GIB = 19.0

# Where the .gguf lives. Kaggle's /kaggle/working is capped at 20 GB and counts
# against the notebook output quota, so the default is the much larger scratch
# overlay. Scratch is wiped when the runtime restarts; set QWENK_MODEL_DIR (or
# edit this line) to keep the download across restarts.
MODEL_DIR = Path(os.environ.get("QWENK_MODEL_DIR") or "/tmp/qwen-k/models")

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
CTX_SIZE = 16384
N_BATCH = 512
N_UBATCH = 512

# -1 offloads every layer to the GPU.
N_GPU_LAYERS = -1

# Equal split across two identical T4s. Set to None for a single GPU, or to
# weights like [1.0, 2.0] for a mismatched pair.
TENSOR_SPLIT: list[float] | None = [1.0, 1.0]

# Keep the KV cache on the GPU.
OFFLOAD_KQV = True

# Sampling defaults, applied when a request does not override them.
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 20
MAX_TOKENS = 2048

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
SERVER_NAME = "0.0.0.0"
SERVER_PORT = 7860

# Create a public *.gradio.live URL. This is the whole point of running on
# Kaggle, so it defaults on. Disable with `python launch.py --no-share`.
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
    return f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}?download=true"


def model_path() -> Path:
    return MODEL_DIR / MODEL_FILE
