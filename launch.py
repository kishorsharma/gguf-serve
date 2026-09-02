#!/usr/bin/env python3
"""gguf-serve entry point: install, download, load, serve.

    python launch.py

Every step checks its own state first, so re-running after a crash or a runtime
restart skips whatever is already done.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python launch.py` from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ggufserve import config
from ggufserve.version import version


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="launch.py",
        description=(
            "Serve a GGUF model with an OpenAI-compatible public API. "
            "Defaults to Qwen3.8-27B on two 16 GB GPUs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    model = parser.add_argument_group("model")
    model.add_argument(
        "--model-file",
        default=config.MODEL_FILE,
        help="the .gguf filename to serve",
    )
    model.add_argument(
        "--model-repo",
        default=config.MODEL_REPO,
        help="Hugging Face repo holding that file",
    )
    model.add_argument(
        "--model-url",
        default=config.MODEL_URL,
        help="download from this URL instead of Hugging Face",
    )
    model.add_argument(
        "--model-dir",
        type=Path,
        default=config.MODEL_DIR,
        help="directory the .gguf is stored in",
    )
    model.add_argument(
        "--ctx",
        type=int,
        default=config.CTX_SIZE,
        help="context window in tokens; lower it if the model will not fit",
    )
    model.add_argument(
        "--gpu-layers",
        type=int,
        default=config.N_GPU_LAYERS,
        help="layers to offload to the GPU; -1 means all of them",
    )

    server = parser.add_argument_group("server")
    server.add_argument(
        "--port",
        type=int,
        default=config.SERVER_PORT,
        help="port to listen on; the next free port is used if it is taken",
    )
    server.add_argument(
        "--no-share",
        action="store_true",
        help="do not create a public gradio.live URL (local access only)",
    )

    behaviour = parser.add_argument_group("behaviour")
    behaviour.add_argument(
        "--no-reasoning",
        action="store_true",
        help="pass model output through untouched instead of splitting off "
        "any </think> reasoning section",
    )
    behaviour.add_argument(
        "--skip-install",
        action="store_true",
        help="assume dependencies are already correct",
    )
    behaviour.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="start serving without first checking that the model generates",
    )
    behaviour.add_argument(
        "--download-only",
        action="store_true",
        help="fetch and validate the model, then exit without loading it",
    )

    parser.add_argument("--version", action="version", version=f"gguf-serve {version}")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Applied before any other module reads them.
    config.MODEL_FILE = args.model_file
    config.MODEL_REPO = args.model_repo
    config.MODEL_URL = args.model_url
    config.MODEL_DIR = args.model_dir
    config.CTX_SIZE = args.ctx
    config.N_GPU_LAYERS = args.gpu_layers
    if args.no_reasoning:
        config.PARSE_REASONING = False

    from ggufserve import chat, installer, model, server, system

    system.banner(f"  gguf-serve {version}  —  {config.model_id()}")

    system.report(config.MODEL_DIR)

    if args.skip_install:
        system.step("Skipping dependency check (--skip-install)")
    else:
        installer.setup()

    model_file = model.acquire()

    if args.download_only:
        system.ok(f"model ready at {model_file}; exiting (--download-only)")
        return

    llm = model.load(model_file)

    if not args.skip_smoke_test:
        chat.smoke_test(llm)

    app = server.build(llm)
    port = server.pick_port(args.port)
    server.launch(
        app,
        port=port,
        share=not args.no_share,
        model_path=model_file,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
