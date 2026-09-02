#!/usr/bin/env python3
"""Qwen-K entry point: install, download, load, serve.

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

from qwenk import config
from qwenk.version import version


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="launch.py",
        description="Serve Qwen3.8-27B with an OpenAI-compatible public API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--port",
        type=int,
        default=config.SERVER_PORT,
        help="port to listen on; the next free port is used if it is taken",
    )
    parser.add_argument(
        "--no-share",
        action="store_true",
        help="do not create a public gradio.live URL (local access only)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=config.MODEL_DIR,
        help="directory holding the .gguf file",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=config.CTX_SIZE,
        help="context window in tokens; lower it if the model will not fit",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="assume dependencies are already correct",
    )
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="start serving without first checking that the model generates",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="fetch and validate the model, then exit without loading it",
    )
    parser.add_argument("--version", action="version", version=f"Qwen-K {version}")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Applied before anything reads them.
    config.MODEL_DIR = args.model_dir
    config.CTX_SIZE = args.ctx

    from qwenk import chat, installer, model, server, system

    system.banner(f"  Qwen-K {version}  —  {config.MODEL_FILE}")

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
    server.launch(app, port=port, share=not args.no_share)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
