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


def _format_split(split: list[float] | None) -> str:
    return "none" if not split else ",".join(str(part) for part in split)


def _parse_split(text: str) -> list[float] | None:
    """Read a tensor split, where a single GPU is spelled `none`."""
    text = text.strip()
    if not text or text.lower() in {"none", "off"}:
        return None
    try:
        parts = [float(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise SystemExit(f"--tensor-split: {text!r} is not a comma-separated list of numbers")
    if not parts or any(part <= 0 for part in parts):
        raise SystemExit(f"--tensor-split: {text!r} must be positive numbers, e.g. 1,1")
    return parts


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
        "--model",
        metavar="URL_OR_ID",
        help="what to serve: a Hugging Face file URL (the address bar of the "
        ".gguf page works), an owner/repo id, a direct download URL, or a "
        ".gguf filename. Sets --model-repo, --model-file and --model-url "
        "together",
    )
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
    model.add_argument(
        "--kv-cache-type",
        choices=["f16", "q8_0"],
        default=config.KV_CACHE_TYPE,
        help="KV cache precision; q8_0 halves the memory a long context needs",
    )
    model.add_argument(
        "--tensor-split",
        default=_format_split(config.TENSOR_SPLIT),
        help="how to divide the model across GPUs, e.g. 1,1 for a matched pair "
        "or 1,2 for 16 GB plus 32 GB; 'none' for a single GPU",
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
        "--verbose",
        action="store_true",
        help="let llama.cpp print its own log; the only place allocation "
        "failures are reported, so use it when a load dies without saying why",
    )
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


def _count_steps(args) -> int:
    """How many numbered steps this run will print, so `[3/8]` is honest.

    Kept next to main() because it has to mirror the step() calls below; a
    mismatch only skews the labels, it does not break the run.
    """
    total = 1  # host check
    if not args.skip_install:
        total += 2  # llama-cpp-python, Gradio
    total += 1  # locate model
    if not args.download_only:
        total += 1  # load model
        if not args.skip_smoke_test:
            total += 1
        total += 2  # build server, start server
    return total


def _apply_model_arg(args) -> None:
    """Fold a single `--model` value into the individual model settings.

    Explicit --model-repo/--model-file/--model-url still win, so `--model
    owner/repo --model-file x.gguf` behaves the way it reads.
    """
    try:
        source = config.parse_source(args.model)
    except ValueError as error:
        raise SystemExit(f"--model: {error}")

    if source.repo and args.model_repo == config.MODEL_REPO:
        args.model_repo = source.repo
    if source.filename and args.model_file == config.MODEL_FILE:
        args.model_file = source.filename
    if source.url and args.model_url == config.MODEL_URL:
        args.model_url = source.url

    # A repo with no file named is the one case we cannot guess at: repos hold a
    # dozen quantizations that differ by 10 GiB, so picking one would be a coin
    # flip on whether it even fits.
    if source.repo and not source.filename and args.model_file == config.MODEL_FILE:
        raise SystemExit(
            f"--model: {source.repo} is a repo, not a file. Open it on Hugging "
            "Face, click the .gguf you want, and pass that page's URL — or add "
            "--model-file <name>.gguf."
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.model:
        _apply_model_arg(args)

    # Applied before any other module reads them.
    config.MODEL_FILE = args.model_file
    config.MODEL_REPO = args.model_repo
    config.MODEL_URL = args.model_url
    config.MODEL_DIR = args.model_dir
    config.CTX_SIZE = args.ctx
    config.N_GPU_LAYERS = args.gpu_layers
    config.KV_CACHE_TYPE = args.kv_cache_type
    config.TENSOR_SPLIT = _parse_split(args.tensor_split)
    config.VERBOSE = args.verbose
    if args.no_reasoning:
        config.PARSE_REASONING = False

    from ggufserve import chat, installer, model, server, system

    system.banner(f"  gguf-serve {version}  —  {config.model_id()}")
    system.set_total_steps(_count_steps(args))

    system.report(config.MODEL_DIR)

    if args.skip_install:
        system.info("skipping dependency check (--skip-install)")
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
