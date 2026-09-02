"""Builds and launches the single public server.

Everything is served by one `gradio.Server`, which is a FastAPI application.
Because the OpenAI routes, the chat UI and the share tunnel all belong to that
one app, they share a single public origin: there is no second process and no
second tunnel to expose.
"""

from __future__ import annotations

import socket
import time

from ggufserve import api, config, installer, webui
from ggufserve.system import ok, step, warn


def _port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.connect_ex((host, port)) != 0


def pick_port(preferred: int, attempts: int = 10) -> int:
    """Return `preferred`, or the next free port after it."""
    for offset in range(attempts):
        candidate = preferred + offset
        if _port_is_free(candidate):
            if offset:
                warn(f"port {preferred} is in use; using {candidate}")
            return candidate
    raise SystemExit(
        f"No free port in range {preferred}-{preferred + attempts - 1}. "
        "Pass --port to choose another."
    )


def build(llm):
    """Create the FastAPI/Gradio app with all routes attached."""
    step("Building server")

    gradio = installer.import_gradio()

    try:
        Server = gradio.Server
    except AttributeError as error:
        raise SystemExit(
            f"This Gradio ({gradio.__version__}) has no `Server` class. "
            f"gguf-serve needs gradio {config.GRADIO_VERSION}, which is what "
            "puts the OpenAI routes and the share tunnel on one origin.\n"
            "Run `python launch.py` without --skip-install to fix this."
        ) from error

    app = Server(
        title=config.model_id(),
        summary=f"{config.MODEL_FILE} served through llama.cpp",
        description=(
            "OpenAI-compatible inference API.\n\n"
            "Point any OpenAI client at `<this origin>/v1` and use the model id "
            f"`{config.model_id()}`."
        ),
        version="1.0.0",
    )

    api.register(app, llm)
    webui.register(app)

    ok("routes ready")
    return app


def launch(app, port: int, share: bool) -> None:
    """Start serving and block until interrupted."""
    step("Starting server")

    app.launch(
        server_name=config.SERVER_NAME,
        server_port=port,
        share=share,
        show_error=True,
        # Return control so we can print our own summary and own the wait loop.
        prevent_thread_lock=True,
    )

    _print_endpoints(port, share)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down.")


def _print_endpoints(port: int, share: bool) -> None:
    local = f"http://127.0.0.1:{port}"

    print()
    print("=" * 70)
    print("  gguf-serve is running")
    print("=" * 70)
    print(f"  chat UI    {local}/")
    print(f"  API docs   {local}/docs")
    print(f"  base URL   {local}/v1")
    print(f"  model id   {config.model_id()}")
    if share:
        print()
        print("  The public https://....gradio.live URL is printed above.")
        print("  Append /v1 to it for the API, /docs for the docs.")
        print()
        print("  Anyone with that link can use this model. It stays up while")
        print("  this process runs, for up to 72 hours.")
    print("=" * 70)
    print("\nPress Ctrl+C (or stop the notebook cell) to shut down.\n")
