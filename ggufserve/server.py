"""Builds and launches the single public server.

Everything is served by one `gradio.Server`, which is a FastAPI application.
Because the OpenAI routes, the chat UI and the share tunnel all belong to that
one app, they share a single public origin: there is no second process and no
second tunnel to expose.
"""

from __future__ import annotations

import socket
import time
from pathlib import Path

from ggufserve import api, config, installer, webui
from ggufserve.system import gpu_stats, human_size, ok, ram_stats, step, warn


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


def launch(app, port: int, share: bool, model_path: Path | None = None) -> None:
    """Start serving, print a summary, and block until interrupted."""
    step("Starting server")

    result = app.launch(
        server_name=config.SERVER_NAME,
        server_port=port,
        share=share,
        show_error=True,
        # Return control so we can print our own summary and own the wait loop.
        prevent_thread_lock=True,
    )

    print_summary(port=port, share_url=_share_url(result), model_path=model_path)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down.")


def _share_url(launch_result) -> str | None:
    """Pull the public URL out of whatever `Server.launch()` handed back.

    Gradio returns `(app, local_url, share_url)`, with `share_url` set to None
    when sharing is off or the tunnel failed. Read defensively: this runs after
    a model load that can take fifteen minutes, and a shape change in Gradio
    should cost us a nice summary, not the whole server.
    """
    try:
        if isinstance(launch_result, tuple) and len(launch_result) == 3:
            share_url = launch_result[2]
            if isinstance(share_url, str) and share_url.startswith("http"):
                return share_url.rstrip("/")
    except Exception:
        pass
    return None


def print_summary(
    port: int,
    share_url: str | None = None,
    model_path: Path | None = None,
) -> None:
    """The everything-you-need block printed once the server is up."""
    local_url = f"http://127.0.0.1:{port}"
    public = share_url or local_url

    line = "=" * 70
    print(f"\n{line}\n  gguf-serve is running\n{line}\n")

    if share_url:
        # Front and centre, on its own line, so it can be copied in one go.
        print(f"  PUBLIC URL   {share_url}")
    else:
        print(f"  LOCAL URL    {local_url}")
        print("               (no public URL: sharing is disabled)")

    print()
    print(f"  chat UI      {public}/")
    print(f"  API base     {public}/v1")
    print(f"  API docs     {public}/docs")
    if share_url:
        print(f"  local        {local_url}/")

    print()
    print(f"  model        {config.model_id()}")

    size = ""
    if model_path is not None and model_path.exists():
        size = f"  ({human_size(model_path.stat().st_size)})"
    print(f"  file         {config.MODEL_FILE}{size}")
    print(f"  context      {config.CTX_SIZE:,} tokens (KV cache {config.KV_CACHE_TYPE})")
    print(
        "  reasoning    "
        + ("</think> sections split off" if config.PARSE_REASONING else "not parsed")
    )

    print()
    for gpu in gpu_stats():
        print(f"  GPU {gpu.index}        {gpu.describe()}")

    ram = ram_stats()
    if ram:
        used, total = ram
        print(f"  RAM          {used:.1f} / {total:.1f} GiB ({used / total * 100:.0f}%)")

    print()
    print("  Point any OpenAI client at:")
    print(f'    base_url = "{public}/v1"')
    print(f'    model    = "{config.model_id()}"')
    print('    api_key  = "not-used"')

    if share_url:
        print()
        print("  Anyone with the public URL can use this model. The link lasts")
        print("  only while this process runs (at most a week).")

    print(line)
    print("\nPress Ctrl+C (or stop the notebook cell) to shut down.\n")
