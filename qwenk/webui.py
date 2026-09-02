"""Routes that serve the browser chat UI from `web/`.

Originally cells "13 - Browser chat UI" and "14 - Root redirect". The HTML, CSS
and JavaScript now live in `web/` as editable files instead of one long Python
string.
"""

from __future__ import annotations

from fastapi.responses import FileResponse, HTMLResponse

from qwenk.config import REPO_ROOT
from qwenk.system import ok, step

WEB_DIR = REPO_ROOT / "web"

_CONTENT_TYPES = {".css": "text/css", ".js": "text/javascript"}


def register(app) -> None:
    """Serve the UI at `/` and `/chat`, with its assets under `/assets`."""
    step("Registering web UI")

    index = WEB_DIR / "index.html"
    if not index.exists():
        raise SystemExit(f"Web UI missing: {index}")

    @app.get("/assets/{filename}", include_in_schema=False)
    async def asset(filename: str):
        target = (WEB_DIR / filename).resolve()

        # Keep `..` in the URL from reaching outside web/.
        if WEB_DIR.resolve() not in target.parents or not target.is_file():
            return HTMLResponse("not found", status_code=404)

        return FileResponse(
            target,
            media_type=_CONTENT_TYPES.get(target.suffix, "application/octet-stream"),
        )

    @app.get("/chat", response_class=HTMLResponse, include_in_schema=False)
    async def chat_page():
        return HTMLResponse(index.read_text(encoding="utf-8"))

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def root():
        return HTMLResponse(index.read_text(encoding="utf-8"))

    # gradio.Server registers its own `/` when it is constructed, and FastAPI
    # matches routes in registration order, so ours would never be reached.
    # Moving it to the front makes the public URL open the chat UI directly.
    _promote_root(app)

    ok("UI at / and /chat")


def _promote_root(app) -> None:
    routes = app.router.routes
    for index, route in enumerate(routes):
        if getattr(route, "path", None) == "/" and getattr(route, "name", "") == "root":
            routes.insert(0, routes.pop(index))
            return
