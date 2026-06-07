"""Launcher for the bundled SymbolicDSGE web UI.

Serves the prebuilt frontend (``SymbolicDSGE/ui/_static``) and the FastAPI
backend from a single origin, then opens a browser. Installed as the
``sdsge-ui`` console script and reused by :meth:`SolvedModel.serve`.
"""

from __future__ import annotations

import argparse
import socket
import threading
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from SymbolicDSGE.core.solved_model import SolvedModel

_STATIC_DIR = Path(__file__).parent / "_static"

_MISSING_UI_DEPS = (
    "The SymbolicDSGE UI extra is not installed. "
    "Install it with: pip install 'SymbolicDSGE[ui]'"
)
_MISSING_ASSETS = (
    "Bundled frontend assets were not found at {path}. "
    "This usually means you are running from a source checkout that was not "
    "built. Build the frontend with: python scripts/build_frontend.py"
)


def _free_port(host: str = "127.0.0.1") -> int:
    """Return an OS-assigned free TCP port on *host*."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def run_server(
    *,
    reference: "SolvedModel | None" = None,
    dgp: "SolvedModel | None" = None,
    host: str = "127.0.0.1",
    port: int | None = None,
    open_browser: bool = True,
) -> None:
    """Build the app, mount the bundled frontend, and run uvicorn (blocking).

    Parameters
    ----------
    reference, dgp:
        Optional pre-solved models to preload into the session.
    host, port:
        Bind address; ``port`` defaults to an OS-assigned free port.
    open_browser:
        Whether to open the default browser at the served URL.
    """
    try:
        import uvicorn
        from fastapi.staticfiles import StaticFiles
        from starlette.types import Receive, Scope, Send
    except ImportError as exc:  # pragma: no cover - exercised without [ui]
        raise SystemExit(_MISSING_UI_DEPS) from exc

    from .app import create_app

    if not _STATIC_DIR.is_dir():
        raise SystemExit(_MISSING_ASSETS.format(path=_STATIC_DIR))

    class _HttpOnlyStaticFiles(StaticFiles):
        """Serve files over HTTP; reject non-HTTP scopes cleanly.

        The catch-all mount at ``/`` also receives WebSocket connections, e.g.
        the frontend's dev-only ``/_lsp/python`` LSP socket, which has no
        backend in the bundled build. Plain ``StaticFiles`` asserts on a
        non-HTTP scope and dumps a traceback; here we close it quietly instead.
        """

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] != "http":
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1000})
                return
            await super().__call__(scope, receive, send)

    app = create_app(reference=reference, dgp=dgp)
    # Mounted last so it does not shadow the /api routes registered above.
    app.mount("/", _HttpOnlyStaticFiles(directory=_STATIC_DIR, html=True), name="ui")

    bind_port = port if port is not None else _free_port(host)
    url = f"http://{host}:{bind_port}"

    if open_browser:
        timer = threading.Timer(0.8, lambda: webbrowser.open(url))
        timer.daemon = True
        timer.start()

    print(f"SymbolicDSGE UI running at {url}  (Ctrl+C to stop)")
    uvicorn.run(app, host=host, port=bind_port)


def main(argv: Sequence[str] | None = None) -> None:
    """Console-script entry point (``sdsge-ui``)."""
    parser = argparse.ArgumentParser(
        prog="sdsge-ui",
        description="Launch the SymbolicDSGE web playground.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Bind port (default: an available port).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser window automatically.",
    )
    args = parser.parse_args(argv)

    run_server(host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
